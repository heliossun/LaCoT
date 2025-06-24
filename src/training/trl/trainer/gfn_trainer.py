# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
import torch.distributed as dist
from ..import_utils import is_peft_available, is_wandb_available
from .utils import (
    disable_dropout_in_model,
    pad_to_length,
    trl_sanitze_kwargs_for_tagging,
)
# import gfn parts
#from .gfn_rebuffer import load_replay_buffer
from .gfn_utils import generate, generate_and_return_eos_logprob, append_sol_and_remove_eos, run_evaluation

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    print("parameters: ",to_return)
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class GFNTrainer(Trainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy. For the IPO loss, beta is the regularization parameter denoted by tau in the paper.
        label_smoothing (`float`, defaults to 0):
            The robust DPO label smoothing parameter from the [cDPO](https://ericmitchell.ai/cdpo.pdf) report that should be between 0 and 0.5.
        loss_type (`str`, defaults to `"sigmoid"`):
            The type of DPO loss to use. Either `"sigmoid"` the default DPO loss,`"hinge"` loss from [SLiC](https://arxiv.org/abs/2305.10425) paper, `"ipo"` from [IPO](https://arxiv.org/abs/2310.12036) paper, or `"kto"` from the HALOs [report](https://github.com/ContextualAI/HALOs/blob/main/assets/report.pdf).
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_target_length (`int`, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        precompute_ref_log_probs (`bool`, defaults to `False`):
            Flag to precompute reference model log probabilities and evaluation datasets. This is useful if you want to train
            without the reference model and reduce the total GPU memory needed.
        dataset_num_proc (`Optional[int]`, *optional*):
            The number of workers to use to tokenize the data. Defaults to None.
        model_init_kwargs (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string
        model_adapter_name (`str`, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`str`, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
        reference_free (`bool`):
            If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.
    """

    _tag_names = ["trl", "gfn"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
        processor=None
    ):

        #print(processor)
        self.processor=processor
        tokenizer=processor.tokenizer
        # import pdb;pdb.set_trace()
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the DPOTrainer. But your model is already instantiated.")


        if isinstance(model, str):
            warnings.warn("You passed a model_id to the DPOTrainer. This will automatically create an " "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you.")
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)


        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if generate_during_eval and not is_wandb_available():
            raise ValueError("`generate_during_eval=True` requires Weights and Biases to be installed." " Please install `wandb` to resolve.")

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = model_adapter_name
        self.ref_adapter_name = ref_adapter_name
        self.reference_free = reference_free

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a DPO dataset.")
        if max_length is None:
            warnings.warn(
                "`max_length` is not set in the DPOTrainer's init" " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        if max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the DPOTrainer's init" " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128

        if max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the DPOTrainer's init" " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_target_length = 128



        if disable_dropout:
            disable_dropout_in_model(model)

        self.max_length = max_length
        self.generate_during_eval = generate_during_eval
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.args=args
        self.dataset_num_proc = dataset_num_proc

        #print(args)
        self.get_reward_temp = lambda x : args.reward_sched_start + (args.reward_sched_end - args.reward_sched_start) * min(1, x / args.reward_sched_horizon)
        self.get_reward_tolerate = lambda x : args.reward_tolarent_start + (args.reward_tolarent_end - args.reward_tolarent_start) * min(1, x / args.reward_tolarent_horizon)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if not hasattr(self, "accelerator"):
            raise AttributeError("Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`.")
        #self.reward_prompt = self.get_prompt_tokens()
        # Deepspeed Zero-3 does not support precompute_ref_log_probs

    def get_prompt_tokens(self):
        prompt_q = "Which country is highlighted?\nContext: N/A\nOptions: (A) Solomon Islands (B) Nauru (C) Vanuatu (D) Fiji"
        prompt_rat="The highlighted country is within the Pacific Islands region. Based on its position relative to neighboring larger landmasses like Australia and nearby countries such as Papua New Guinea and New Zealand, the highlighted country aligns with the location of Vanuatu. According to the context, Vanuatu has a territorial dispute over Matthew and Hunter Islands, claimed by both Vanuatu and France. Therefore, the presence of a dashed box labeled \"Disputed island\" suggests the inclusion of this dispute in the overview of the country's territories."
        propmt_answer="The answer is C."
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        self.tokenizer.chat_template = chat_template
        input_id=[]
        system_message = "You are a question analyzer, you will first provide detail thinking process and act as a helpful assistant by giving final solution."
        input_id += self.tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        roles=['user',"Analyzer","assistant"]
        content=[prompt_q,prompt_rat,propmt_answer]
        for rol,ct in zip(roles,content):
            conv = [{"role" : rol, "content" : ct}]
        # print(tokenizer.apply_chat_template(conv,tokenize=False))
            input_id += self.tokenizer.apply_chat_template(conv)
        return torch.tensor([input_id], dtype=torch.long)

    # def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
    #     # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    #     deepspeed_plugin = self.accelerator.state.deepspeed_plugin
    #     config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

    #     if model is not None:
    #         if hasattr(model, "config"):
    #             hidden_size = max(model.config.hidden_sizes) if getattr(model.config, "hidden_sizes", None) else getattr(model.config, "hidden_size", None)
    #             if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
    #                 # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
    #                 # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
    #                 config_kwargs.update(
    #                     {
    #                         "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
    #                         "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
    #                         "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
    #                     }
    #                 )

    #     # If ZeRO-3 is used, we shard both the active and reference model.
    #     # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
    #     if config_kwargs["zero_optimization"]["stage"] != 3:
    #         config_kwargs["zero_optimization"]["stage"] = 0
    #     model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    #     model.eval()
    #     return model

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle": False,
        }

        # prepare dataloader
        data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))
        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))
        return data_loader


    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(self.model).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")


    def GFlowNet_loss(
            self,
            logrewards,
            logPF,
            eos_logprob,
            generated_text,
            eos_token_id,
            pad_token_id,
            subtb_lambda,
            max_len,
            loss_type
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        if loss_type == 'modified_db':
            # modified db loss with logpb=0
            db_loss = (logrewards[:, :-1] + logPF[:, :-1] + eos_logprob[:, 1:] - logrewards[:, 1:] - eos_logprob[:,
                                                                                                     :-1]) ** 2
            # get a mask for newly generated tokens after the first eos in generated_text
            mask = (generated_text[:, encoded_input.size(-1):] == eos_token_id).cumsum(dim=-1) > 1
            # if mask is too short, pad it
            if mask.size(-1) < max_len:
                mask = torch.cat(
                    [mask, torch.ones(mask.size(0), max_len - 1 - mask.size(-1), dtype=torch.bool, device='cuda')],
                    dim=-1)
            mask = mask[:, :max_len]
            # get trajectory lengths by summing the mask
            traj_len = (~mask).sum(dim=-1)
            # get rid of the loss for the terminating step
            db_loss[mask] = 0
            batch_loss = db_loss.sum(-1) / traj_len
            # batch_loss = batch_loss.topk(bsz//2, largest=False, sorted=False).values.mean()
        elif loss_type == 'modified_subtb':
            # modified subTB loss with logpb=0

            delta = (logrewards[:, :-1] - eos_logprob[:, :-1] + logPF[:, :-1] - (
                        logrewards[:, 1:] - eos_logprob[:, 1:]))
            delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], 1).cumsum(1)

            # get a mask for tokens after the first eos in generated_text
            mask = (generated_text == eos_token_id).cumsum(dim=-1) > 1
            mask=mask[:,:max_len]
            if mask.size(-1) < max_len:
                mask = torch.cat([mask, torch.ones(mask.size(0), max_len-mask.size(-1), dtype=torch.bool, device='cuda')], dim=-1)
            #

            # get trajectory lengths by summing the mask

            batch_loss = 0.
            total_lambda = 0.
            for subtraj_len in range(1, max_len + 1):
                subtb_term = (delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]) ** 2
                #print("subtb_term: ",subtb_term)
                subtb_term[mask[:, subtraj_len - 1:]] = 0
                #print("subtb_term after: ", subtb_term)
                batch_loss += subtb_lambda ** (subtraj_len - 1) * subtb_term.sum()
                #print('batch loss:',batch_loss)
                total_lambda += subtb_lambda ** (subtraj_len - 1) * (~mask[:, subtraj_len - 1:]).sum()

            batch_loss /= total_lambda
        return batch_loss


    def get_sft_loss(self, logits, labels):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        # Enable model/pipeline parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return loss


    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
        loss_type:str="modified_subtb",
        subtb_lambda=1.
    ):
        """Compute the sub trajectory balance loss
        """
        metrics = {}

        b_policy_choice = random.randint(0, 3)
        temperature=1

        if b_policy_choice == 1:
            temperature = random.random()*(self.args.pf_temp_high-self.args.pf_temp_low)+self.args.pf_temp_low
        sampled_rational, logPF, eos_logprob, logrewards,reward_mask = \
            generate_and_return_eos_logprob(model,
                                            batch,
                                            processor=self.processor,
                                            min_step_len=self.args.skip_reward_step,
                                            temperature=temperature,
                                            eos_token_id=self.tokenizer.eos_token_id,
                                            label_pad_token_id=self.tokenizer.pad_token_id,
                                            reward_temperature=self.get_reward_temp(self.state.global_step),
                                            reward_tolerance=self.get_reward_tolerate(self.state.global_step),
                                            tokenizer=self.tokenizer,
                                            prompt_tokens=None,
                                            explore_nums=self.args.explore_nums,
                                            min_bs=self.args.explore_min_bs,
                                            max_len=self.args.rat_max_len,
                                            min_len=self.args.rat_min_len,
                                            )
        #dist.barrier()
        valid_mask = reward_mask.nonzero().squeeze(dim=-1)
        if self.state.global_step>self.args.warm_up_step and self.state.global_step % 25 ==0:
            print("generated rational: ",self.tokenizer.batch_decode(sampled_rational, skip_special_tokens=True))
        #print("valid mask: ", valid_mask, "reward before: ",logrewards,"rewawrd now: ",  logrewards[valid_mask])
        gfn_loss = self.GFlowNet_loss(logrewards[valid_mask], logPF[valid_mask], eos_logprob[valid_mask], sampled_rational[valid_mask],
                                      self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, subtb_lambda,
                                      self.args.rat_max_len, loss_type)

        gfn_loss = gfn_loss.mean()
        #print("Before: .....",torch.distributed.get_rank())
        def all_gather_tensor(tensor):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                tensor = tensor.detach()
                gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(gathered_tensor, tensor)
                tensor = torch.cat(gathered_tensor, dim=0)
            # else:
            #     print('not distributed')
            return tensor

        # gather chosen_rewards across devices
        #print(logrewards.sha
        # pe,logPF.shape,rat_reward.shape)
        #print("after masking:",logrewards.shape,reward_mask.shape)
        extended_mask = reward_mask[:, None].expand(-1, logrewards.size(1))
        #print("after masking:",logrewards*extended_mask)
        logrewards = all_gather_tensor(logrewards*extended_mask)

        logPF = all_gather_tensor(logPF*extended_mask)
        #rat_reward = all_gather_tensor(rat_reward.mean())
        #print("after: ",torch.distributed.get_rank())
        #print("Done calculating loss")


        #print("2 Done geting metrics")
        #print("rational reward : ",rat_reward)
        prefix = "eval_" if train_eval == "eval" else ""
        #metrics[f"{prefix}losses/gfn"] = gfn_loss.cpu()
       # print("3 Done geting metrics")
        metrics[f"{prefix}logrewards"] = logrewards.mean().cpu()
       #metrics[f"{prefix}rat_reward"] = rat_reward.mean().cpu()
        #print("4 Done geting metrics")
        # policy logps
        metrics[f"{prefix}logps"] = logPF.mean().cpu()
        #print("5 Done geting metrics")
        return gfn_loss, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:


        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext
        #print("in me")
        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
        #print("out me")
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")


        return loss




    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):

        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, random_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[[prompt, pol[len(prompt) :], ref[len(prompt) :]] for prompt, pol, ref in zip(random_batch["prompt"], policy_output_decoded, ref_output_decoded)],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

        return initial_output

    def log(self, logs: Dict[str, float],start_time) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs,start_time)
    def save_my_lora_ckpt(self, output_dir, training_args, model):
        self.state.save_to_json(os.path.join(output_dir, 'trainer_state.json'))
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(output_dir)
            model.save_pretrained(output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(output_dir, "non_lora_trainables.bin"))
            # if training_args.vit_lora_enable:
