import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64
import json
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen2_5_vl_gfn")
class Qwen2_5_VL_GFN(lmms):
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
            self,
            pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
            device: Optional[str] = "cuda",
            device_map: Optional[str] = "auto",
            batch_size: Optional[Union[int, str]] = 1,
            use_cache=True,
            use_flash_attention_2: Optional[bool] = False,
            min_pixels: int = 256 * 28 * 28,
            max_pixels: int = 1605632,
            max_num_frames: int = 32,
            use_custom_video_loader: Optional[bool] = False,
            fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
            max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
            **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")
        if '3b' in pretrained:
            model_base = "ZachSun/Qwen2.5-gfn-sft-3b-250k"
        else:
            model_base = "ZachSun/Qwen2.5-gfn-sft-7b-250k"
        accelerator = Accelerator()
        print(model_base)
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        use_flash_attention_2=True
        if use_flash_attention_2:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
            self.answer_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_base,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype="auto",
                                                                             device_map=self.device_map).eval()
            self.answer_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_base, torch_dtype="auto",
                                                                             device_map=self.device_map).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.out_file=open("/mnt/disks/new-disk/lmms_eval/mathVista_7B_sft_t0.5.json",'a')
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
        # self.analyzer_ids = self.processor.tokenizer("<|im_start|>Analyzer\n<think>\n", return_tensors="pt").input_ids
        # self.assistant_ids = self.processor.tokenizer("<|im_start|>assistant\n", return_tensors="pt").input_ids
        self.end_think_tok = self.processor.tokenizer("\n</think>\n", return_tensors="pt")['input_ids'].to(self._device)
        self.eos_id = torch.tensor([151645, 198]).to(self._device)
    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def get_generate_logits(self, model, inputs, do_sample, temperature, max_len, min_len,
                            pad_token_id, eos_token_id, min_bs=2, use_cache=False):
        # query_id = batch["prompt_input_ids"]
        # analyzer_tokens = torch.tensor([151644, 54911, 198]).repeat(input_ids.shape[0], 1).to(input_ids.device)
        # rational_logits = []
        rational = []
        input_ids=inputs.input_ids
        generated = input_ids.repeat(min_bs, 1)
        # generated = input_ids.clone()
        past_key_values = None
        # print(input_ids)
        is_finished = torch.zeros(min_bs, dtype=torch.bool, device=input_ids.device)

        with torch.inference_mode():
            if use_cache:
                output = model(
                    input_ids=generated[:, :-1],
                    attention_mask=generated[:, :-1] != pad_token_id,
                    pixel_values=inputs["pixel_values"].repeat(min_bs, 1),
                    image_grid_thw=inputs["image_grid_thw"].repeat(min_bs, 1),
                    use_cache=use_cache,
                )
                #print(output.keys())
                past_key_values = output.past_key_values
            # max_len - 2 since we have to prepend analyzer token
            for step in range(max_len - 2):
                if past_key_values is not None:
                    output = model(
                        generated,
                        past_key_values=past_key_values,
                        pixel_values=inputs["pixel_values"].repeat(min_bs, 1),
                        image_grid_thw=inputs["image_grid_thw"].repeat(min_bs, 1),
                        use_cache=True,
                    )
                    all_logits = output.logits
                    past_key_values = output.past_key_values
                    
                    # print("logits shape:",all_logits.shape)
                    modified_logits = all_logits[:, -1, :].to(torch.float32)

                    # logits_all = all_logits.detach()
                else:
                    all_logits = model(
                        generated,
                        attention_mask=generated != pad_token_id,
                        pixel_values=inputs["pixel_values"].repeat(min_bs, 1),
                        image_grid_thw=inputs["image_grid_thw"].repeat(min_bs, 1),
                        use_cache=False,
                    ).logits
                    # print("logits shape:",all_logits.shape)
                    modified_logits = all_logits[:, -1, :].to(torch.float32)
                    # logits_all = all_logits.detach()
                # rational_logits.append(last_logits.unsqueeze(1))

                # modified_logits = last_logits.clone().detach()
                modified_logits[:, pad_token_id] = -float('inf')
                if step < min_len:
                    modified_logits[:, eos_token_id] = -float('inf')
                if do_sample:
                    probs = torch.nn.functional.softmax(modified_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(modified_logits, dim=-1, keepdim=True)
                next_token = torch.where(
                    is_finished.unsqueeze(1),
                    torch.full_like(next_token, pad_token_id),
                    next_token
                )
                generated = torch.cat((generated, next_token), dim=-1)
                # Mask finished samples: once EOS is generated, always emit pad_token
                

                rational.append(next_token)
                is_finished = is_finished | (next_token.squeeze(1) == eos_token_id)
                if is_finished.all():
                    break

            #torch.cuda.empty_cache()
            rational = torch.cat(rational, dim=1)
        return rational

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []


        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            cur_id=doc_id[0]
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")



            messages = []
            processed_visuals = []
            for i, context in enumerate(contexts):
                message = [{"role": "system", "content": "You are a helpful assistant."}]

                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        if self.use_custom_video_loader:
                            visual = read_video_pyav_base64(visual, num_frm=self.max_num_frames, fps=self.fps,
                                                            img_format="JPEG", max_image_size=self.max_image_size)
                            image_contents = list(map(lambda x: f"data:image/jpeg;base64,{x}", visual))
                            message.append({"role": "user", "content": [{"type": "video", "video": image_contents},
                                                                        {"type": "text", "text": context}]})
                        else:
                            vr = decord.VideoReader(visual)
                            first_frame = vr[0].asnumpy()
                            height, width = first_frame.shape[:2]
                            # max_pixels = height * width
                            message.append({"role": "user",
                                            "content": [{"type": "video", "video": visual, "max_pixels": 360 * 420},
                                                        {"type": "text", "text": context}]})
                    elif isinstance(visual, Image.Image):  # Single image
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        message.append({"role": "user", "content": [
                            {"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"},
                            {"type": "text", "text": context}]})
                    elif isinstance(visual, (list, tuple)) and all(
                            isinstance(v, Image.Image) for v in visual):  # Multiple images
                        image_content = []
                        for v in visual:
                            base64_image = v.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            image_content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
                        message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
                    else:
                        message.append({"role": "user", "content": [{"type": "text", "text": context}]})
                else:
                    message.append({"role": "user", "content": [{"type": "text", "text": context}]})

                messages.append(message)

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for i, t in enumerate(text):
                text[i] += "<|im_start|>Analyzer\n"
            image_inputs, video_inputs = process_vision_info(messages)
            # print(text)
            # print(self.assistant_ids)
            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                # fps=self.fps,
                padding=True,
                return_tensors="pt",
            )
            pad_token_id = self.processor.tokenizer.pad_token_id
            # self.analyzer_ids=self.analyzer_ids.repeat(inputs.input_ids.shape[0],1)
            # self.assistant_ids = self.assistant_ids.repeat(inputs.input_ids.shape[0], 1)
            # old_inputs=inputs.input_ids
            # inputs.input_ids = torch.cat((inputs.input_ids,self.analyzer_ids),dim=-1)
            # inputs.attention_mask = inputs.input_ids!=pad_token_id
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
                # self.assistant_ids=self.assistant_ids.to("cuda")
                # old_inputs=old_inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)
                # self.assistant_ids=self.assistant_ids.to(self.device)
                # old_inputs=old_inputs.to(self.device)
            #print("inputs: ",inputs.keys()) dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'])
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 2048
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            rationals = []
            solutions = []
            explore_nums = 5
            exp_temp = 0.7
            min_bs=1
            if "explore_nums" in gen_kwargs:
                explore_nums = gen_kwargs["explore_nums"]
            if "min_bs" in gen_kwargs:
                min_bs = gen_kwargs["min_bs"]
            if "exp_temp" in gen_kwargs:
                exp_temp = gen_kwargs["exp_temp"]
            keep_cot=False
            if "keep_cot" in gen_kwargs:
                keep_cot = gen_kwargs["keep_cot"]
                gen_kwargs.pop("keep_cot")
            #messages*=explore_nums
            messages=[messages[0].copy() for _ in range(explore_nums)]
            
            prefix_len=inputs.input_ids.shape[1]
            
            with (torch.inference_mode()):
                cumulate_lops = inputs.input_ids.new_zeros(explore_nums, explore_nums).float()
                # print(cumulate_lops.shape)
                for i in range(explore_nums // min_bs):
                    # rational = self.get_generate_logits(self.model, inputs, True,
                    #                                     temperature=exp_temp, max_len=700, min_len=64,
                    #                                     pad_token_id=pad_token_id,
                    #                                     eos_token_id=self.tokenizer.eos_token_id, min_bs=min_bs,
                    #                                     use_cache=True)
                    # text_outputs = self.tokenizer.batch_decode(rational, skip_special_tokens=True)
                    # rational = [response.strip() for response in text_outputs]
                    # print("rat: ",rational)

                    # generate rationals
                    cont = self.model.generate(
                        inputs.input_ids.repeat(min_bs, 1),
                        pixel_values=inputs.pixel_values.repeat(min_bs, 1),
                        image_grid_thw=inputs.image_grid_thw.repeat(min_bs, 1),
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        pad_token_id=pad_token_id,
                        do_sample=True if exp_temp > 0 else False,
                        temperature=exp_temp,
                        max_new_tokens=1024,
                        use_cache=self.use_cache,
                    )
                    #print(inputs.input_ids.shape,cont.shape)
                    generated_ids_trimmed=[]
                    for out_ids in cont:
                        modified_ids = out_ids[prefix_len:]
                        #print(modified_ids)
                        if modified_ids[-1]!=self.processor.tokenizer.eos_token_id:
                            modified_ids=torch.cat((modified_ids,self.end_think_tok.squeeze(),self.eos_id),dim=-1)
                            #print(modified_ids)
                        generated_ids_trimmed.append(modified_ids)
                    rational = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=False)
                    
                        
                    #print("rational: ", rational)
                    rationals.extend(rational)
                for i in range(explore_nums):
                    #print(i,"before: ", len(messages[i]))
                    messages[i].append({"role": "Analyzer", "content": [{"type": "text", "text": rationals[i]}]})
                    #print(i,"after: ",len(messages[i]))
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                #print("full test: ",text)
                # image_inputs, video_inputs = process_vision_info(messages)
                self.processor.tokenizer.padding_side = 'left'
                input_ids = self.processor(
                    text=text,
                    images=image_inputs*explore_nums,
                    # fps=self.fps,
                    padding=True,
                    return_tensors="pt",
                ).input_ids
                if self.device_map == "auto":
                    input_ids = input_ids.to("cuda")
                else:
                    input_ids = input_ids.to(self.device)
                new_input_len=input_ids.shape[1]

                #generate answers given each rational
                for i in range(explore_nums // min_bs):

                    cont = self.answer_model.generate(
                        input_ids[i*min_bs:i*min_bs+min_bs,:],
                        pixel_values=inputs.pixel_values.repeat(min_bs, 1),
                        image_grid_thw=inputs.image_grid_thw.repeat(min_bs, 1),
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=pad_token_id,
                        do_sample=True if gen_kwargs["temperature"] > 0 else False,
                        temperature=gen_kwargs["temperature"],
                        max_new_tokens=gen_kwargs["max_new_tokens"],
                        use_cache=self.use_cache,
                    )
                    generated_ids_trimmed = [out_ids[new_input_len:] for out_ids in cont]

                    # print(generated_ids_trimmed)
                    answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                                          clean_up_tokenization_spaces=False)
                    
                    solutions.extend(answers)
               
                # combine each answer with all rationals and get logps
                for j,solution in enumerate(solutions):
                    temp_messages=[msg.copy() for msg in messages]
                    for i in range(explore_nums):
                        temp_messages[i].append({"role": "assistant", "content": [{"type": "text", "text": solution}]})

                    text = self.processor.apply_chat_template(temp_messages, tokenize=False, add_generation_prompt=False)
                    self.processor.tokenizer.padding_side = 'right'
                    input_ids = self.processor(
                        text=text,
                        images=image_inputs*explore_nums,
                        # fps=self.fps,
                        padding=True,
                        return_tensors="pt",
                    ).input_ids
                    if self.device_map == "auto":
                        input_ids = input_ids.to("cuda")
                    else:
                        input_ids = input_ids.to(self.device)
                    
                    for i in range(explore_nums // min_bs):
                        cur_inp=input_ids[i*min_bs:i*min_bs+min_bs,:]
                        all_logits = self.model(
                            cur_inp,
                            pixel_values=inputs.pixel_values.repeat(min_bs, 1),
                            image_grid_thw=inputs.image_grid_thw.repeat(min_bs, 1),
                            use_cache=self.use_cache,
                        ).logits
                        all_logits = all_logits.to(torch.float32)
                        all_logits = all_logits[:, :-1, :]
                        new_labels = cur_inp[:, 1:]
                        comp_ids = new_labels[:,prefix_len:]
                        comp_logits = all_logits[:,prefix_len:,:]
                        loss_mask = comp_ids != pad_token_id
                        # non_eos_mask = rat_sol_labels != eos_token_id
                        logPF = torch.gather(comp_logits.log_softmax(-1), dim=2,
                                            index=comp_ids.unsqueeze(2)).squeeze(2)
                        logPF *= loss_mask
                        
                        cumulate_lops[j][i*min_bs:i*min_bs+min_bs]=logPF.sum(dim=-1)/ loss_mask.sum(-1)
                    del temp_messages
                   
                #print(cumulate_lops)
                marginal_log_y=torch.mean(cumulate_lops,dim=-1)
                #print("marginal log y: ",marginal_log_y)
                max_index = torch.argmax(marginal_log_y)
                
                text_outputs = [solutions[max_index]]
                rational_outputs=[rationals[max_index]]
                
                cotent={
                    "doc_id": cur_id,
                    "rationals":rationals,
                    "solutions": solutions,
                    "likelihoods":cumulate_lops.cpu().detach().numpy().tolist(),
                    "best_index":max_index.item()
                }
                self.out_file.write(json.dumps(cotent)+'\n')
                self.out_file.flush()
            outputs = []
            if keep_cot:
                outputs = [rat.strip()+ response.strip() for rat,response in zip(rational_outputs,text_outputs)]
            else:
                try:
                    # output=[rat.strip()+response.strip() for rat,response in zip(rational,answers)]
                    for response in text_outputs:
                        if "**Answer:**" in response:
                            outputs.append(response.split("**Answer:**", 1)[1].strip())
                        else:
                            outputs.append(response.split("Answer:", 1)[1].strip())
                except:
                    outputs = [response.strip() for response in text_outputs]
            #print("final output: ", outputs)
            for ans, context in zip(outputs, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
