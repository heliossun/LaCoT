from collections import Counter
import torch
from torch.distributions import Categorical
import math
import re
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"

def lora_to_base(model):
    try:
        model.disable_adapter_layers()
    except:
        print("+++++++++++No adapter layers to disable")
    model.eval()


def base_to_lora(model):
    try:
        model.enable_adapter_layers()
    except:
        print("----------No adapter layers to enable")
    model.train()


def get_limited_expr(expr, k):
    tokens = re.split('(-?\d+)', expr)

    if k > len(tokens) / 2:
        return expr

    operands = tokens[1::2]
    if len(operands) <= k:
        return expr

    operators = tokens[::2]
    if k > 1:
        operators = operators[:-1]

    end_operands = operands[-k:]
    if k > 1:
        end_operators = operators[-(k - 1):]
    else:
        end_operators = []

    end_expr = ''
    for i in range(len(end_operands)):
        if i > 0:
            end_expr += end_operators[i - 1]
            end_expr += end_operands[i]

    return end_expr


def generate(model, encoded_prompt, eos_token_id, max_len=10, temperature=1., vocab_nice_mask=None,
             vocab_naughty_mask=None, vocab_alpha=-99, top_k=999999, top_p=1., tokenizer=None, use_tools=False,
             limit_capability=0, operators=None, use_cache=True):
    active_seqs = torch.ones(encoded_prompt.size(0)).bool().to(encoded_prompt.device)
    logPF = encoded_prompt.new_zeros(encoded_prompt.size(0)).float()
    actions = encoded_prompt.clone()
    state = encoded_prompt.clone()
    if use_cache:
        past_key_values = model(state[:, :-1])['past_key_values']
    if use_tools:
        EQUAL_TOKS = [tokenizer.encode(" =")[-1], tokenizer.encode("=")[-1]]
    for i in range(max_len):
        if use_tools and i > 0:
            if use_cache and past_key_values is not None:
                output = model(state[:, -1:], attention_mask=state != eos_token_id, past_key_values=past_key_values)
            else:
                output = model(state, attention_mask=state != eos_token_id, position_ids=pos)
        else:
            if use_cache and not use_tools:
                output = model(state[:, -1:], past_key_values=past_key_values)
            else:
                output = model(state)
        if use_cache:
            past_key_values = output['past_key_values']
        with torch.no_grad():
            prob = (output['logits'][:, -1, :]).softmax(dim=-1)
            modified_logits = output['logits'][:, -1, :].clone()
            # implement top-k by getting the top-k largest values and setting the rest to 0
            if top_k < 999999:
                modified_logits[prob >= prob.topk(top_k)] = -float('inf')
            # implement top-p by getting indices in the top-p prob mass and setting the rest to 0
            if top_p < 1.:
                sorted_probs, indices = torch.sort(prob, dim=-1, descending=True)
                cumsum_prob = torch.cumsum(sorted_probs, dim=-1)
                nucleus = cumsum_prob < top_p
                nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
                modified_logits[~nucleus] = -float('inf')
            if vocab_nice_mask is not None:
                # add vocab_alpha to the logits of the unmasked vocab items
                modified_logits[:, ~vocab_nice_mask] += vocab_alpha
            prob = (modified_logits / temperature).softmax(dim=-1)
            token_ids = torch.multinomial(prob, num_samples=1)
        logprob = output['logits'][:, -1, :].log_softmax(dim=-1)
        logPF += logprob.gather(-1, token_ids).squeeze(-1)
        actions = torch.cat([actions, token_ids], dim=-1)
        state = torch.cat([state, token_ids], dim=-1)
        if use_tools:
            tool_used = False
            new_str = []
            for j, tok in enumerate(token_ids):
                if tok[0].item() in EQUAL_TOKS:
                    expr = tokenizer.batch_decode(remove_left_pad([state[j]], eos_token_id, eos_token_id))[0]
                    to_eval = expr.split("Answer:")[-1].split(',')[-1].split(".")[-1].split('=')[0]
                    if limit_capability > 1 and to_eval != "":
                        to_eval = get_limited_expr(to_eval, limit_capability)
                    try:
                        val = eval(to_eval)
                        tool_used = True
                    except:
                        new_str.append(state[j])
                        continue
                    new_str.append(torch.cat((state[j],
                                              tokenizer.batch_encode_plus([" " + str(val)], return_tensors='pt')[
                                                  'input_ids'].cuda()[0])))
                else:
                    new_str.append(state[j])
            state, pos, _ = left_pad(remove_left_pad(new_str, eos_token_id, eos_token_id, False), eos_token_id,
                                     eos_token_id)
            state = state.cuda()
            pos = pos.cuda()
            if tool_used:
                past_key_values = None
        # check if all sequences have generated eos
        active_seqs = active_seqs * (token_ids != eos_token_id).squeeze(-1)
        if torch.all(~active_seqs):
            break
    return actions, logPF, remove_left_pad(state, eos_token_id, eos_token_id)


def get_logits(model, rat_id, rat_lbl, batch, use_cache, dpo_forward, max_len, label_pad_token_id, pad_token_id,
               skip_first=None):
    query_id = batch["query_input_ids"].repeat(rat_id.shape[0], 1)
    query_label = batch["query_labels"].repeat(rat_id.shape[0], 1)
    input_ids = torch.cat((query_id, rat_id), dim=1)
    label = torch.cat((query_label, rat_lbl), dim=1)
    all_logits = model(
        input_ids,
        attention_mask=label != label_pad_token_id,
        pixel_values=batch["pixel_values"] * rat_id.shape[0],
        image_grid_thw=batch["image_grid_thw"] * rat_id.shape[0],
        use_cache=use_cache,
    ).logits
    all_logits = all_logits.to(torch.float32)
    all_logits = all_logits[:, :-1, :]
    new_labels = input_ids[:, 1:]
    if skip_first is None:
        skip_first = new_labels[:, :-rat_lbl.shape[1]].shape[1]
    rational_logits = all_logits[:, skip_first:, :].clone()
    
    all_logits = all_logits.detach()
    return rational_logits, skip_first


def all_gather_tensor(tensor):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor = tensor.detach()
        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_tensor, tensor)
        tensor = torch.cat(gathered_tensor, dim=0)
    # else:
    #     print('not distributed')
    return tensor


def get_generate_logits(model, batch, do_sample, temperature, max_len, min_len, cur_batch, min_step, reward_temperature,
                        top_k, top_p, label_pad_token_id, pad_token_id, skip_first, eos_token_id, min_bs=2,
                        use_cache=False,analyzer_token=None):
    query_id = batch["query_input_ids"]
    analyzer_tokens = analyzer_token.repeat(query_id.shape[0], 1).to(query_id.device)
    # rational_logits = []
    rational = []
    generated = torch.cat((query_id, analyzer_tokens), dim=-1).repeat(min_bs, 1)
   # generated = query_id
    past_key_values = None
    #print(generated)
    #print(s)
    is_finished = torch.zeros(query_id.shape[0], dtype=torch.bool, device=query_id.device)
    with torch.no_grad():
        if use_cache:
            output = model(
                input_ids=generated[:, :-1],
                attention_mask=generated[:, :-1] != label_pad_token_id,
                pixel_values=batch["pixel_values"] * min_bs,
                image_grid_thw=batch["image_grid_thw"] * min_bs,
                use_cache=use_cache,
            )
            past_key_values = output.past_key_values
        # max_len - 2 since we have to prepend analyzer token
        for step in range(max_len - 2):
            if past_key_values is not None:
                output = model(
                    generated[:, -1:],
                    attention_mask=generated[:, -1:] != label_pad_token_id,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                all_logits = output.logits
                past_key_values = output.past_key_values
                # print("logits shape:",all_logits.shape)
                modified_logits = all_logits[:, -1, :].to(torch.float32)

                #logits_all = all_logits.detach()
            else:
                all_logits = model(
                    generated,
                    attention_mask=generated != label_pad_token_id,
                    pixel_values=batch["pixel_values"] * min_bs,
                    image_grid_thw=batch["image_grid_thw"] * min_bs,
                    use_cache=False,
                ).logits
                # print("logits shape:",all_logits.shape)
                modified_logits = all_logits[:, -1, :].to(torch.float32)
                #logits_all = all_logits.detach()
            # rational_logits.append(last_logits.unsqueeze(1))

            # modified_logits = last_logits.clone().detach()
            if step < min_len:
                modified_logits[:, eos_token_id] = -float('inf')
            if do_sample:
                probs = torch.nn.functional.softmax(modified_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(modified_logits, dim=-1, keepdim=True)

            generated = torch.cat((generated, next_token), dim=-1)
            # Mask finished samples: once EOS is generated, always emit pad_token
            next_token = torch.where(
                is_finished.unsqueeze(1),
                torch.full_like(next_token, pad_token_id),
                next_token
            )

            rational.append(next_token)
            is_finished = is_finished | (next_token.squeeze(1) == eos_token_id)
            # Optional: early exit if all samples finished
            # if is_finished.all():
            if all_gather_tensor(is_finished).all():
                break

        torch.cuda.empty_cache()
    # rational_logits = torch.cat(rational_logits, dim=1)  # [B, T, vocab]
    rational = torch.cat(rational, dim=1)  # [B, T]
    rational = torch.cat((analyzer_tokens.repeat(min_bs, 1), rational), dim=-1)
    rational_logits, _ = get_logits(model, rational, rational, batch, False,
                                    True, max_len, label_pad_token_id, pad_token_id)
    return rational_logits, rational


def generate_and_return_eos_logprob(model, batch, processor, eos_token_id, label_pad_token_id, max_len=512, min_len=32,
                                    top_k=999999, top_p=1.0, min_step_len=16, min_bs=2,
                                    temperature=1, sep_token_id=0, reward_temperature=1.0,
                                    tokenizer=None, prompt_tokens=None, explore_nums=5, reward_tolerance=0.8):
    """
    explore_nums: number of rationals for exploration
    min_len: minimum rational length
    min_step_len: numer of reward been skipped
    """
    # action_seq: rational - input-ids [B,N]
    # generate and return the probability of generating eos at every step
    # output latent cot, logPF, eos_logprob, logrewards

    # reward_prompt=prompt_tokens.repeat(query_id.shape[0],1).to(query_id.device)
    # reward_prompt_label=torch.full(reward_prompt.size(),IGNORE_INDEX).to(query_id.device)

    # Exploit rational label and logits
    # print("in me +++++++")
    assistant_tok = f"{DEFAULT_IM_START_TOKEN}assistant\n"
    analyzer_tok = f"{DEFAULT_IM_START_TOKEN}Analyzer\n"
    assistant_token=processor.tokenizer(assistant_tok, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']
    analyzer_token=processor.tokenizer(analyzer_tok, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']
    
    rat_id = batch["rational_input_ids"]
    rat_lbl = batch["rational_labels"]
    # print("gt rational length: ",rat_lbl.shape)
    if rat_id.shape[1] > max_len:
        rat_id = rat_id[:, :max_len]
        rat_lbl = rat_lbl[:, :max_len]
    rational_logits_gt, skip_first = get_logits(model,
                                                rat_id, rat_lbl,
                                                batch=batch,
                                                use_cache=False,
                                                dpo_forward=True,
                                                max_len=max_len,
                                                label_pad_token_id=label_pad_token_id,
                                                pad_token_id=tokenizer.pad_token_id
                                                )

    # Create exploit-explore minibath
    # print(rational_logits_gt.shape)
    batch_rat_label = [rat_lbl.squeeze()]
    batch_rad_logits = [rational_logits_gt.squeeze()]
    # rat_len_mask = rational_logits_gt.new_ones(rational_logits_gt.size(0)*(mini_batch_size+1)).float()

    # Explore rational label and logits
    # start = timeit.default_timer()
    for i in range(explore_nums // min_bs):
        rational_logits, rational_labels = get_generate_logits(model, batch, True, cur_batch=i + 1,
                                                               min_step=min_step_len, min_bs=min_bs,
                                                               temperature=temperature, max_len=max_len,
                                                               reward_temperature=reward_temperature,
                                                               min_len=min_len, top_k=top_k, top_p=top_p,
                                                               label_pad_token_id=label_pad_token_id,
                                                               pad_token_id=tokenizer.pad_token_id,
                                                               skip_first=skip_first, eos_token_id=eos_token_id,
                                                               use_cache=False,analyzer_token=analyzer_token
                                                               )

        # print(torchaudio.functional.edit_distance(rational_labels_gt[0],rational_labels[0]),'      ',(rational_labels.shape[1]+rational_Len)*sim_tolerance)
        # if torchaudio.functional.edit_distance(rational_labels_gt[0],rational_labels[0]) > (rational_labels.shape[1]+rational_Len)*sim_tolerance:
        #     rat_len_mask[i+1]=0
        # zero_indices = ((rational_labels == label_pad_token_id).cumsum(dim=-1) == 1).nonzero()
        # print("rational: ",rational_labels.shape,rational_logits.shape)
        #print(tokenizer.batch_decode(rational_labels, skip_special_tokens=True))
        rational_logits = [rational_logits[i] for i in range(rational_logits.shape[0])]
        rational_labels = [rational_labels[i] for i in range(rational_labels.shape[0])]
        batch_rat_label.extend(rational_labels)
        batch_rad_logits.extend(rational_logits)
        torch.cuda.empty_cache()
    # stop = timeit.default_timer()
    # print('Explore rational Time: ', stop - start)
    batch_rat_label = torch.nn.utils.rnn.pad_sequence(batch_rat_label, batch_first=True,
                                                      padding_value=label_pad_token_id)
    batch_rad_logits = torch.nn.utils.rnn.pad_sequence(batch_rad_logits, batch_first=True, padding_value=0)
    # print("rational: ",batch_rat_label.shape,batch_rad_logits.shape)
    # print(tokenizer.batch_decode(batch_rat_label, skip_special_tokens=True))

    # initialize

    rational_Len = batch_rad_logits.shape[1]
    logeosprobs = batch_rad_logits.new_zeros(batch_rad_logits.size(0), max_len + 1).float()
    logpf = batch_rad_logits.new_zeros(batch_rad_logits.size(0), max_len + 1).float()
    logrewards = batch_rad_logits.new_zeros(batch_rad_logits.size(0), max_len + 1).float()

    loss_mask = batch_rat_label != label_pad_token_id
    logeosprobs[:, :rational_Len] = batch_rad_logits.log_softmax(dim=-1)[:, :, eos_token_id] * loss_mask
    # shift left by 1 to save log P of each token
    logpf[:, :rational_Len] = torch.gather(batch_rad_logits.log_softmax(-1), dim=2,
                                           index=batch_rat_label.unsqueeze(2)).squeeze(2) * loss_mask
    logrewards[:, 1:min_len + 1] = -99

    def rational_reward(logrewards):
        zero_indices = ((logrewards == 0).cumsum(dim=-1) == 1).nonzero()
        rows = zero_indices[:, 0]
        cols = zero_indices[:, 1] - 1
        return logrewards[rows, cols]

    lora_to_base(model.module)
    reward, _ = score_fast_new(model, batch, rational=None, skip_first=skip_first, reward_prompt=None,
                               reward_p_label=None, pad_token_id=label_pad_token_id,
                               eos_token_ids=eos_token_id, reduction="sum")
    logrewards[:, 0] = reward * (1. / reward_temperature)
    # print("first reward:",logrewards[:, 0])
    rat_rewards = []
    for j, rat in enumerate(batch_rat_label):
        rat = rat.unsqueeze(0)
        eos_idx = None
        is_finished = torch.zeros(rat.shape[0], dtype=torch.bool, device=rat.device)
        anchor_idx = min_len
        for i in range(min_len, max_len):
            if i < rat.shape[1]:
                is_finished = is_finished | (rat[0, i] == eos_token_id)
                if rat[0, i].item() == eos_token_id:
                    eos_idx = i

            if i % min_step_len == 0:
                if anchor_idx is not None:
                    # Now it's time to calculate follow-up rewards for the *previous* anchor
                    logrewards, _ = calculate_followup_rewards(model, batch, rat[:, :i + 1], label_pad_token_id,
                                                               eos_token_id, skip_first, 'sum',
                                                               anchor_idx=anchor_idx, cur_s=i, cur_bs=j,
                                                               reward_temp=reward_temperature, log_rewards=logrewards)
                # Update anchor to current step
                anchor_idx = i + 1
                # reward = score_fast_new(model, batch, rat[:, :i+1], label_pad_token_id, eos_token_id,skip_first, 'sum') * (
                #         1. / reward_temperature)

                # reward = logrewards[j, i]
            # print("reward: ",reward)
            # logrewards[j, i+1] = reward
            # Optional: early exit if all samples finished
            # if is_finished.all():
            if all_gather_tensor(is_finished).all():
                break
        # print("anchor index",anchor_idx)
        last_s_idx = eos_idx if eos_idx is not None else max_len - 1
        # print("eos token idx: ",last_s_idx)
        cur_rat = rat[:, :eos_idx + 1] if eos_idx is not None else rat
        # print("final rat",cur_rat)
        # if anchor_idx>last_s_idx:
        #     anchor_idx=last_s_idx-min_step_len
        # if eos_idx is not None:
        #     reward = score_fast_new(model, batch, rat[:, :eos_idx + 1], label_pad_token_id,eos_token_id, skip_first, 'sum') * (
        #             1. / reward_temperature)
        #     logrewards[j, eos_idx] = reward
        #     #print("in me 1:",logrewards[j, eos_idx])
        #     rat_rewards.append(reward/eos_idx)
        # else:
        #     reward = score_fast_new(model, batch, rat, label_pad_token_id,eos_token_id,
        #                             skip_first, 'sum') * (1. / reward_temperature)
        #     logrewards[j, -1] = reward
        #     #print("in me 2:", logrewards[j, -2])
        #     rat_rewards.append(reward/max_len)
        # print("final reward update?>>>>>")
        # if last_s_idx != max_len-1:
        #     print("before: ",logrewards)
        # logrewards, rat_reward = calculate_followup_rewards(model, batch, cur_rat, label_pad_token_id, eos_token_id,skip_first, 'sum',
        #                                        anchor_idx=anchor_idx, cur_s=last_s_idx, cur_bs=j, reward_temp=reward_temperature, log_rewards=logrewards)
        # if last_s_idx != max_len-1:
        #     print("after: ",logrewards)
        # print("before: ",logrewards[j][last_s_idx+1])
        reward, rat_reward = score_fast_new(model, batch, cur_rat, label_pad_token_id, eos_token_id, skip_first, 'sum')
        logrewards[j, last_s_idx + 1] = reward * (1. / reward_temperature)
        
        rat_rewards.append(rat_reward * (1. / reward_temperature))
    base_to_lora(model.module)

    rat_rewards = torch.cat(rat_rewards, dim=0)
    reward_mask = rat_rewards >= rat_rewards[0] * reward_tolerance
    # print(rat_rewards,reward_mask)
    #print("final reward",logrewards[1])

    # add eos token to the end of the sequence
    batch_rat_label = torch.cat([batch_rat_label, batch_rat_label.new_ones(batch_rat_label.size(0), 1) * eos_token_id],
                                dim=-1)
    # print("rational_labels++++++++++++++",rational_labels)
    logpf[:, -1] = 0
    logeosprobs[:, -1] = 0

    return batch_rat_label, logpf, logeosprobs, logrewards, reward_mask


@torch.no_grad()
def calculate_followup_rewards(model, batch, rational, pad_token_id, eos_token_ids,
                               skip_first=0, reduction='sum', reward_prompt=None,
                               reward_p_label=None, anchor_idx=None, cur_s=None, cur_bs=None,
                               reward_temp=1, log_rewards=None, delta_y=False, reward_est_beta=0.2
                               ):
    input_ids, labels = append_sol_and_remove_eos_simple(batch, rational, eos_token_ids, pad_token_id)
    logits = model(input_ids,
                   attention_mask=input_ids != pad_token_id,
                   pixel_values=batch["pixel_values"],
                   image_grid_thw=batch["image_grid_thw"],
                   use_cache=False).logits

    logits = logits.to(torch.float32)
    logits = logits[:, :-1, :]
    logits = logits[:, skip_first:, :]
    loss_mask = labels != pad_token_id
    # non_eos_mask = rat_sol_labels != eos_token_id
    logPF = torch.gather(logits.log_softmax(-1), dim=2,
                         index=labels.unsqueeze(2)).squeeze(2)
    logPF *= loss_mask
    if reduction == 'sum':
        res = logPF.sum(dim=-1)
        # res = torch.where((non_eos_mask.sum(dim=-1) - solution_len) < min_len, -99, res)
    elif reduction == "mean":
        res = logPF.sum(dim=-1) / loss_mask.sum(-1)
    else:
        res = logPF
    # print("current state: ",cur_s+1, "rational: ",labels.shape,labels)
    current_rewawrd = res * (1. / reward_temp)
    log_rewards[cur_bs, cur_s + 1] = current_rewawrd
    logPF_sums = torch.cumsum(logPF.squeeze(), dim=-1)

    if cur_s > anchor_idx:
        anchor_reward = log_rewards[cur_bs, anchor_idx]
        n = cur_s - anchor_idx + 2
        # print(current_rewawrd, anchor_reward)
        t = torch.arange(1, n + 1).to(anchor_reward.device)
        R_hat = anchor_reward + (t - 1) / (n - 1) * (current_rewawrd - anchor_reward)
        # print("r:",R_hat.shape,R_hat, )
        log_rewards[cur_bs, anchor_idx + 1:cur_s + 1] = R_hat[1:-1]

    return log_rewards, res * (1. / reward_temp) / cur_s


@torch.no_grad()
def score_fast_new(model, batch, rational, pad_token_id, eos_token_ids, skip_first=0, reduction='sum',
                   reward_prompt=None,
                   reward_p_label=None):
    # encoded_input: Q + Rational + Solution
    input_ids, labels = append_sol_and_remove_eos_simple(batch, rational, eos_token_ids, pad_token_id)
    # if prompt_reward:
    # input_ids=torch.cat([reward_prompt,input_ids],dim=-1)
    # labels=torch.cat([reward_p_label,labels],dim=-1)
    # skip_first+=reward_p_label.shape[1]
    
    logits = model(input_ids,
                   attention_mask=input_ids != pad_token_id,
                   pixel_values=batch["pixel_values"],
                   image_grid_thw=batch["image_grid_thw"],
                   use_cache=False).logits

    logits = logits.to(torch.float32)
    logits = logits[:, :-1, :]
    logits = logits[:, skip_first:, :]
    loss_mask = labels != pad_token_id
    # non_eos_mask = rat_sol_labels != eos_token_id
    # print(" in reward: ",logits.shape, labels.shape,labels)
    logPF = torch.gather(logits.log_softmax(-1), dim=2,
                         index=labels.unsqueeze(2)).squeeze(2)
    logPF *= loss_mask
    if reduction == 'sum':
        res = logPF.sum(dim=-1)
        # res = torch.where((non_eos_mask.sum(dim=-1) - solution_len) < min_len, -99, res)
    elif reduction == "mean":
        res = logPF.sum(dim=-1) / loss_mask.sum(-1)
    else:
        res = logPF
    return res, logPF.sum(dim=-1) / loss_mask.sum(-1)


@torch.no_grad()
def score_fast(model, batch_input, pad_token_id, skip_first=0, reduction='sum', reward_prompt=None,
               reward_p_label=None):
    input_ids, labels, images, image_sizes, modalities = batch_input
    # encoded_input: Q + Rational + Solution

    input_ids = torch.cat([reward_prompt, input_ids], dim=-1)
    labels = torch.cat([reward_p_label, labels], dim=-1)
    skip_first += reward_p_label.shape[1]
    logits, labels = model(input_ids,
                           attention_mask=input_ids != pad_token_id,
                           labels=labels,
                           images=images,
                           image_sizes=image_sizes,
                           modalities=modalities,
                           use_cache=False,
                           dpo_forward=True)
    # get the logits for rational+solutions only
    # print(solution_len,rational_len)
    logits = logits.to(torch.float32)
    logits = logits[:, :-1, :]
    labels = labels[:, 1:].clone()
    logits = logits[:, skip_first:, :]
    labels = labels[:, skip_first:]
    loss_mask = labels != pad_token_id
    # non_eos_mask = rat_sol_labels != eos_token_id
    logPF = torch.gather(logits.log_softmax(-1), dim=2,
                         index=labels.unsqueeze(2)).squeeze(2)
    logPF *= loss_mask
    if reduction == 'sum':
        res = logPF.sum(dim=-1)
        # res = torch.where((non_eos_mask.sum(dim=-1) - solution_len) < min_len, -99, res)
    elif reduction == "mean":
        res = logPF.sum(dim=-1) / loss_mask.sum(-1)
    else:
        res = logPF
    return res


@torch.no_grad()
def reward_keywords(model, encoded_input, keywords, reward, skip_first=1, gamma=1.):
    has_keywords = torch.isin(encoded_input[:, skip_first - 1:], keywords).sum(dim=-1).bool()
    # if torch.any(has_keywords):
    #    import pdb; pdb.set_trace()
    return torch.where(has_keywords, reward + math.log(gamma), reward)


def append_sol_and_remove_eos_simple(batch, rational=None, eos_token_id=None, pad_token_id=None):
    device = batch["query_input_ids"].device
    end_toks = torch.tensor([151645, 198]).to(device)
    new_text = []
    new_labels = []

    # if rational is not None:
    #     for rat in rational:
    #     # check if rational has eos token,
    #         eos_ind = ((rat == eos_token_id).cumsum(dim=-1) >=1).nonzero()
    #         if eos_ind>1:
    #             # if has eos, remove all padding tokens including and after eos token
    #             rat=rat[:eos_ind[0]]
    #         new_text.append(torch.cat([batch["prompt_input_ids"], rational, end_toks, batch["solution_input_ids"]], dim=-1))
    #         labels = torch.cat([rational, end_toks, batch["solution_labels"]], dim=-1)
    # else:
    #     input_ids = torch.cat([batch["prompt_input_ids"], batch["solution_input_ids"]], dim=-1)
    #     labels = batch["solution_labels"]
    if rational is not None:
        for query, rat, solution, solution_labl in zip(batch["query_input_ids"], rational, batch["solution_input_ids"],
                                                       batch["solution_labels"]):

            eos_ind = ((rat == eos_token_id).cumsum(dim=-1) >= 1).nonzero()
            if len(eos_ind) > 1:
                # if has eos, remove all padding tokens including and after eos token
                rat = rat[:eos_ind[0]]
            new_text.append(torch.cat([query, rat, end_toks, solution], dim=-1))
            new_labels.append(torch.cat([rat, end_toks, solution_labl], dim=-1))
        new_text = torch.nn.utils.rnn.pad_sequence(new_text, batch_first=True, padding_value=pad_token_id)
        new_labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=pad_token_id)
    else:
        new_text = torch.cat([batch["query_input_ids"], batch["solution_input_ids"]], dim=-1)
        new_labels = batch["solution_labels"]
    return new_text, new_labels


def append_sol_and_remove_eos(text: torch.Tensor, result: torch.Tensor, text_label: torch.Tensor,
                              result_label: torch.Tensor, eos_token_id: int, pad_token_id: int):
    # remove anything after the first eos token and append the result
    # if there is no eos token, append the result
    # text is a torch tensor with the first dimension being the batch
    # result is a torch tensor with the first dimension being the batch
    # this is a vectorized implementation
    # returns a torch tensor with the first dimension being the batch
    # and the second dimension being the length of the sequence
    # EOS token: "<|im_end|>" id=151645
    new_text = []
    new_labels = []
    end_toks = torch.tensor([151645, 198]).to(text.device)
    new_rat_label = None  # append eos tokens after rational if there is not any
    for t, r, t_l, r_l in zip(text, result, text_label, result_label):
        t[(t == eos_token_id).cumsum(dim=-1) >= 3] = eos_token_id
        eos_ind = ((t == eos_token_id).cumsum(dim=-1) == 3).nonzero()
        if len(eos_ind) < 1:
            new_text.append(t if r is None else torch.cat([t, end_toks, r]))
            new_labels.append(t_l if r_l is None else torch.cat([t_l, end_toks, r_l]))
            # append_rat=None
            continue
        # find the third eos token

        # remove the eos tokens from the result and shift the result to the left
        if r is not None:
            # print("in me <<<<<<<<<<<>>>>>>>>>>>>>")
            # print("find eos in input: ", eos_ind, t,t[:eos_ind[0]])
            new_text.append(torch.cat([t[:eos_ind[0]], end_toks, r]))
            new_labels.append(torch.cat([t_l[:eos_ind[0]], end_toks, r_l]))
            new_rat_label = torch.nn.utils.rnn.pad_sequence([t_l[:eos_ind[0]]], batch_first=True,
                                                            padding_value=pad_token_id)

            # print("<<<>>>New Text2: ", t[:eos_ind[0]])
        else:
            new_text.append(t[:eos_ind[0]])

    new_text = torch.nn.utils.rnn.pad_sequence(new_text, batch_first=True, padding_value=pad_token_id)
    new_labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=pad_token_id)
    # print(new_text)
    return new_text, new_labels, new_rat_label


def left_pad_generate(model, encoded_prompt, position_ids, eos_token_id, pad_token_id, max_len=10, temperature=1.,
                      top_k=999999, top_p=1.):
    active_seqs = torch.ones(encoded_prompt.size(0)).bool().to(encoded_prompt.device)
    logPF = encoded_prompt.new_zeros(encoded_prompt.size(0)).float()
    new_input = encoded_prompt.clone()
    for _ in range(max_len):
        output = model(new_input, attention_mask=(new_input != pad_token_id).long(), position_ids=position_ids)
        with torch.no_grad():
            prob = (output['logits'][:, -1, :]).softmax(dim=-1)
            modified_logits = output['logits'][:, -1, :].clone()
            # implement top-k by getting the top-k largest values and setting the rest to 0
            if top_k < 999999:
                modified_logits[prob >= prob.topk(top_k)] = -float('inf')
            # implement top-p by getting indices in the top-p prob mass and setting the rest to 0
            if top_p < 1.:
                sorted_probs, indices = torch.sort(prob, dim=-1, descending=True)
                cumsum_prob = torch.cumsum(sorted_probs, dim=-1)
                nucleus = cumsum_prob < top_p
                nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
                modified_logits[~nucleus] = -float('inf')
            prob = (modified_logits / temperature).softmax(dim=-1)
            token_ids = torch.multinomial(prob, num_samples=1)
        logprob = output['logits'][:, -1, :].log_softmax(dim=-1)
        logPF += logprob.gather(-1, token_ids).squeeze(-1)
        new_input = torch.cat([new_input, token_ids], dim=-1)
        position_ids = torch.cat([position_ids, position_ids[:, -1:] + 1], dim=-1)
        active_seqs = active_seqs * (token_ids != eos_token_id).squeeze(-1)
        if torch.all(~active_seqs):
            break
    return new_input, logPF


def remove_eos_and_pad_left(text: torch.Tensor, eos_token_id: int, pad_token_id: int):
    """
    remove anything after the first eos token, and left pad sequences"""
    stripped_text = []
    lens = []
    position_ids = []
    for t in text:
        if eos_token_id not in t:
            stripped_text.append(t)
            position_ids.append(torch.arange(t.size(-1)))
            lens.append(t.size(-1))
            continue
        # find the first eos token
        t[(t == eos_token_id).cumsum(dim=-1) >= 1] = eos_token_id
        eos_ind = ((t == eos_token_id).cumsum(dim=-1) == 1).nonzero()[0]
        stripped_text.append(t[:eos_ind])
        lens.append(eos_ind)
        position_ids.append(torch.arange(eos_ind.item()))
    left_pad_seqs = torch.nn.utils.rnn.pad_sequence([i.flip(0) for i in stripped_text], batch_first=True,
                                                    padding_value=pad_token_id).flip(dims=[1])
    left_pad_position_ids = torch.nn.utils.rnn.pad_sequence([i.flip(0) for i in position_ids], batch_first=True,
                                                            padding_value=0).flip(dims=[1])
    return left_pad_seqs, left_pad_position_ids, torch.tensor(lens)


def left_pad(text: torch.Tensor, eos_token_id: int, pad_token_id: int):
    stripped_text = []
    lens = []
    position_ids = []
    for t in text:
        # if eos_token_id not in t:
        stripped_text.append(t)

        position_ids.append(torch.arange(t[(t != eos_token_id).cumsum(dim=-1) >= 1].size(-1)))
        lens.append(t.size(-1))

    left_pad_seqs = torch.nn.utils.rnn.pad_sequence([i.flip(0) for i in stripped_text], batch_first=True,
                                                    padding_value=pad_token_id).flip(dims=[1])
    left_pad_position_ids = torch.nn.utils.rnn.pad_sequence([i.flip(0) for i in position_ids], batch_first=True,
                                                            padding_value=0).flip(dims=[1])
    return left_pad_seqs, left_pad_position_ids, torch.tensor(lens)


def remove_left_pad(text: torch.Tensor, eos_token_id: int, pad_token_id: int, return_padded: bool = True):
    """
    remove left padding and pad to the right"""
    stripped_text = []
    for t in text:
        # t[(t != eos_token_id).cumsum(dim=-1) >= 1] = eos_token_id
        # neos_ind = ((t != eos_token_id).cumsum(dim=-1) == 1).nonzero()[0]
        stripped_text.append(t[(t != eos_token_id).cumsum(dim=-1) >= 1])
        # stripped_text.append(torch.cat([t[-l:], t.new_ones(t.size(0), t.size(1)-l)*pad_token_id], dim=-1))
    # print(stripped_text)
    if return_padded:
        return torch.nn.utils.rnn.pad_sequence(stripped_text, batch_first=True, padding_value=pad_token_id)
    else:
        return stripped_text


def extract_answer(str_answer):
    try:
        try:
            extracted_answer = int(str_answer.split('=')[-1].strip().strip('.'))
        except:
            extracted_answer = int(str_answer.split()[-1].strip().strip('.'))
    except:
        extracted_answer = -1
    return extracted_answer


def check_answer(str_answer, num_sol):
    extracted_answer = extract_answer(str_answer)
    if num_sol == extracted_answer:
        return True
    return False


def run_evaluation(model, tokenizer, encoded_test_queries, test_num_sols, eos_token_id, pad_token_id, vocab_nice_list,
                   max_eval_len, num_samples, use_tools, limit_capability=0, operators=None, use_cache=True):
    base_to_lora(model)
    model.eval()
    test_correct = 0
    test_total = 0
    incorrect_answers = []
    for query_ind in tqdm(range(len(encoded_test_queries))):
        encoded_input = encoded_test_queries[query_ind]
        with torch.no_grad():
            generated_text, _, full_state = generate(model,
                                                     encoded_input.repeat(num_samples, 1),
                                                     eos_token_id=eos_token_id,
                                                     vocab_nice_mask=vocab_nice_list,
                                                     max_len=max_eval_len,
                                                     temperature=.1,
                                                     tokenizer=tokenizer,
                                                     use_tools=use_tools,
                                                     limit_capability=limit_capability,
                                                     operators=operators,
                                                     use_cache=use_cache)
        decoded_answers = tokenizer.batch_decode(
            append_sol_and_remove_eos(full_state if use_tools else generated_text, [None, ] * generated_text.size(0),
                                      eos_token_id, pad_token_id))
        extracted_answers = [extract_answer(i) for i in decoded_answers]
        answer = Counter(extracted_answers).most_common(1)[0][0]
        # if check_answer(decoded_answer, test_num_sols[query_ind]):
        if answer == test_num_sols[query_ind]:
            test_correct += 1
        else:
            incorrect_answers.append(decoded_answers)
        test_total += 1
        # print(decoded_answer)
    return test_correct / test_total, incorrect_answers