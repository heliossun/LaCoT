

> **`W1-1`**: The main concern lies in the experimental section which I find is a bit thin.

We conduct additional experiments on widely used benchmarks including **MMMU<sup>pro</sup>**, **MMVet**, and **MME**, where MMMU-Pro is a more robust version of MMMU, designed to more rigorously assess LVLMs' understanding and reasoning capabilities. These benchmarks cover a broad range of tasks, including visual commonsense, fine-grained recognition, and multi-choice QA. 



| Method        	| MMMU<sup>pro</sup> 	| MMVet 	| MME  	| 
|---------------	|-----------	|-------	|------	|
| InternVL2-4B  	| -         	| 55.7  	| 2046 	|		
| Qwen2.5-VL-3B 	|      22.4     	|    61.4   	|    2134  	|		
| LaCoT-Qwen-3B (ours)	|   **28.9**       	| **69.6**  	|  **2208**    	|		
| InternVL2-8B  	| 25.4      	| 60.0  	| 2210 	|		
| Qwen2.5-VL-7B 	| 34.6      	| 70.5  	| 2333 	|		
| R1-Onevision  	| 28.2      	| 71.1  	|  1111    	|		
| LaCoT-Qwen-7B (ours) 	|       **35.3**    	| **74.2**  	| **2372** 	|		

Table T1. Test accuracy on various benchmarks. 

These results demonstrate that LaCoT consistently improves performance across different domains and model scales, suggesting strong generalization beyond mathematical reasoning tasks.


> **`W1-2`**:   I wonder if the comparison with GRPO in Table 1 is fair, as LaCoT uses a different training set as R1-OneVision.


Our SFT data is comparable with R1-OneVision. However, in the RL stage—which we consider the most critical for performance—R1-OneVision (GRPO) utilizes 10k newly collected samples. In contrast, LaCoT reuses only 3k samples from the SFT stage without additional data collection, which highlights its efficiency and generalization.
    
> **`W1-3`**: The performance of GRPO in Table 2 is a bit strange, as is contradictory with most recent papers observation. My empirical experience also indicates that GRPO works better than SFT especially in terms of generalization. The details of this ablation is not provided and no analysis on why this abnormal behavior happens.

In Table 2,  we study the effectiveness of different training algorithms for reasoning on 3k data samples. We maintain a consistent hyperparameter setting for all methods, including batch size, base model, and learning rate. For GRPO and RGFN, we set exploration number to 6 and sequence length to 700. The empirical results show that GRPO has slight worse performance than SFT, we conjecture that GRPO is more sensitive to hyperparameters and requires more training data, which aligns with our observation in Table T1.

> **`Q1-1`**: Since LaCoT introduces an additional neural-based reward model, what would be the computational overhead compared to, e.g., GRPO?

During training, LaCoT introduces no additional GPU memory or data preprocessing overhead compared to GRPO. However, due to the token-level reward approximation, LaCoT incurs higher computational cost in terms of runtime. Under the same experimental settings, LaCoT requires approximately 90 hours of total training time, while GRPO takes around 64 hours.

> **`W2-1`**: The proposed method is evaluated only on mathematical reasoning tasks. The generalizability to other domains remains unclear.

We followed the baseline (R1-OneVision) to evaluate LVLM reasoning on mathematic benchmarks. To further assess the generalizability of **LaCoT** beyond mathematical reasoning, we conducted additional experiments on three diverse visual understanding benchmarks: **MMMU<sup>pro</sup>**, **MMVet**, and **MME**. These benchmarks cover a broad range of tasks, including visual commonsense, fine-grained recognition, and multi-choice QA. 

| Method        	| MMMU<sup>pro</sup> 	| MMVet 	| MME  	| 
|---------------	|-----------	|-------	|------	|
| InternVL2-4B  	| -         	| 55.7  	| 2046 	|		
| Qwen2.5-VL-3B 	|      22.4     	|    61.4   	|    2134  	|		
| LaCoT-Qwen-3B (ours)	|   **28.9**       	| **69.6**  	|  **2208**    	|		
| LLaVA-CoT-11B | - | 60.3| - |
| InternVL2-8B  	| 25.4      	| 60.0  	| 2210 	|		
| Qwen2.5-VL-7B 	| 34.6      	| 70.5  	| 2333 	|		
| R1-Onevision  	| 28.2      	| 71.1  	|  1111    	|		
| LaCoT-Qwen-7B (ours) 	|       **35.3**    	| **74.2**  	| **2372** 	|		

These results demonstrate that LaCoT consistently improves performance across different domains and model scales, suggesting strong generalization beyond mathematical reasoning tasks.


> **`W2-2`**: The proposed method requires sampling multiple rationales at inference time, which is computationally expensive. The paper can benefit from a detailed analysis of computational costs compared to baselines.

We agree that rationale sampling introduces additional inference cost, and we address this by using mini-batching (with batch size k=5) to generate N rationales in N/k forward passes. Below, we report the average per-sample inference time (reasoning + answering) and corresponding performance of different reasoning-LVLM on MathVista and MathVerse:

| #Rationals (N)           	| 1   	| 5    	| 10   	| MathVista 	| MathVerse |
|------------------------	|-----	|------	|------	|-------------	|-------------	|
| LLaVA-CoT-11B	| -   	| 340s 	| 830s 	|       52.5  	| 22.6           |
| R1-OneVision-7B                	| 32s 	| -    	| -    	|     64.1    	|    37.8            |
| LaCoT-7B (ours)                   	| -   	| 30s  	| 65s  	|       **68.4**  	|      **39.7**      |

Where LLaVA-CoT-11B utilizes stage-wise beam search [1] at inference-time.

As shown in the table, LaCoT-7B achieves consistently stronger performance even with modest increases in inference time. Compared to other multi-rationale baselines, LaCoT strikes a favorable balance between computational cost and reasoning reliability, thereby improving both the trustworthiness of rationales and the accuracy of final answers.

> **`Q2-1`**:  How sensitive is the performance to the choice of lambda for approximating reward?

We examined the sensitivity of performance to the choice of $\lambda$ in reward approximation by training with values in the range [8,32]. While the approximated rewards showed slight differences in early stages, they rapidly converged and became nearly identical after around 250 training steps, suggesting that LaCoT is relatively robust to the choice of $\lambda$ within a reasonable range.
 
> **`Q2-2`**: What are training and inference times compared to baselines (SFT, GRPO)?

We report the total training time of fine-tuning Qwen2.5-VL using different objective functions on 3k samples, all conducted on 8×A100 (80G) GPUs: 
| Method | Training Time | Inference Time/sample | MathVista  |
|--------|---------------|-----------------------|------------|
| SFT    | 1 hour        | 34s                   | 62.7       |
| GRPO   | 64 hours      | 32s                   | 62.6       |
| RGFN   | 90 hours      | 30s                   | 66.8 (N=5) |


Where N indicates the number of explored rationals of BiN.

To avoid out-of-memory (OOM) issues during GRPO and LaCoT fine-tuning, we employed DeepSpeed Stage 3 with gradient checkpointing enabled. While this setting substantially increased training time, it enabled on-policy exploration.

These results demonstrate that while LaCoT incurs additional computational cost due to exploration and token-level reasoning supervision, it remains feasible for practical training pipelines and yields notable performance gains.



> **`Q2-3`**: How does the proposed reference-guided filtering compare to other exploration strategies like epsilon-greedy or entropy-based exploration?


We studied Epsilon-greedy in our early study, but we found that it doesn't scale well in training a reasoning LVLM. Specifically, the action space comprises the entire vocabulary (~50,000 tokens). In such a large space, most random actions are **semantically nonsensical** and low-reward, making exploration **inefficient** and often harmful. This results in **high-variance gradients** and **slow convergence**, leading to a catastrophic forgetting issue in our experiment.

Entropy-based exploration methods such as PPO rely on optimizing policy gradients to maximize expected rewards. However, these methods often struggle to align the learned policy with a target distribution, especially in complex reasoning tasks. In contrast, our approach builds on the GFlowNet framework, which is designed to sample from a target distribution via trajectory-level credit assignment, making it better suited for tasks requiring diverse and calibrated generation [2].

Our reference-guided filtering further enhances the stability of on-policy exploration by selectively retaining high-quality trajectories, ensuring both efficiency and robustness during training. This strategy offers a more principled and controllable alternative to stochastic exploration heuristics.

[1] Xu, G., Jin, P., Li, H., Song, Y., Sun, L., & Yuan, L. LLaVA-CoT: Let Vision Language Models Reason Step-by-Step. ICCV 2025.
[2] Hu, E.J., Jain, M., Elmoznino, E., Kaddar, Y., Lajoie, G., Bengio, Y., & Malkin, N. Amortizing intractable inference in large language models. ICLR 2024

> **`Q3-1`**:  Although the model is compared with SFT and GRPO results. Results on popular inference scaling approaches such as BofN etc. should also be compared with?

We provide comparison results of Best-of-N (BoN) approach below. Specifically, during inference, we sample N rational-answer pairs using LaCoT. For each pair, we compute the length-normalized log-likelihood of the generated answer as reward, and then select the final answer corresponding to the highest reward. To ensure a fair comparison, we do not use any external reward model.

| Method        	| MathVerse 	| MathVista 	| MMMU 	| MMVet 	|
|---------------	|:---------:	|:---------:	|:----:	|:-----:	|
| 3B w/ BofN         	|   21.2    	|    57.1   	| 44.7 	|  67.1 	|
| 3B w/ BiN (ours)       	|    **40.0**   	|    **63.2**   	| **48.8** 	|  **69.6** 	|
| 7B w/ BofN         	|   26.5   	|    62.2   	| 47.3 	|  71.2 	|
| 7B w/ BiN (ours)        	|    **39.7**   	|    **68.4**   	| **54.9** 	|  **74.2** 	|

We set number of candidates N={5, 10} for BoN and BiN, and we report the highest score of each method.

> **`Q3-2`**: Is this approach the first to adopt Amortizing Variational Inference method, more comparison with related work is needed. Also comparison with existing latent space reasoning approaches?


To the best of our knowledge, our work is the first to apply amortized variational inference to latent visual reasoning with a long CoT chain, where reasoning steps are learned and inferred in a latent space conditioned on both visual and textual input.

Previous works ([1] and [2]) focus solely on the textual domain, extending these techniques to the multimodal setting—particularly for long visual CoT—is nontrivial due to the need to model visual grounding and latent step-wise reasoning jointly.

[1] Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J.E., & Tian, Y. (2024). Training Large Language Models to Reason in a Continuous Latent Space. _ArXiv, abs/2412.06769_.

[2] Geiping, J., McLeish, S., Jain, N., Kirchenbauer, J., Singh, S., Bartoldson, B.R., Kailkhura, B., Bhatele, A., & Goldstein, T. (2025). Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach. _ArXiv, abs/2502.05171_.




> **`W4-1`**: Perhaps detailed at the beginning of section 3 with the potential inclusion of a new figure and that underlines the key fundamental strengths of the approach that you propose...

We will add a figure at the beginning of section 3 to clearly show our motivation.

>**`W4-2`**: Then in Section 3.3 I see the approach in general however I think it could benefit from a bit more detail in the figures and main text. ...

Figure 4 visualizes Equation 8, illustrating how BiN performs Bayesian marginalization over latent chain-of-thought trajectories Z to approximate the likelihood of the answer P(Y|X). We agree that additional detail would improve clarity, and we will revise both the figure and the accompanying explanation in the final version to better highlight the role of latent reasoning and its connection to our overall inference pipeline.

> **`W4-3`**:One nitpick, In the intro on the second to last paragraph you mention three points that overlap with your main contributions. Could you simply include this all under one list? It may make things a little clearer.

We will revise the introduction and consolidate the overlapping points into a unified contribution list to improve clarity and readability in the final version.

> **`W4-4`**: Consider LaCoT on benchmark domains targeting decision making and acting responses (e.g. E3VQA[1], MMIU[2]) that rely on some scene understanding and reasoning from world priors.

We attempt to evaluate on **MMIU**, but find that LaCoT and our baseline models (e.g., Qwen2.5-VL) struggle to generalize due to their limited training on single-image reasoning tasks. Regarding **E3VQA**, the dataset was not publicly available at the time of writing, which prevented us from conducting a fair comparison.

> **`Q4-1`**: I couldn't see where the reference rationales come from? Could you explain that? If it hasn't been included would it be worth briefly touching on this?

Each training sample in our dataset consists of a tuple {image, query, CoT, answer}, where the CoT (Chain-of-Thought) serves as the reference rationale for the answer. These rationales are generated by teacher models such as **GPT-4o** or **Deepseek-R1**, depending on the data source. We will clarify this detail in the revised version of the paper for better transparency.

> **`Q4-2`**: Did you have any issues with hallucinations in your rationales during inference? Presumably these would be more problematic as the sample size is lowered in BiN?

We observed some hallucination issues in generated rationales when the sample size N=1, where BiN can occasionally produce incorrect or misleading reasoning steps on the **MMMU** dataset.

To assess this, we used **one-shot reasoning** as a baseline and varied the number of sampled rationales N during inference. As shown below, increasing N from 1 to 5 significantly mitigates hallucination and improves answer accuracy:

| MathVerse     | 1    | 5    | 10   |
|---------------|------|------|------|
| one-shot reasoning     | 31.5 | -    | -    |
| LaCoT-Qwen-3B | 33.7 | 37.2 | 40.0 |

| MMMU          | 1    | 5    | 10   |
|---------------|------|------|------|
| one-shot reasoning     | 47.1 | -    | -    |
| LaCoT-Qwen-3B | 44.7 | 48.7 | 48.8 |

These results suggest that **hallucination can indeed be more pronounced at smaller N**, but BiN’s sampling strategy is effective in addressing it as N increases. We will include additional qualitative examples in the final version to illustrate this effect.

> **`Q4-3`**: When doing token level reward approximation I was a bit uncertain where rewards we actually be sampled vs. interpolated from Eq. 4, can you expand on this a bit?

The trade-off between accuracy and efficiency motivated our use of approximated rewards based on interpolated steps, which still retain strong performance while being computationally viable.
While $\lambda$=1 (i.e., no interpolation) would give slightly better performance, it is computationally infeasible. Following our current training setting, computing token-wise rewards with $\lambda$=1 takes **~50 minutes per sample (long visual CoT sequences)**. This requires **~400 hours of training for 1 epoch** on a 3k-sample dataset on 8*80G GPUs.




[1] Fanqing Meng, Jin Wang, Chuanhao Li, Quanfeng Lu, Hao Tian, Jiaqi Liao, Xizhou Zhu, Jifeng Dai, Yu Qiao, Ping Luo, et al. Mmiu: Multimodal multi-image understanding for evaluating large vision-language models. arXiv preprint arXiv:2408.02718, 2024

[2] I Lee, W Park, J Jang, M Noh, K Shim, B Shim. Towards Comprehensive Scene Understanding: Integrating First and Third-Person Views for LVLMs. arXiv preprint arXiv:2505.21955, 2025
