
> **`Q1-1`**: The main concern lies in the experimental section which I find is a bit thin.

We provide new experimental results on three diverse domain visual tasks to asses LaCoT's general visual reasoning ability.




> **`Q1-2`**:   I wonder if the comparison with GRPO in Table 1 is fair, as LaCoT uses a different training set as R1-OneVision.

For the SFT stage, approximately 60% of our training data overlaps with that used by R1-OneVision, while the remaining 40% is drawn from LLaVA-CoT to mitigate data bias and enhance reasoning diversity. Importantly, in the RL stage—which we consider the most critical for performance—R1-OneVision (GRPO) utilizes 10k newly collected samples, whereas LaCoT reuses only 3k samples from the SFT stage without additional data collection. This indicates that LaCoT achieves strong performance with significantly less RL supervision, highlighting its efficiency and generalization.
    
> **`Q1-3`**: The performance of GRPO in Table 2 is a bit strange, as is contradictory with most recent papers observation. My empirical experience also indicates that GRPO works better than SFT especially in terms of generalization. The details of this ablation is not provided and no analysis on why this abnormal behavior happens.

We agree that GRPO typically demonstrates stronger generalization than SFT, as shown in prior work. However, GRPO is a reinforcement learning algorithm that is highly sensitive to hyperparameters such as the number of generations, sequence length, and batch size. In our GRPO fine-tuning setup, we used 4 generations per input, a max sequence length of 512, and a batch size of 8. The need to search for good parameters can be costly, and we have not explored the full range of possible settings and tricks.

> **`Q1-4`**: Since LaCoT introduces an additional neural-based reward model, what would be the computational overhead compared to, e.g., GRPO?

LaCoT introduces a reward model, but its overall computational overhead is lower than that of GRPO during training. GRPO requires loading both a policy model and a reference model of the same size, effectively doubling the memory footprint. In contrast, LaCoT only loads a single reward model and applies lightweight LoRA adapters to convert it into a trainable policy model. This design reduces memory consumption and simplifies the training pipeline, making LaCoT more efficient in practice.

> **`Q2-1`**: The proposed method is evaluated only on mathematical reasoning tasks. The generalizability to other domains remains unclear.

To assess the generalizability of **LaCoT** beyond mathematical reasoning, we conducted additional experiments on three diverse visual reasoning benchmarks: **MMMU<sup>pro</sup>**, **MMVet**, and **MME**. These benchmarks cover a broad range of tasks, including visual commonsense, fine-grained recognition, and multi-choice QA.

| Method        	| MMMU<sup>pro</sup> 	| MMVet 	| MME  	| 
|---------------	|-----------	|-------	|------	|
| InternVL2-4B  	| -         	| 55.7  	| 2046 	|		
| Qwen2.5-VL-3B 	|      22.4     	|    61.4   	|    2134  	|		
| LaCoT-Qwen-3B (ours)	|   **28.9**       	| **69.6**  	|  **2208**    	|		
| InternVL2-8B  	| 25.4      	| 60.0  	| 2210 	|		
| Qwen2.5-VL-7B 	| 34.6      	| 70.5  	| 2333 	|		
| R1-Onevision  	| 28.2      	| 71.1  	|  1111    	|		
| LaCoT-Qwen-7B (ours) 	|           	| **74.2**  	| **2372** 	|		

These results demonstrate that LaCoT consistently improves performance across different domains and model scales, suggesting strong generalization beyond mathematical reasoning tasks.


> **`Q2-2`**: The proposed method requires sampling multiple rationales at inference time, which is computationally expensive. The paper can benefit from a detailed analysis of computational costs compared to baselines.

We agree that rationale sampling introduces additional inference cost, and we address this by using mini-batching (with batch size k=5) to generate N rationales in N/k forward passes. Below, we report the per-sample inference time (reasoning + answering) and corresponding average performance of different inference-time scaling approach on MathVista and MathVerse:

| #Rationals (N)           	| 1   	| 5    	| 10   	| Performance 	|
|------------------------	|-----	|------	|------	|-------------	|
| Greedy                	| 32s 	| -    	| -    	| 50.7        	|
| BoN                    	| -   	| 25s  	| 60s  	| 51.2        	|
| Stage-wise beam search [1]	| -   	| 340s 	| 830s 	| 37.6        	|
| BiN (ours)                   	| -   	| 30s  	| 65s  	| 54.1        	|

For stage-wise beam search [1], we use the official implementation provided by LLaVA-CoT-11B [1].

As shown in the table, BiN achieves consistently stronger performance even with modest increases in inference time. Compared to other multi-rationale baselines, LaCoT strikes a favorable balance between computational cost and reasoning reliability, improving both the trustworthiness of rationales and final answer accuracy.

> **`Q2-3`**:  How sensitive is the performance to the choice of lambda for approximating reward?

We examined the sensitivity of performance to the choice of $\lambda$ in reward approximation by comparing settings with $\lambda$=8 and $\lambda$=32 during training. Our observations show that although the approximated rewards differ slightly at the beginning, they quickly converge and become nearly identical after around 250 training steps. This suggests that LaCoT is relatively robust to the choice of $\lambda$ within a reasonable range.
 
> **`Q2-4`**: What are training and inference times compared to baselines (SFT, GRPO)?

Below we report the total training time for Qwen2.5-VL using 3k samples under different objective functions, all conducted on 8×A100 (80G) GPUs: 
- SFT: 1 hour
- w/ GRPO: 30 hours
- LaCoT: 90 hours


To avoid out-of-memory (OOM) issues during GRPO and LaCoT fine-tuning, we employed DeepSpeed Stage 3 with gradient checkpointing enabled. While this setting substantially increased training time, it enabled efficient on-policy exploration.

We also analyzed the primary contributors to LaCoT’s training time. The most time-consuming components are:  
(1) **On-policy trajectory sampling**, and  
(2) **Token-wise reward approximation**, controlled by $\lambda$


| Running-Time         	| N=6, $\lambda$=8 	| N=4, $\lambda$=8 	| N=4, $\lambda$=32 	|
|----------------------	|------------------	|------------------	|-------------------	|
| Exploration          	| 960s             	| 880s             	| 880s              	|
| Reward approximation 	| 380s             	| 270s             	| 70s               	|

Where N is the number of explored trajectories.

These results show that while LaCoT introduces additional computational cost due to exploration and token-level reasoning supervision, it remains feasible for practical training pipelines and leads to notable performance gains.

> **`Q2-5`**: How does the proposed reference-guided filtering compare to other exploration strategies like epsilon-greedy or entropy-based exploration?

While we were unable to conduct a comprehensive comparison with other exploration strategies (e.g., epsilon-greedy or entropy-based methods) due to time constraints, we would like to highlight the conceptual advantages of our approach.

Entropy-based exploration methods such as PPO rely on optimizing policy gradients to maximize expected rewards. However, these methods often struggle to align the learned policy with a target distribution, especially in complex reasoning tasks. In contrast, our approach builds on the GFlowNet framework, which is designed to sample from a target distribution via trajectory-level credit assignment, making it better suited for tasks requiring diverse and calibrated generation.

Our reference-guided filtering further enhances this by stabilizing on-policy exploration through selective retention of high-quality trajectories, ensuring both efficiency and robustness during training. This strategy offers a more principled and controllable alternative to stochastic exploration heuristics.

[1] Xu, G., Jin, P., Li, H., Song, Y., Sun, L., & Yuan, L. LLaVA-CoT: Let Vision Language Models Reason Step-by-Step. ICCV 2025.

> **`Q3-1`**:  Although the model is compared with SFT and GRPO results. Results on popular inference scaling approaches such as BofN etc. should also be compared with?

We provide comparison results of Best-of-N (BoN) approach below. Specifically, during inference, we sample N rational-answer pairs using LaCoT. For each pair, we compute the length-normalized log-likelihood of the generated answer as reward, and then select the final answer corresponding to the highest reward. To ensure a fair comparison, we do not use any external reward model.

| Method        	| MathVerse 	| MathVista 	| MMMU 	| MMVet 	|
|---------------	|:---------:	|:---------:	|:----:	|:-----:	|
| 3B w/ BoN         	|       	|    57.1   	| 44.7 	|  67.1 	|
| 3B w/ BiN (ours)       	|    **40.0**   	|    **63.2**   	| **48.8** 	|  **69.6** 	|
| 7B w/ BoN         	|      	|    62.2   	| 47.3 	|  71.2 	|
| 7B w/ BiN (ours)        	|    **39.7**   	|    **68.4**   	| **54.9** 	|  **74.2** 	|

We set number of candidates N={5, 10} for BoN and BiN, and we report the highest score of each method.

> **`Q3-2`**: Is this approach the first to adopt Amortizing Variational Inference method, more comparison with related work is needed. Also comparison with existing latent space reasoning approaches?

To the best of our knowledge, our work is the first to apply amortized variational inference to latent visual reasoning, where reasoning steps are learned and inferred in a latent space conditioned on both visual and textual input.

While there have been recent efforts exploring latent-space reasoning in large language models (LLMs), such as [1] and [2], these works focus on purely textual tasks and do not incorporate multimodal inputs or vision-language grounding. We acknowledge the importance of positioning our work in the broader context of latent reasoning, and we will include a more detailed comparison with these methods in the final version of the paper due to rebuttal time constraints.

[1] Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J.E., & Tian, Y. (2024). Training Large Language Models to Reason in a Continuous Latent Space. _ArXiv, abs/2412.06769_.
[2] Geiping, J., McLeish, S., Jain, N., Kirchenbauer, J., Singh, S., Bartoldson, B.R., Kailkhura, B., Bhatele, A., & Goldstein, T. (2025). Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach. _ArXiv, abs/2502.05171_.




> **`Q4-1`**: Unclear writing in section 3. 

As a general comment, we thank the reviewer for their detailed comments. We will revise our writing in the final version.

> **`Q4-2`**: Consider LaCoT on benchmark domains targeting decision making and acting responses (e.g. E3VQA[1], MMIU[2]) that rely on some scene understanding and reasoning from world priors.

We have added additional experimental results on several single-image visual reasoning benchmarks to broaden the evaluation of LaCoT.

However, we were unable to include results on **MMIU**, as it involves multi-image inputs, which are currently beyond the scope of our model and training setup. Similarly, we could not evaluate on **E3VQA** because the dataset is not publicly available at the time of writing. We hope to include these in future work.

> **`Q4-3`**: I couldn't see where the reference rationales come from? Could you explain that? If it hasn't been included would it be worth briefly touching on this?

Each training sample in our dataset consists of a tuple {image, query, CoT, answer}, where the CoT (Chain-of-Thought) serves as the reference rationale. These rationales are generated by powerful teacher models such as **GPT-4o** or **Deepseek-R1**, depending on the data source. We will clarify this detail in the revised version of the paper for better transparency.

> **`Q4-4`**: Did you have any issues with hallucinations in your rationales during inference? Presumably these would be more problematic as the sample size is lowered in BiN?

We did observe some hallucination issues in generated rationales when sample size N=1,  where BiN can occasionally produce incorrect or misleading reasoning steps on datasets **MMMU**.

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

> **`Q4-5`**: When doing token level reward approximation I was a bit uncertain where rewards we actually be sampled vs. interpolated from Eq. 4, can you expand on this a bit?

Theoretically, the approximation error decays as $\mathcal{O}(\lambda^{2})$; choosing $\lambda$ sufficiently small keeps it arbitrarily close to~$0$. Our empirical results show that $\lambda$=8 has similar performance with $\lambda$=16. 

While $\lambda$=1 would give the most accurate, fully-sampled reward (i.e., no interpolation), it is computationally infeasible. As shown below, computing token-wise rewards with $\lambda$=1 takes **~50 minutes per sample** for long visual CoT sequences. This would require **~400 hours for  training 1 epoch** on a 3k-sample dataset with our current computational resources.

| Running-Time         	| E=6, $\lambda$=1 | E=6, $\lambda$=8 	| E=4, $\lambda$=8 	| E=4, $\lambda$=32 	|
|----------------------	|---	|------------------	|------------------	|-------------------	|
| Exploration          	| 960s | 960s             	| 880s             	| 880s              	|
| Reward approximation 	| 3040s  |380s             	| 270s             	| 70s               	|

This trade-off between accuracy and efficiency motivated our use of approximated rewards based on interpolated steps, which still retain strong performance while being computationally viable.

[1] Fanqing Meng, Jin Wang, Chuanhao Li, Quanfeng Lu, Hao Tian, Jiaqi Liao, Xizhou Zhu, Jifeng Dai, Yu Qiao, Ping Luo, et al. Mmiu: Multimodal multi-image understanding for evaluating large vision-language models. arXiv preprint arXiv:2408.02718, 2024
[2] I Lee, W Park, J Jang, M Noh, K Shim, B Shim. Towards Comprehensive Scene Understanding: Integrating First and Third-Person Views for LVLMs. arXiv preprint arXiv:2505.21955, 2025
