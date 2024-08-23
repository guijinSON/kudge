from vllm import SamplingParams
import torch

# Configuration for sampling parameters and parallel size
sampling_params = SamplingParams(
    temperature=0.3, #top_p=0.95,
    min_tokens=20, max_tokens=512,
    include_stop_str_in_output=True,
    stop=['[[A]]','[[B]]','[RESULT]1','[RESULT] 1', '[RESULT]2','[RESULT] 2', '[RESULT]3','[RESULT] 3', '[RESULT]4','[RESULT] 4','[RESULT]5','[RESULT] 5','Score:\n1', 'Score:\n2', 'Score:\n3', 'Score:\n4', 'Score:\n5','Score: \n1', 'Score: \n2', 'Score: \n3', 'Score: \n4', 'Score: \n5']
)

tensor_parallel_size = torch.cuda.device_count()
