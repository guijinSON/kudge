from transformers import AutoTokenizer
from vllm import LLM
from config import sampling_params, tensor_parallel_size

def init_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)
    return tokenizer, llm

def get_model_outputs(llm, queries):
    outputs = llm.generate(queries, sampling_params)
    return [o.outputs[0].text for o in outputs]
