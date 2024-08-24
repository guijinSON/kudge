from transformers import AutoTokenizer
from vllm import LLM
from config import sampling_params, tensor_parallel_size
import openai
from tqdm import tqdm

def init_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)
    return tokenizer, llm

def init_model_litellm(api_key,base_url):
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url
)
    return client


def get_model_outputs(llm, queries):
    outputs = llm.generate(queries, sampling_params)
    return [o.outputs[0].text for o in outputs]

def get_model_outputs(llm, queries):
    outputs = llm.generate(queries, sampling_params)
    return [o.outputs[0].text for o in outputs]

def get_model_outputs_litellm(client, model_name, queries):
    outputs = []
    for query in tqdm(queries):
        response = client.chat.completions.create(
        model=model_name,
        messages = [
            {
                "role": "user",
                "content": query
            }
        ],
    )
        outputs.append(response.choices[0].message.content)
    return outputs