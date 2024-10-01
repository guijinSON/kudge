import pandas as pd
import re
from model_utils import get_model_outputs, get_model_outputs_litellm

def process_queries(tokenizer, queries):
    processed_queries = []
    for qry in queries:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": qry}
        ]
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processed_queries.append(qry)
    return processed_queries

def extract_pairwise(text):
    pattern = r'\[\[(.*?)\]\]'
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None

def extract_pointwise(text):
    pattern = r'\[RESULT\]\s*\d'
    matches = re.findall(pattern, text)
    return matches[-1][-1] if matches else None

def create_dataframe(queries, outputs, decisions):
    return pd.DataFrame({
        'query': queries,
        'output': outputs,
        'decision': decisions
    })


def retry_none_decisions(queries, outputs, decisions, llm, tokenizer, extraction_function, max_retries=2):
    dataframes = []
    for iteration in range(max_retries):
        retry_indices = [index for index, decision in enumerate(decisions) if decision is None]
        if not retry_indices:
            break
        retry_queries = [queries[i] for i in retry_indices]
        retry_outputs = get_model_outputs(llm, retry_queries)
        retry_decisions = [extraction_function(o) for o in retry_outputs]
        
        # Update the original lists with the new results
        for idx, new_idx in enumerate(retry_indices):
            outputs[new_idx] = retry_outputs[idx]
            decisions[new_idx] = retry_decisions[idx]
        
        # Store iteration results
        df = create_dataframe(queries, outputs, decisions)
        dataframes.append(df)
    return dataframes


def retry_none_decisions_litellm(queries, outputs, decisions, client, model_name, extraction_function, max_retries=2):
    dataframes = []
    for iteration in range(max_retries):
        retry_indices = [index for index, decision in enumerate(decisions) if decision is None]
        if not retry_indices:
            break
        retry_queries = [queries[i] for i in retry_indices]
        retry_outputs = get_model_outputs_litellm(client, model_name, retry_queries)
        retry_decisions = [extraction_function(o) for o in retry_outputs]
        
        # Update the original lists with the new results
        for idx, new_idx in enumerate(retry_indices):
            outputs[new_idx] = retry_outputs[idx]
            decisions[new_idx] = retry_decisions[idx]
        
        # Store iteration results
        df = create_dataframe(queries, outputs, decisions)
        dataframes.append(df)
    return dataframes