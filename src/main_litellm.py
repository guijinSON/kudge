import argparse
import pandas as pd
import os
from model_utils import init_model_litellm, get_model_outputs_litellm
from data_processing import process_queries, extract_pairwise, extract_pointwise, create_dataframe, retry_none_decisions_litellm

def main(mode, subset, model_name, api_key, base_url, output_path):
    client = init_model_litellm(api_key,base_url)
    if subset=='full':
        data = pd.read_csv(f'data/k2-eval-{mode}.csv')
    elif subset == 'false_info':
        data = pd.read_csv(f'data/k2-eval-{mode}-falseinfo.csv')
        
    queries = data['judge_query'].values
    outputs = get_model_outputs_litellm(client, model_name, queries)

    if mode == 'pairwise':
        decisions = [extract_pairwise(o) for o in outputs]
    else:
        decisions = [extract_pointwise(o) for o in outputs]

    dfs = [create_dataframe(queries, outputs, decisions)]
    dfs.extend(retry_none_decisions_litellm(queries, outputs, decisions, client, model_name, extract_pairwise if mode == 'pairwise' else extract_pointwise))

    # Ensure output directory exists
    os.makedirs(f'{output_path}/{model_name.replace("/", "--")}', exist_ok=True)
    for i, df in enumerate(dfs):
        df.to_csv(f'{output_path}/{model_name.replace("/", "--")}/_{mode}_iteration_{i+1}-{subset}.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model inference on datasets.')
    parser.add_argument('mode', choices=['pairwise', 'pointwise'], help='Mode to process data: pairwise or pointwise')
    parser.add_argument("subset", choices=['full', 'false_info'], help="Which subset to use either full or false_info")
    parser.add_argument('model_name', type=str, help='Model name to use for loading and inference')
    parser.add_argument('api_key', type=str, help='API Key for LiteLLM.')
    parser.add_argument('base_url', type=str, help='Base URL for LiteLLM.')
    parser.add_argument('output_path', type=str, help='Path to save output results')
    args = parser.parse_args()

    main(args.mode, args.subset, args.model_name, args.api_key, args.base_url, args.output_path)

