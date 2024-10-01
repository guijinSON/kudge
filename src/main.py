import argparse
import pandas as pd
import os
from model_utils import init_model, get_model_outputs
from data_processing import process_queries, extract_pairwise, extract_pointwise, create_dataframe, retry_none_decisions

def main(mode, subset, model_name, output_path):
    tokenizer, llm = init_model(model_name)
    
    if subset=='full':
        data = pd.read_csv(f'data/k2-eval-{mode}.csv')
    elif subset == 'false_info':
        data = pd.read_csv(f'data/k2-eval-{mode}-falseinfo.csv')
    elif subset == 'challenge':
        data = pd.read_csv(f'data/kudge-challenge.csv')
        
    queries = process_queries(tokenizer, data['judge_query'].values)
    outputs = get_model_outputs(llm, queries)

    if mode == 'pairwise':
        decisions = [extract_pairwise(o) for o in outputs]
    else:
        decisions = [extract_pointwise(o) for o in outputs]

    dfs = [create_dataframe(queries, outputs, decisions)]
    dfs.extend(retry_none_decisions(queries, outputs, decisions, llm, tokenizer, extract_pairwise if mode == 'pairwise' else extract_pointwise))

    # Ensure output directory exists
    os.makedirs(f'{output_path}/{model_name.replace("/", "--")}', exist_ok=True)
    for i, df in enumerate(dfs):
        df.to_csv(f'{output_path}/{model_name.replace("/", "--")}/_{mode}_iteration_{i+1}-{subset}.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model inference on datasets.')
    parser.add_argument('mode', choices=['pairwise', 'pointwise'], help='Mode to process data: pairwise or pointwise')
    parser.add_argument("subset", choices=['full', 'false_info','challenge'], help="Which subset to use either full, false_info, or challenge.")
    parser.add_argument('model_name', type=str, help='Model name to use for loading and inference')
    parser.add_argument('output_path', type=str, help='Path to save output results')
    args = parser.parse_args()

    main(args.mode, args.subset, args.model_name, args.output_path)

