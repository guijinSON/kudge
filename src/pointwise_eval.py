import pandas as pd
import os
import numpy as np
from scipy.stats import pearsonr
import argparse

def process_pointwise_data(pointwise):
    failure_count = pointwise['decision'].isna().sum()
    total_entries = len(pointwise)

    pointwise = pointwise.dropna(subset=['decision'])
    average_error = (pointwise['decision'] - pointwise['final_score']).mean()
    off_by_05 = np.mean(np.abs(pointwise['decision'] - pointwise['final_score']) < 1)
    pearson_corr, _ = pearsonr(pointwise['final_score'].values, pointwise['decision'].values)

    return {
        'average_error': average_error,
        'off_by_05': off_by_05,
        'pearson_corr': pearson_corr,
        'failure_count': failure_count,
        'total_entries': total_entries
    }


def main(input_path, data, output_path):
    if data == 'full':
        data_path = f'data/k2-eval-pointwise.csv'
    elif data == 'false_info':
        data_path = f'data/k2-eval-pointwise-falseinfo.csv'
        
    pointwise = pd.read_csv(data_path)
    results = pd.DataFrame()

    os.makedirs(output_path, exist_ok=True)
    for i, file in enumerate(sorted(os.listdir(input_path))):
        if 'pointwise' in file:
            v = pd.read_csv(f"{input_path}/{file}")
            batch = pd.concat([pointwise, v], axis=1)
            batch_results = process_pointwise_data(batch)
            # Add results from the current batch to the results DataFrame
            results = pd.concat([results, pd.DataFrame([batch_results])], ignore_index=True)
            # Save results for this iteration separately
    results.to_csv(f"{output_path}/pointwise_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process pairwise data and generate results.")
    parser.add_argument("input_path", type=str, help="Path to the input data directory.")
    parser.add_argument("subset", type=str, help="Which subset to use either full or false_info")
    parser.add_argument("output_path", type=str, help="Path to the output results directory.")
    args = parser.parse_args()
    main(args.input_path, args.subset, args.output_path)