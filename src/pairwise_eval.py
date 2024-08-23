import pandas as pd
import os
import numpy as np
import argparse

def process_pairwise_data(pairwise):
    # Count the number of NaN decisions
    failure_count = pairwise['decision'].isna().sum()
    total_entries = len(pairwise)
    
    # Drop rows where decisions are NaN before calculating accuracy
    pairwise = pairwise.dropna(subset=['decision'])
    accuracy = np.mean(pairwise['winner'] == pairwise['decision'])
    
    # Return accuracy and failure metrics
    return {'accuracy': accuracy, 'failure_count': failure_count, 'total_entries': total_entries}

def save_results(results, output_file):
    """ Save DataFrame results to a CSV file. """
    results.to_csv(output_file, index=False)

def main(input_path, output_path):
    """ Main execution function to process pairwise data and save results. """
    pairwise = pd.read_csv('data/k2-eval-pairwise.csv')
    results = pd.DataFrame()

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    for i, file in enumerate(sorted(os.listdir(input_path))):
        if 'pairwise' in file:
            v = pd.read_csv(f"{input_path}/{file}")
            batch = pd.concat([pairwise, v], axis=1)
            batch_results = process_pairwise_data(batch)
            # Add results from the current batch to the results DataFrame
            results = pd.concat([results, pd.DataFrame([batch_results])], ignore_index=True)
            # Save results for this iteration separately
    results.to_csv(f"{output_path}/pairwise_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process pairwise data and generate results.")
    parser.add_argument("input_path", type=str, help="Path to the input data directory.")
    parser.add_argument("output_path", type=str, help="Path to the output results directory.")
    args = parser.parse_args()
    main(args.input_path, args.output_path)


