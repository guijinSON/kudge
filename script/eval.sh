#!/bin/bash

# Array of model names
models=(
   # "CohereForAI--c4ai-command-r-v01"
   #  "CohereForAI--c4ai-command-r-plus"
   #  "NousResearch--Meta-Llama-3-8B-Instruct"
   #  "NousResearch--Meta-Llama-3-70B-Instruct"
   #  "NousResearch--Meta-Llama-3.1-8B-Instruct"
   #  "NousResearch--Meta-Llama-3.1-70B-Instruct"
    "meta-llama--Meta-Llama-3.1-405B-Instruct-FP8"
    # "Qwen--Qwen1.5-4B-Chat"
    # "Qwen--Qwen1.5-7B-Chat"
    # "Qwen--Qwen1.5-14B-Chat"
    # "Qwen--Qwen1.5-MoE-A2.7B-Chat"
    # "Qwen--Qwen1.5-32B-Chat"
    # "Qwen--Qwen1.5-72B-Chat"
    # "01-ai--Yi-6B-Chat"
    # "01-ai--Yi-34B-Chat"
    # "mistralai--Mixtral-8x7B-Instruct-v0.1"
    # "mistralai--Mixtral-8x22B-Instruct-v0.1"
    # "mistralai--Mistral-Large-Instruct-2407"
    # "databricks--dbrx-instruct"
    # "mistralai--Mistral-7B-Instruct-v0.2"
    # "mistralai--Mistral-Nemo-Instruct-2407"
    # "yanolja--EEVE-Korean-Instruct-2.8B-v1.0"
    "yanolja--EEVE-Korean-Instruct-10.8B-v1.0"
    # "Qwen--Qwen2-0.5B-Instruct"
    # "Qwen--Qwen2-1.5B-Instruct"
    # "Qwen--Qwen2-7B-Instruct"
    # "Qwen--Qwen2-72B-Instruct"
    # "LGAI-EXAONE:EXAONE-3.0-7.8B-Instruct"
    # "gpt-4o"
    # "claude-sonnet"
    # "NCSOFT--Llama-3-OffsetBias-8B"
    # "prometheus-eval--prometheus-7b-v2.0"
    # "prometheus-eval--prometheus-8x7b-v2.0"
)

# Output directory
output_dir="results"

# Function to run a model for both pairwise and pointwise evaluations
run_model() {
    local model=$1
    echo "Running evaluations for model: $model"
    
    echo "Running pairwise evaluation..."
    python src/pointwise_eval.py results/$model report/$model

    
    echo "Running pointwise evaluation..."
    python src/pairwise_eval.py results/$model report/$model
    
    echo "Completed evaluations for model: $model"
    echo "----------------------------------------"
}

# Main execution
echo "Starting model evaluations..."


# Loop through each model and run evaluations
for model in "${models[@]}"; do
    run_model "$model"
    rm -rf ~/.cache/huggingface/hub
done

echo "All model evaluations completed."