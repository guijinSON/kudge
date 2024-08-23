#!/bin/bash

# Array of model names
models=(
    "NousResearch--Meta-Llama-3-8B-Instruct"
    "amphora--eli"
    "Qwen--Qwen2-0.5B-Instruct"
    "Qwen--Qwen2-1.5B-Instruct"
    "Qwen--Qwen2-7B-Instruct"
    "prometheus-eval--prometheus-7b-v2.0"
    "NCSOFT--Llama-3-OffsetBias-8B"
    # Add more models here
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