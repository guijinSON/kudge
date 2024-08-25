# KUDGE: A Benchmark for Korean LLM-as-a-Judge

This is the official repository for KUDGE, a dataset aimed to evaluate LLMs in judging Korean responses.

# Quick Usage 

To run your model on our dataset run the following: 

```python
python src/main.py pairwise "$model" "$output_dir"
python src/main.py pointwise "$model" "$output_dir"
```

To receive evaluation results:

```python
python src/pointwise_eval.py results/$model report/$model
python src/pairwise_eval.py results/$model report/$model
```
