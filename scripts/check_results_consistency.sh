#!/bin/bash

cd /home/ndreeaheorghe/bachelor-thesis

# echo "=== Baseline Everyday (10 runs) ==="
# for i in {1..10}; do
#   echo "--- Run $i/10 ---"
#   python -m experiments.eval_model --cfg experiments/two_piece_finetuned_scripts/everyday_eval.yaml
# done

# echo "=== Baseline Artifact (10 runs) ==="
# for i in {1..10}; do
#   echo "--- Run $i/10 ---"
#   python -m experiments.eval_model --cfg experiments/two_piece_finetuned_scripts/artifact_eval.yaml
# done

# echo "=== Pair Attn Everyday (10 runs) ==="
# for i in {1..10}; do
#   echo "--- Run $i/10 ---"
#   python -m experiments.eval_model --cfg experiments/pair_attn_finetuned_scripts/everyday_eval.yaml
# done

# echo "=== Pair Attn Artifact (10 runs) ==="
# for i in {1..10}; do
#   echo "--- Run $i/10 ---"
#   python -m experiments.eval_model --cfg experiments/pair_attn_finetuned_scripts/artifact_eval.yaml
# done

echo "=== Double Attn Layers Everyday (10 runs) ==="
for i in {1..10}; do
  echo "--- Run $i/10 ---"
  python -m experiments.eval_model --cfg experiments/new_arch_scripts/everyday_eval.yaml
done

echo "=== Double Attn Layers Artifact (10 runs) ==="
for i in {1..10}; do
  echo "--- Run $i/10 ---"
  python -m experiments.eval_model --cfg experiments/new_arch_scripts/artifact_eval.yaml
done

echo "=== Double Layers and Pair Attn Everyday (10 runs) ==="
for i in {1..10}; do
  echo "--- Run $i/10 ---"
  python -m experiments.eval_model --cfg experiments/pair_double_attn_new_arch_scripts/everyday_eval.yaml
done

echo "=== Double Layers and Pair Attn Artifact (10 runs) ==="
for i in {1..10}; do
  echo "--- Run $i/10 ---"
  python -m experiments.eval_model --cfg experiments/pair_double_attn_new_arch_scripts/artifact_eval.yaml
done
