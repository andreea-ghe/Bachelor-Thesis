#!/bin/bash

cd /home/ndreea/bachelor-thesis

# echo "=== Baseline Everyday (3 runs) ==="
# for i in {1..1}; do
#   echo "--- Run $i/3 ---"
#   python -m experiments.eval_model --cfg experiments/multi_piece_scripts/everyday_eval.yaml
# done

# echo "=== Baseline Artifact (3 runs) ==="
# for i in {1..1}; do
#   echo "--- Run $i/3 ---"
#   python -m experiments.eval_model --cfg experiments/multi_piece_scripts/artifact_eval.yaml
# done

# echo "=== Pair Attn Everyday (3 runs) ==="
# for i in {1..1}; do
#   echo "--- Run $i/3 ---"
#   python -m experiments.eval_model --cfg experiments/multi_pair_attn_scripts/everyday_eval.yaml
# done

# echo "=== Pair Attn Artifact (3 runs) ==="
# for i in {1..1}; do
#   echo "--- Run $i/3 ---"
#   python -m experiments.eval_model --cfg experiments/multi_pair_attn_scripts/artifact_eval.yaml
# done

echo "=== Double Attn Layers Everyday (3 runs) ==="
for i in {1..3}; do
  echo "--- Run $i/3 ---"
  python -m experiments.eval_model --cfg experiments/multi_double_attn_scripts/everyday_eval.yaml
done

echo "=== Double Attn Layers Artifact (3 runs) ==="
for i in {1..3}; do
  echo "--- Run $i/3 ---"
  python -m experiments.eval_model --cfg experiments/multi_double_attn_scripts/artifact_eval.yaml
done

# echo "=== Double Layers and Pair Attn Everyday (3 runs) ==="
# for i in {1..1}; do
#   echo "--- Run $i/3 ---"
#   python -m experiments.eval_model --cfg experiments/pair_double_attn_scripts/everyday_eval.yaml
# done

#   echo "=== Double Layers and Pair Attn Artifact (3 runs) ==="
# for i in {1..1}; do
#   echo "--- Run $i/3 ---"
#   python -m experiments.eval_model --cfg experiments/pair_double_attn_scripts/artifact_eval.yaml
# done
