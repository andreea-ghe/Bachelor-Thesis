#!/bin/bash

cd /home/ndreeaheorghe/bachelor-thesis

echo "=== Baseline Everyday (5 runs) ==="
for i in {1..5}; do
  echo "--- Run $i/5 ---"
  python -m experiments.eval_model --cfg experiments/two_piece_scripts/everyday_eval.yaml
done

echo "=== Baseline Artifact (5 runs) ==="
for i in {1..5}; do
  echo "--- Run $i/5 ---"
  python -m experiments.eval_model --cfg experiments/two_piece_scripts/artifact_eval.yaml
done

echo "=== Pair Attn Everyday (5 runs) ==="
for i in {1..5}; do
  echo "--- Run $i/5 ---"
  python -m experiments.eval_model --cfg experiments/pair_attn_scripts/everyday_eval.yaml
done

echo "=== Pair Attn Artifact (5 runs) ==="
for i in {1..5}; do
  echo "--- Run $i/5 ---"
  python -m experiments.eval_model --cfg experiments/pair_attn_scripts/artifact_eval.yaml
done

echo "=== Double Attn Layers Everyday (5 runs) ==="
for i in {1..5}; do
  echo "--- Run $i/5 ---"
  python -m experiments.eval_model --cfg experiments/double_attn_scripts/everyday_eval.yaml
done

echo "=== Double Attn Layers Artifact (5 runs) ==="
for i in {1..5}; do
  echo "--- Run $i/5 ---"
  python -m experiments.eval_model --cfg experiments/double_attn_scripts/artifact_eval.yaml
done

echo "=== Double Layers and Pair Attn Everyday (5 runs) ==="
for i in {1..5}; do
  echo "--- Run $i/5 ---"
  python -m experiments.eval_model --cfg experiments/pair_double_attn_scripts/everyday_eval.yaml
done

echo "=== Double Layers and Pair Attn Artifact (5 runs) ==="
for i in {1..5}; do
  echo "--- Run $i/5 ---"
  python -m experiments.eval_model --cfg experiments/pair_double_attn_scripts/artifact_eval.yaml
done
