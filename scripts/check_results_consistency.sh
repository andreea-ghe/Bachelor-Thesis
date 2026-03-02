#!/bin/bash

cd /home/ndreeaheorghe/bachelor-thesis

echo "=== Baseline Everyday (10 runs) ==="
for i in {1..10}; do
  echo "--- Run $i/10 ---"
  python -m experiments.eval_model --cfg experiments/two_piece_finetuned_scripts/everyday_eval.yaml
done

echo "=== Baseline Artifact (10 runs) ==="
for i in {1..10}; do
  echo "--- Run $i/10 ---"
  python -m experiments.eval_model --cfg experiments/two_piece_finetuned_scripts/artifact_eval.yaml
done

echo "=== Pair Attn Everyday (10 runs) ==="
for i in {1..10}; do
  echo "--- Run $i/10 ---"
  python -m experiments.eval_model --cfg experiments/pair_attn_finetuned_scripts/everyday_eval.yaml
done

echo "=== Pair Attn Artifact (10 runs) ==="
for i in {1..10}; do
  echo "--- Run $i/10 ---"
  python -m experiments.eval_model --cfg experiments/pair_attn_finetuned_scripts/artifact_eval.yaml
done

echo "=== Dist Bias Everyday (10 runs) ==="
for i in {1..10}; do
  echo "--- Run $i/10 ---"
  python -m experiments.eval_model --cfg experiments/dist_bias_finetuned_scripts/everyday_eval.yaml
done

echo "=== Dist Bias Artifact (10 runs) ==="
for i in {1..10}; do
  echo "--- Run $i/10 ---"
  python -m experiments.eval_model --cfg experiments/dist_bias_finetuned_scripts/artifact_eval.yaml
done

echo "=== All done! ==="
