#!/bin/bash

cd /home/ndreeaheorghe/bachelor-thesis

echo "=== Baseline Everyday (10 runs) ==="
for i in {1..10}; do
  echo "--- Run $i/10 ---"
  python -m experiments.eval_model --cfg experiments/finetune_everyday_eval.yaml
done

echo "=== Baseline Artifact (10 runs) ==="
for i in {1..10}; do
  echo "--- Run $i/10 ---"
  python -m experiments.eval_model --cfg experiments/finetune_artifact_eval.yaml
done

echo "=== Pair Attn Everyday (10 runs) ==="
for i in {1..10}; do
  echo "--- Run $i/10 ---"
  python -m experiments.eval_model --cfg experiments/finetune_pair_attn_everyday_eval.yaml
done

echo "=== Pair Attn Artifact (10 runs) ==="
for i in {1..10}; do
  echo "--- Run $i/10 ---"
  python -m experiments.eval_model --cfg experiments/finetune_pair_attn_artifact_eval.yaml
done

echo "=== All done! ==="
