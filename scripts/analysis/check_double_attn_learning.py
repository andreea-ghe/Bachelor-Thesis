"""
Check if the double attention layers learned anything meaningful.
"""
import sys
import torch
import numpy as np


def analyze_checkpoint(ckp_path):
    print(f"Loading: {ckp_path}\n")
    ckp = torch.load(ckp_path, map_location='cpu', weights_only=False)

    if 'state_dict' in ckp:
        sd = ckp['state_dict']
    else:
        sd = ckp

    # gate values
    print("=" * 60)
    print("GATE VALUES (0 = layer bypassed, >0 = layer active)")
    print("=" * 60)
    for name in ['gate_self2', 'gate_cross2']:
        if name in sd:
            val = sd[name].item()
            print(f"  {name}: {val:.6f}")
        else:
            print(f"  {name}: NOT FOUND in checkpoint")

    # weight divergence: how much did tf_self2/tf_cross2 change from tf_self1/tf_cross1?
    print()
    print("=" * 60)
    print("WEIGHT DIVERGENCE (tf_self2 vs tf_self1, tf_cross2 vs tf_cross1)")
    print("=" * 60)

    layer_pairs = [
        ('tf_self1', 'tf_self2'),
        ('tf_cross1', 'tf_cross2'),
    ]

    for base_name, new_name in layer_pairs:
        base_keys = [k for k in sd if k.startswith(f'{base_name}.')]
        new_keys = [k for k in sd if k.startswith(f'{new_name}.')]

        if not new_keys:
            print(f"  {new_name}: NOT FOUND in checkpoint (no double attn)")
            continue

        total_diff = 0.0
        total_norm = 0.0
        param_count = 0

        for bk in base_keys:
            nk = bk.replace(base_name, new_name, 1)
            if nk in sd:
                diff = (sd[bk] - sd[nk]).float().norm().item()
                norm = sd[bk].float().norm().item()
                total_diff += diff
                total_norm += norm
                param_count += 1

        if param_count > 0:
            rel_divergence = total_diff / max(total_norm, 1e-8) * 100
            print(f"  {new_name} vs {base_name}:")
            print(f"    Absolute L2 diff (sum over {param_count} params): {total_diff:.4f}")
            print(f"    Relative divergence: {rel_divergence:.2f}%")
            if rel_divergence < 1:
                print(f"    --> Layers barely changed from initialization (< 1%)")
            elif rel_divergence < 10:
                print(f"    --> Layers moderately diverged")
            else:
                print(f"    --> Layers significantly diverged from copies")

    # pair geometric encoder
    print()
    print("=" * 60)
    print("PAIR GEOMETRIC ENCODER")
    print("=" * 60)
    pair_keys = [k for k in sd if 'pair_geometric_encoder' in k]
    if pair_keys:
        total_norm = sum(sd[k].float().norm().item() for k in pair_keys)
        print(f"  Found {len(pair_keys)} parameters, total norm: {total_norm:.4f}")
        for k in pair_keys:
            print(f"    {k}: shape={list(sd[k].shape)}, norm={sd[k].float().norm().item():.4f}")
    else:
        print("  NOT FOUND (no pair attention in this checkpoint)")

    # summary
    print()
    print("=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    gate_s = sd.get('gate_self2', None)
    gate_c = sd.get('gate_cross2', None)
    if gate_s is not None and gate_c is not None:
        gs = gate_s.item()
        gc = gate_c.item()
        if abs(gs) < 0.01 and abs(gc) < 0.01:
            print("  Gates are ~0: double attention layers are effectively BYPASSED.")
            print("  The model learned that the extra layers are not needed.")
        elif abs(gs) < 0.1 and abs(gc) < 0.1:
            print(f"  Gates are small ({gs:.4f}, {gc:.4f}): layers contribute weakly.")
        else:
            print(f"  Gates are open ({gs:.4f}, {gc:.4f}): layers are actively used.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)
    analyze_checkpoint(sys.argv[1])
