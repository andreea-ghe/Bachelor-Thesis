"""
Check if the pair geometric encoder is actually learning.
"""

import sys
import os
import glob
import re
import torch
import numpy as np

def extract_epoch(filename):
    """Extract epoch number from checkpoint filename."""
    basename = os.path.basename(filename)
    if 'last' in basename:
        return 99999  # sort last checkpoints at the end
    match = re.search(r'epoch=(\d+)', basename)
    if match:
        return int(match.group(1))
    return -1

def analyze_checkpoint(filepath, reference_sd=None):
    """Analyze pair_geometric_encoder weights in a checkpoint."""
    ckpt = torch.load(filepath, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)

    pair_keys = sorted([k for k in sd.keys() if 'pair_geometric' in k])
    if not pair_keys:
        return None

    results = {}
    for k in pair_keys:
        v = sd[k]
        short_name = k.replace('pair_geometric_encoder.', '')
        stats = {
            'mean': v.mean().item(),
            'std': v.std().item() if v.numel() > 1 else 0.0,
            'min': v.min().item(),
            'max': v.max().item(),
            'norm': v.norm().item(),
        }
        if reference_sd and k in reference_sd:
            ref = reference_sd[k]
            diff = (v - ref).abs()
            stats['diff_mean'] = diff.mean().item()
            stats['diff_max'] = diff.max().item()
        results[short_name] = stats

    # check LR if available
    lr_info = {}
    if 'lr_schedulers' in ckpt:
        for i, sched in enumerate(ckpt['lr_schedulers']):
            if 'lr_ratios' in sched:
                lr_info['lr_ratios'] = sched['lr_ratios']
    if 'optimizer_states' in ckpt:
        for i, opt_state in enumerate(ckpt['optimizer_states']):
            if 'param_groups' in opt_state:
                for j, pg in enumerate(opt_state['param_groups']):
                    lr_info[f'pg{j}_lr'] = pg.get('lr', '?')
                    lr_info[f'pg{j}_initial_lr'] = pg.get('initial_lr', '?')

    return {'weights': results, 'lr_info': lr_info, 'state_dict': sd}


def main():
    if len(sys.argv) < 2:
        sys.exit(1)

    ckpt_dir = sys.argv[1]

    patterns = [os.path.join(ckpt_dir, '*.ckpt')]
    ckpt_files = []
    for pattern in patterns:
        ckpt_files.extend(glob.glob(pattern))

    if not ckpt_files:
        print(f"No .ckpt files found in {ckpt_dir}")
        sys.exit(1)

    ckpt_files.sort(key=extract_epoch)

    print(f"Found {len(ckpt_files)} checkpoints in {ckpt_dir}\n")

    reference_sd = None
    first_results = None

    for i, filepath in enumerate(ckpt_files):
        epoch = extract_epoch(filepath)
        epoch_str = f"epoch={epoch}" if epoch < 99999 else "last"
        basename = os.path.basename(filepath)

        print(f"{'='*80}")
        print(f"[{basename}]  ({epoch_str})")
        print(f"{'='*80}")

        results = analyze_checkpoint(filepath, reference_sd)
        if results is None:
            print("  No pair_geometric_encoder keys found!\n")
            continue

        if reference_sd is None:
            reference_sd = results['state_dict']
            first_results = results

        # print weight stats
        for name, stats in results['weights'].items():
            print(f"  {name}:")
            print(f"    mean={stats['mean']:+.6f}  std={stats['std']:.6f}  "
                  f"min={stats['min']:+.6f}  max={stats['max']:+.6f}  norm={stats['norm']:.6f}")
            if 'diff_mean' in stats:
                print(f"    diff from first ckpt:  mean_abs_diff={stats['diff_mean']:.8f}  max_diff={stats['diff_max']:.8f}")

        if results['lr_info']:
            print(f"\n  LR info:")
            for k, v in results['lr_info'].items():
                print(f"    {k}: {v}")

        print()

    # summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    if reference_sd and len(ckpt_files) > 1:
        last_results = analyze_checkpoint(ckpt_files[-1], reference_sd)
        if last_results:
            print(f"\nWeight changes (first -> last checkpoint):")
            for name, stats in last_results['weights'].items():
                if 'diff_mean' in stats:
                    if stats['diff_mean'] < 1e-6:
                        verdict = "FROZEN — not learning at all"
                    elif stats['diff_mean'] < 1e-4:
                        verdict = "BARELY MOVING — LR may still be too low"
                    elif stats['diff_mean'] < 1e-2:
                        verdict = "LEARNING SLOWLY — some movement"
                    else:
                        verdict = "LEARNING — significant weight changes"
                    print(f"  {name}: mean_abs_diff={stats['diff_mean']:.8f} -> {verdict}")

if __name__ == '__main__':
    main()

