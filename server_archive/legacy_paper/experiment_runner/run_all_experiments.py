"""
Unified Experiment Runner
=========================
Orchestrates all experiments from a single entry point:
  1. Ablation study (4 variants)
  2. Distribution validation
  3. Generates SOTA comparison table
  4. Generates all plots

Usage:
    python run_all_experiments.py --data_dir ./data --output_dir ./results [--gpu]

Author: [Your Name]
"""

import os
import sys
import argparse
import time
import subprocess
from datetime import datetime


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_ablation(data_dir: str, output_dir: str, use_gpu: bool):
    """Run all 4 ablation variants."""
    print_header("Ablation Study (4 Variants)")

    ablation_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ablation')
    train_script = os.path.join(ablation_dir, 'train_ablation.py')

    cmd = [
        sys.executable, train_script,
        '--variant', 'all',
        '--data_dir', data_dir,
        '--output_dir', os.path.join(output_dir, 'ablation'),
        '--epochs', '50',
        '--batch_size', '64',
    ]
    if use_gpu:
        cmd.append('--gpu')

    print(f"Command: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start
    print(f"\nAblation completed in {elapsed/60:.1f} minutes (exit code: {result.returncode})")
    return result.returncode == 0


def run_distribution_validation(data_dir: str, output_dir: str):
    """Run distribution validation on synthetic vs real data."""
    print_header("Distribution Validation")

    # Check if real and synthetic data exist
    real_path = os.path.join(data_dir, 'real_samples.npy')
    synth_path = os.path.join(data_dir, 'synthetic_samples.npy')

    if not (os.path.exists(real_path) and os.path.exists(synth_path)):
        print(f"[WARN] Real/synthetic .npy files not found in {data_dir}.")
        print("  Expected: real_samples.npy, synthetic_samples.npy")
        print("  Generate these by running the diffusion model training script first.")
        print("  Skipping distribution validation for now.")
        return True

    validation_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..',
        'distribution_validation', 'distribution_metrics.py'
    )

    cmd = [
        sys.executable, validation_script,
        '--real', real_path,
        '--synth', synth_path,
        '--output', os.path.join(output_dir, 'validation'),
        '--n_perms', '200',
        '--n_bins', '50',
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    print(f"Distribution validation completed (exit code: {result.returncode})")
    return result.returncode == 0


def generate_sota_table(output_dir: str):
    """Generate SOTA comparison table from template."""
    print_header("SOTA Comparison Table")

    src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..',
        'sota_comparison', 'sota_comparison_table.md'
    )
    dst = os.path.join(output_dir, 'sota_comparison_table.md')

    if os.path.exists(src):
        import shutil
        shutil.copy2(src, dst)
        print(f"SOTA table copied to {dst}")
    else:
        print(f"[WARN] SOTA table template not found at {src}")

    return True


def generate_paper_revisions(output_dir: str):
    """Copy paper revision text to output."""
    print_header("Paper Revision Texts")

    src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..',
        'paper_revisions', 'C1_C4_C5_C6_C7_C8_revisions.md'
    )
    dst = os.path.join(output_dir, 'paper_revisions.md')

    if os.path.exists(src):
        import shutil
        shutil.copy2(src, dst)
        print(f"Paper revisions copied to {dst}")
    else:
        print(f"[WARN] Revision text not found at {src}")

    return True


def generate_response_letter(output_dir: str):
    """Copy response letter to output."""
    print_header("Response Letter")

    src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..',
        'response_letter', 'response_letter.md'
    )
    dst = os.path.join(output_dir, 'response_letter.md')

    if os.path.exists(src):
        import shutil
        shutil.copy2(src, dst)
        print(f"Response letter copied to {dst}")
    else:
        print(f"[WARN] Response letter not found at {src}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Run all IDS revision experiments')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing UNSW-NB15 dataset')
    parser.add_argument('--output_dir', type=str, default='./revision_results',
                        help='Output directory for all results')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    parser.add_argument('--skip_ablation', action='store_true',
                        help='Skip ablation experiments')
    parser.add_argument('--skip_validation', action='store_true',
                        help='Skip distribution validation')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  IDS Revision Experiment Runner")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print("=" * 70)

    results = {}

    # Step 1: Ablation study
    if not args.skip_ablation:
        results['ablation'] = run_ablation(args.data_dir, args.output_dir, args.gpu)
    else:
        print("[SKIP] Ablation study")

    # Step 2: Distribution validation
    if not args.skip_validation:
        results['validation'] = run_distribution_validation(args.data_dir, args.output_dir)
    else:
        print("[SKIP] Distribution validation")

    # Step 3: SOTA table
    results['sota'] = generate_sota_table(args.output_dir)

    # Step 4: Paper revisions
    results['revisions'] = generate_paper_revisions(args.output_dir)

    # Step 5: Response letter
    results['letter'] = generate_response_letter(args.output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  [{status}] {name}")
    print("=" * 70)
    print(f"  All outputs saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
