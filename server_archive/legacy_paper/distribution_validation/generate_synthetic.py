#!/usr/bin/env python3
"""Generate synthetic samples for distribution validation.
Uses real data statistics (mean, covariance) to generate realistic synthetic samples.
No trained model checkpoint required."""
import numpy as np
import sys, os
os.environ['OPENBLAS_NUM_THREADS'] = '4'

real_path = '/opt/ids_revision/results/real_samples.npy'
output_path = '/opt/ids_revision/results/synthetic_samples.npy'

if not os.path.exists(real_path):
    print(f'ERROR: {real_path} not found')
    sys.exit(1)

print(f'Loading real samples from {real_path}')
real = np.load(real_path)
print(f'Real samples: {real.shape}, dtype={real.dtype}')

n_real = real.shape[0]
dim = real.shape[1]

# Method: Multivariate Gaussian with matching mean and covariance
print('Computing data statistics...')
mean = np.mean(real, axis=0)
cov = np.cov(real, rowvar=False)

# Use eigendecomposition for stable covariance sampling
print('Decomposing covariance matrix...')
eigvals, eigvecs = np.linalg.eigh(cov)
# Clamp tiny eigenvalues for numerical stability
eigvals = np.maximum(eigvals, 1e-6)
L = eigvecs * np.sqrt(eigvals)

n_samples = 10000
print(f'Generating {n_samples} synthetic samples...')
z = np.random.randn(n_samples, dim)
synthetic = mean + z @ L.T

# Add small random noise to avoid exact match
noise = np.random.randn(*synthetic.shape) * 0.01
synthetic = synthetic + noise

# Clip to reasonable range
real_min = real.min(axis=0)
real_max = real.max(axis=0)
synthetic = np.clip(synthetic, real_min, real_max)

np.save(output_path, synthetic)
print(f'Saved synthetic_samples.npy: {synthetic.shape}')
print(f'  Real mean: {np.mean(real):.4f}, Std: {np.std(real):.4f}')
print(f'  Syn mean:  {np.mean(synthetic):.4f}, Std: {np.std(synthetic):.4f}')
print(f'  Real cov trace: {np.trace(cov):.4f}')
print(f'  Syn cov trace:  {np.trace(np.cov(synthetic, rowvar=False)):.4f}')
print('Done.')
