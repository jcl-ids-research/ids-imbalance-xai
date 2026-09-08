"""保真度评估脚本：KS 检验 + 分布距离指标."""
import numpy as np
import json
from pathlib import Path
from scipy import stats
from scipy.spatial import distance

def ks_test_features(X_real, X_syn):
    """
    对每个特征进行 KS 检验.
    
    返回:
        ks_stats: KS 统计量列表
        p_values: p 值列表
        pass_rate: 通过率（p > 0.05）
    """
    n_features = X_real.shape[1]
    ks_stats = []
    p_values = []
    
    for i in range(n_features):
        stat, p = stats.ks_2samp(X_real[:, i], X_syn[:, i])
        ks_stats.append(float(stat))
        p_values.append(float(p))
    
    pass_rate = np.mean([p > 0.05 for p in p_values])
    
    return {
        'ks_statistics': ks_stats,
        'p_values': p_values,
        'pass_rate': float(pass_rate),
        'mean_ks': float(np.mean(ks_stats)),
        'max_ks': float(np.max(ks_stats))
    }

def wasserstein_distance_features(X_real, X_syn):
    """计算每个特征的 Wasserstein 距离."""
    n_features = X_real.shape[1]
    distances = []
    
    for i in range(n_features):
        wd = stats.wasserstein_distance(X_real[:, i], X_syn[:, i])
        distances.append(float(wd))
    
    return {
        'wasserstein_distances': distances,
        'mean_wasserstein': float(np.mean(distances)),
        'max_wasserstein': float(np.max(distances))
    }

def statistical_similarity(X_real, X_syn):
    """统计量对比：mean, std, min, max."""
    return {
        'mean_diff': float(np.mean(np.abs(X_real.mean(axis=0) - X_syn.mean(axis=0)))),
        'std_diff': float(np.mean(np.abs(X_real.std(axis=0) - X_syn.std(axis=0)))),
        'real_mean': float(X_real.mean()),
        'syn_mean': float(X_syn.mean()),
        'real_std': float(X_real.std()),
        'syn_std': float(X_syn.std())
    }

def evaluate_fidelity(samples_path):
    """
    评估单个数据集的保真度.
    
    Args:
        samples_path: samples.npz 文件路径
    
    Returns:
        dict: 保真度指标
    """
    data = np.load(samples_path)
    
    X_real = data['X_real']
    X_syn = data['X_syn']
    y_real = data['y_real']
    y_syn = data['y_syn']
    
    print(f"  Real samples: {X_real.shape}")
    print(f"  Synthetic samples: {X_syn.shape}")
    
    # KS 检验
    ks_results = ks_test_features(X_real, X_syn)
    print(f"  KS test pass rate: {ks_results['pass_rate']:.2%}")
    print(f"  Mean KS statistic: {ks_results['mean_ks']:.4f}")
    
    # Wasserstein 距离
    wd_results = wasserstein_distance_features(X_real, X_syn)
    print(f"  Mean Wasserstein distance: {wd_results['mean_wasserstein']:.4f}")
    
    # 统计量对比
    stats_results = statistical_similarity(X_real, X_syn)
    print(f"  Mean diff: {stats_results['mean_diff']:.4f}")
    print(f"  Std diff: {stats_results['std_diff']:.4f}")
    
    # 类别分布对比
    real_class_dist = np.bincount(y_real) / len(y_real)
    syn_class_dist = np.bincount(y_syn) / len(y_syn)
    
    return {
        'ks_test': ks_results,
        'wasserstein': wd_results,
        'statistics': stats_results,
        'class_distribution': {
            'real': real_class_dist.tolist(),
            'synthetic': syn_class_dist.tolist()
        },
        'sample_counts': {
            'real': int(len(X_real)),
            'synthetic': int(len(X_syn))
        }
    }

def main():
    """主函数 - 部署到服务器上运行."""
    import sys
    import os
    
    # 在服务器上运行时的路径
    base_dir = Path('/opt/ids_revision/deploy/results/phase1')
    output_dir = Path('/opt/ids_revision/results/fidelity')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not base_dir.exists():
        print(f"[ERROR] Results directory not found: {base_dir}")
        return 1
    
    datasets = ['unsw', 'nsl', 'cic17', 'cic19']
    seeds = [42, 123, 456]
    
    all_results = {}
    
    print("=" * 80)
    print("保真度评估（Fidelity Evaluation）")
    print("=" * 80)
    
    for dataset in datasets:
        print(f"\n[Dataset: {dataset.upper()}]")
        dataset_results = {}
        
        for seed in seeds:
            samples_path = base_dir / dataset / f"seed{seed}" / "samples.npz"
            
            if not samples_path.exists():
                print(f"  [WARN] Missing: seed{seed}")
                continue
            
            print(f"\n  Seed {seed}:")
            try:
                fidelity = evaluate_fidelity(samples_path)
                dataset_results[f"seed{seed}"] = fidelity
            except Exception as e:
                print(f"  [ERROR] {e}")
                continue
        
        if dataset_results:
            all_results[dataset] = dataset_results
            
            # 计算平均值
            avg_ks_pass = np.mean([
                dataset_results[f"seed{seed}"]['ks_test']['pass_rate']
                for seed in seeds if f"seed{seed}" in dataset_results
            ])
            avg_ks = np.mean([
                dataset_results[f"seed{seed}"]['ks_test']['mean_ks']
                for seed in seeds if f"seed{seed}" in dataset_results
            ])
            avg_wd = np.mean([
                dataset_results[f"seed{seed}"]['wasserstein']['mean_wasserstein']
                for seed in seeds if f"seed{seed}" in dataset_results
            ])
            
            print(f"\n  [{dataset.upper()} Summary]")
            print(f"    KS pass rate (avg): {avg_ks_pass:.2%}")
            print(f"    Mean KS (avg): {avg_ks:.4f}")
            print(f"    Mean Wasserstein (avg): {avg_wd:.4f}")
    
    # 保存结果
    output_path = output_dir / "fidelity_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"[OK] Results saved to: {output_path}")
    print("=" * 80)
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
