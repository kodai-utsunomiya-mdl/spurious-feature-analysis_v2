# sp/plotting.py

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd

ot = None 

plt.rc("figure", dpi=100, facecolor=(1, 1, 1))
plt.rc("font", family='stixgeneral', size=13)
plt.rc("axes", facecolor='white', titlesize=16)
plt.rc("mathtext", fontset='cm')
plt.rc('text', usetex=False)

# ==============================================================================
# 共通ヘルパー関数
# ==============================================================================

def _save_and_close(fig, save_dir, filename):
    """
    MatplotlibのFigureを指定されたディレクトリにファイルとして保存し，メモリを解放
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {save_path}")

# ==============================================================================
# 学習履歴プロット関数
# ==============================================================================

def plot_training_history(history_df, save_dir):
    """学習過程の各種メトリクスをプロットし，2つの画像ファイルとして保存"""
    epochs = history_df.index + 1
    
    # --- Figure 1: 平均とワーストグループのメトリクス ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle('Overall and Worst-Group Metrics Over Epochs', fontsize=16)

    axes1[0, 0].plot(epochs, history_df['train_avg_loss'], 'b-', label='Train Avg Loss')
    axes1[0, 0].plot(epochs, history_df['test_avg_loss'], 'b--', label='Test Avg Loss')
    axes1[0, 0].set(title='Average Loss', xlabel='Epochs', ylabel='Loss')
    axes1[0, 0].legend(); axes1[0, 0].grid(True)
    axes1[0, 1].plot(epochs, history_df['train_avg_acc'], 'r-', label='Train Avg Accuracy')
    axes1[0, 1].plot(epochs, history_df['test_avg_acc'], 'r--', label='Test Avg Accuracy')
    axes1[0, 1].set(title='Average Accuracy', xlabel='Epochs', ylabel='Accuracy', ylim=(0, 1.05))
    axes1[0, 1].legend(); axes1[0, 1].grid(True)
    axes1[1, 0].plot(epochs, history_df['train_worst_loss'], 'g-', label='Train Worst-Group Loss')
    axes1[1, 0].plot(epochs, history_df['test_worst_loss'], 'g--', label='Test Worst-Group Loss')
    axes1[1, 0].set(title='Worst-Group Loss', xlabel='Epochs', ylabel='Loss')
    axes1[1, 0].legend(); axes1[1, 0].grid(True)
    axes1[1, 1].plot(epochs, history_df['train_worst_acc'], 'm-', label='Train Worst-Group Accuracy')
    axes1[1, 1].plot(epochs, history_df['test_worst_acc'], 'm--', label='Test Worst-Group Accuracy')
    axes1[1, 1].set(title='Worst-Group Accuracy', xlabel='Epochs', ylabel='Accuracy', ylim=(0, 1.05))
    axes1[1, 1].legend(); axes1[1, 1].grid(True)
    
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    _save_and_close(fig1, save_dir, "training_history_main.png")

    # --- Figure 2: グループごとのメトリクス ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 12))
    fig2.suptitle('Per-Group Metrics Over Epochs', fontsize=16)
    group_labels = {0: '$y=-1, a=-1$', 1: '$y=-1, a=+1$', 2: '$y=+1, a=-1$', 3: '$y=+1, a=+1$'}
    
    train_losses = np.array(history_df['train_group_losses'].tolist())
    test_losses = np.array(history_df['test_group_losses'].tolist())
    train_accs = np.array(history_df['train_group_accs'].tolist())
    test_accs = np.array(history_df['test_group_accs'].tolist())
    
    for i in range(4):
        r, c = i // 2, i % 2
        ax_loss = axes2[r, c]
        ax_acc = ax_loss.twinx()
        p1, = ax_loss.plot(epochs, train_losses[:, i], 'c-', label=f'Train Loss ({group_labels[i]})')
        p2, = ax_loss.plot(epochs, test_losses[:, i], 'c--', label=f'Test Loss ({group_labels[i]})')
        ax_loss.set_ylabel('Loss', color='c'); ax_loss.tick_params(axis='y', labelcolor='c')
        ax_loss.grid(True, axis='y', linestyle=':')
        p3, = ax_acc.plot(epochs, train_accs[:, i], 'y-', label=f'Train Acc ({group_labels[i]})')
        p4, = ax_acc.plot(epochs, test_accs[:, i], 'y--', label=f'Test Acc ({group_labels[i]})')
        ax_acc.set_ylabel('Accuracy', color='y'); ax_acc.tick_params(axis='y', labelcolor='y'); ax_acc.set_ylim(0, 1.05)
        ax_loss.set(title=f'Group: {group_labels[i]}', xlabel='Epochs')
        ax_loss.legend(handles=[p1, p2, p3, p4], loc='best')

    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    _save_and_close(fig2, save_dir, "training_history_groups.png")

def plot_misclassification_rates(final_metrics_series, dataset_name, save_dir):
    """最終テストセットのグループ別誤分類率をプロット"""
    print("\nPlotting misclassification rates...")
    if 'WaterBirds' in dataset_name:
        group_labels = ['Landbird\non Land\n($y=-1, a=-1$)', 'Landbird\non Water\n($y=-1, a=+1$)', 
                        'Waterbird\non Land\n($y=+1, a=-1$)', 'Waterbird\non Water\n($y=+1, a=+1$)']
    else: # ColoredMNIST
        group_labels = ['Digit<5, Green\n($y=-1, a=-1$)', 'Digit<5, Red\n($y=-1, a=+1$)',
                        'Digit>=5, Green\n($y=+1, a=-1$)', 'Digit>=5, Red\n($y=+1, a=+1$)']
    
    group_accs = final_metrics_series['test_group_accs']
    rates = [1 - (acc if not np.isnan(acc) else 0) for acc in group_accs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(group_labels, rates, color=['cyan', 'blue', 'orange', 'red'])
    ax.set(ylabel='Misclassification Rate', title='Final Test Set Misclassification Rate by Group')
    ax.set_ylim(0, max(1.05, max(rates) * 1.2 if rates else 1.05))
    ax.bar_label(bars, fmt='%.3f', fontsize=10, padding=3)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    fig.tight_layout()
    _save_and_close(fig, save_dir, "misclassification_rates.png")

# ==============================================================================
# 分析結果の時系列プロット関数
# ==============================================================================
# 勾配グラム行列のプロット関数
def plot_gradient_gram_evolution(history_train, history_test, save_dir):
    if not history_train and not history_test: return
    epochs = sorted(history_train.keys() if history_train else history_test.keys())
    
    group_keys = [(-1,-1), (-1,1), (1,-1), (1,1)]
    maj_maj_keys = [f"G({y1},{a1})_vs_G({y2},{a2})" for (y1, a1), (y2, a2) in combinations(group_keys, 2) if y1==a1 and y2==a2]
    min_min_keys = [f"G({y1},{a1})_vs_G({y2},{a2})" for (y1, a1), (y2, a2) in combinations(group_keys, 2) if y1!=a1 and y2!=a2]
    maj_min_keys = [f"G({y1},{a1})_vs_G({y2},{a2})" for (y1, a1), (y2, a2) in combinations(group_keys, 2) if (y1==a1 and y2!=a2) or (y1!=a1 and y2==a2)]
    
    plot_configs = [
        ('Majority-Majority', maj_maj_keys),
        ('Minority-Minority', min_min_keys),
        ('Majority-Minority', maj_min_keys)
    ]

    for title, keys in plot_configs:
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.suptitle(f'Evolution of Gradient Gram Matrix ({title})', fontsize=16)
        colors = plt.cm.jet(np.linspace(0, 1, len(keys)))

        for i, key in enumerate(keys):
            if history_train:
                vals = [history_train.get(e, {}).get(key, np.nan) for e in epochs]
                ax.plot(epochs, vals, marker='o', markersize=3, linestyle='-', color=colors[i], label=f'{key} (Train)')
            if history_test:
                vals = [history_test.get(e, {}).get(key, np.nan) for e in epochs]
                ax.plot(epochs, vals, marker='x', markersize=3, linestyle='--', color=colors[i])

        ax.set(xlabel='Epoch', ylabel='Inner Product (log)', yscale='symlog')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, which="both", ls="--")
        fig.tight_layout(rect=[0, 0, 0.85, 0.95])
        _save_and_close(fig, save_dir, f'gradient_gram_{title.lower().replace("-", "_")}.png')

# ヤコビアンノルムのプロット関数
def plot_jacobian_norm_evolution(history_train, history_test, save_dir):
    if not history_train and not history_test: return
    epochs = sorted(history_train.keys() if history_train else history_test.keys())
    
    # Norms
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    fig1.suptitle('Evolution of Jacobian Norms', fontsize=16)
    norm_keys = [k for k in next(iter(history_train.values())).keys() if k.startswith('norm')]
    colors = plt.cm.jet(np.linspace(0, 1, len(norm_keys)))
    for i, key in enumerate(norm_keys):
        if history_train:
            vals = [history_train.get(e, {}).get(key, np.nan) for e in epochs]
            ax1.plot(epochs, vals, marker='o', linestyle='-', color=colors[i], label=f'{key} (Train)')
        if history_test:
            vals = [history_test.get(e, {}).get(key, np.nan) for e in epochs]
            ax1.plot(epochs, vals, marker='x', linestyle='--', color=colors[i])
    ax1.set(xlabel='Epoch', ylabel='Squared Norm (log)', yscale='log')
    ax1.legend(); ax1.grid(True, which="both", ls="--")
    _save_and_close(fig1, save_dir, 'jacobian_norms.png')

    # Inner Products
    dot_keys = [k for k in next(iter(history_train.values())).keys() if k.startswith('dot')]
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    fig2.suptitle('Evolution of Jacobian Inner Products', fontsize=16)
    colors = plt.cm.jet(np.linspace(0, 1, len(dot_keys)))
    for i, key in enumerate(dot_keys):
        if history_train:
            vals = [history_train.get(e, {}).get(key, np.nan) for e in epochs]
            ax2.plot(epochs, vals, marker='o', linestyle='-', color=colors[i], label=f'{key} (Train)')
        if history_test:
            vals = [history_test.get(e, {}).get(key, np.nan) for e in epochs]
            ax2.plot(epochs, vals, marker='x', linestyle='--', color=colors[i])
    ax2.set(xlabel='Epoch', ylabel='Inner Product (log)', yscale='symlog')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5)); ax2.grid(True, which="both", ls="--")
    fig2.tight_layout(rect=[0, 0, 0.8, 0.95])
    _save_and_close(fig2, save_dir, 'jacobian_inner_products.png')

# 勾配グラム行列のスペクトルプロット関数
def plot_gradient_gram_spectrum_evolution(history_train, history_test, save_dir):
    """勾配グラム行列の固有値と主固有ベクトルの成分の変遷をプロット"""
    if not history_train and not history_test:
        print("No gradient gram spectrum history to plot.")
        return
    
    epochs = sorted(history_train.keys() if history_train else history_test.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle('Evolution of Gradient Gram Matrix Spectrum', fontsize=16)

    # --- 1. 固有値のプロット ---
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    for i in range(4):
        if history_train:
            vals = [history_train.get(e, {}).get('eigenvalues', [np.nan]*4)[i] for e in epochs]
            ax1.plot(epochs, vals, marker='o', markersize=3, linestyle='-', color=colors[i], label=f'λ_{i+1} (Train)')
        if history_test:
            vals = [history_test.get(e, {}).get('eigenvalues', [np.nan]*4)[i] for e in epochs]
            ax1.plot(epochs, vals, marker='x', markersize=3, linestyle='--', color=colors[i], label=f'λ_{i+1} (Test)')
    ax1.set(xlabel='Epoch', ylabel='Eigenvalue', title='Eigenvalues (λ₁ ≥ λ₂ ≥ λ₃ ≥ λ₄)', yscale='symlog', ylim_bottom=0)
    ax1.legend()
    ax1.grid(True, which="both", ls="--")

    # --- 2. 主固有ベクトルの成分のプロット ---
    ax2 = axes[1]
    group_labels = ['$G_{(-1,-1)}$', '$G_{(-1,1)}$', '$G_{(1,-1)}$', '$G_{(1,1)}$']
    for i in range(4):
        if history_train:
            vals = [history_train.get(e, {}).get('eigenvector1', [np.nan]*4)[i] for e in epochs]
            ax2.plot(epochs, vals, marker='o', markersize=3, linestyle='-', color=colors[i], label=f'{group_labels[i]} (Train)')
        if history_test:
            vals = [history_test.get(e, {}).get('eigenvector1', [np.nan]*4)[i] for e in epochs]
            ax2.plot(epochs, vals, marker='x', markersize=3, linestyle='--', color=colors[i], label=f'{group_labels[i]} (Test)')
    ax2.set(xlabel='Epoch', ylabel='Component Value', title='Components of 1st Eigenvector (u₁)', ylim=(-1.05, 1.05))
    ax2.legend()
    ax2.grid(True, which="both", ls="--")

    # --- 3. 2番目の固有ベクトルの成分のプロット ---
    ax3 = axes[2]
    for i in range(4):
        if history_train:
            vals = [history_train.get(e, {}).get('eigenvector2', [np.nan]*4)[i] for e in epochs]
            ax3.plot(epochs, vals, marker='o', markersize=3, linestyle='-', color=colors[i], label=f'{group_labels[i]} (Train)')
        if history_test:
            vals = [history_test.get(e, {}).get('eigenvector2', [np.nan]*4)[i] for e in epochs]
            ax3.plot(epochs, vals, marker='x', markersize=3, linestyle='--', color=colors[i], label=f'{group_labels[i]} (Test)')
    ax3.set(xlabel='Epoch', ylabel='Component Value', title='Components of 2nd Eigenvector (u₂)', ylim=(-1.05, 1.05))
    ax3.legend()
    ax3.grid(True, which="both", ls="--")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_and_close(fig, save_dir, 'gradient_gram_spectrum_evolution.png')

# 勾配ノルム比のプロット関数
def plot_gradient_norm_ratio_evolution(history_train, history_test, save_dir):
    """
    設定ファイルで指定されたすべての勾配ノルム比の変遷をプロット
    """
    if not history_train and not history_test:
        print("No gradient norm ratio history to plot.")
        return
        
    epochs = sorted(history_train.keys() if history_train else history_test.keys())
    if not epochs:
        print("No epochs found in gradient norm ratio history.")
        return

    # 履歴からプロット対象のキー (ratio_...) をすべて収集
    all_keys = set()
    # 最初の利用可能なエポックデータからキーを取得
    if history_train:
        first_epoch_data = history_train.get(epochs[0], {})
        all_keys.update(first_epoch_data.keys())
    if history_test:
        first_epoch_data = history_test.get(epochs[0], {})
        all_keys.update(first_epoch_data.keys())
    
    ratio_keys = sorted([k for k in all_keys if k.startswith('ratio_g(')])
    
    if not ratio_keys:
        print("No 'ratio_g(...' keys found in gradient norm ratio history.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle('Evolution of Gradient Norm Ratios', fontsize=16)
    
    colors = plt.cm.jet(np.linspace(0, 1, len(ratio_keys)))

    for i, key in enumerate(ratio_keys):
        # "ratio_g(-1, -1)_vs_g(-1, 1)" -> "g(-1, -1) / g(-1, 1)"
        label_name = key.replace('ratio_', '').replace('_vs_', ' / ')
        
        if history_train:
            vals = [history_train.get(e, {}).get(key, np.nan) for e in epochs]
            ax.plot(epochs, vals, marker='o', markersize=3, linestyle='-', color=colors[i], label=f'{label_name} (Train)')
        if history_test:
            vals = [history_test.get(e, {}).get(key, np.nan) for e in epochs]
            # TestはTrainと同じ色で破線
            ax.plot(epochs, vals, marker='x', markersize=3, linestyle='--', color=colors[i], label=f'{label_name} (Test)')

    ax.set(xlabel='Epoch', ylabel='Norm Ratio', yscale='log')
    # 凡例が重ならないようにグラフの外側に配置
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, which="both", ls="--")
    # 凡例スペースを確保
    fig.tight_layout(rect=[0, 0, 0.8, 0.95])
    _save_and_close(fig, save_dir, 'gradient_norm_ratio_evolution.png')


# ==============================================================================
# 全てのプロットを統括するラッパー関数
# ==============================================================================

def plot_all_results(history_df, analysis_histories, layers, save_dir, config):
    """全てのプロット関数を呼び出し，結果を保存"""
    print("\n--- Generating and saving all plots ---")
    
    plot_training_history(history_df, save_dir)
    plot_misclassification_rates(history_df.iloc[-1], config['dataset_name'], save_dir)

    if config.get('analyze_gradient_gram', False):
        plot_gradient_gram_evolution(analysis_histories['grad_gram_train'], analysis_histories['grad_gram_test'], save_dir)
    if config.get('analyze_jacobian_norm', False):
        plot_jacobian_norm_evolution(analysis_histories['jacobian_norm_train'], analysis_histories['jacobian_norm_test'], save_dir)
        
    if config.get('analyze_gradient_gram_spectrum', False):
        plot_gradient_gram_spectrum_evolution(
            analysis_histories['grad_gram_spectrum_train'],
            analysis_histories['grad_gram_spectrum_test'],
            save_dir
        )
    if config.get('analyze_gradient_norm_ratio', False):
        plot_gradient_norm_ratio_evolution(
            analysis_histories['grad_norm_ratio_train'],
            analysis_histories['grad_norm_ratio_test'],
            save_dir
        )
