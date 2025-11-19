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

# ヤコビアンノルムのプロット関数
def plot_jacobian_norm_evolution(history_train, history_test, save_dir):
    if not history_train and not history_test: return
    epochs = sorted(history_train.keys() if history_train else history_test.keys())

    fig1, ax1 = plt.subplots(figsize=(12, 7))
    fig1.suptitle('Evolution of Jacobian Norms (η-weighted)', fontsize=16)
    # 履歴からキーをチェック
    first_epoch_data = history_train.get(epochs[0], {}) or history_test.get(epochs[0], {})
    norm_keys = sorted([k for k in first_epoch_data.keys() if k.startswith('norm_G(')])
    
    colors = plt.cm.jet(np.linspace(0, 1, len(norm_keys)))
    for i, key in enumerate(norm_keys):
        if history_train:
            vals = [history_train.get(e, {}).get(key, np.nan) for e in epochs]
            ax1.plot(epochs, vals, marker='o', linestyle='-', color=colors[i], label=f'{key} (Train)')
        if history_test:
            vals = [history_test.get(e, {}).get(key, np.nan) for e in epochs]
            ax1.plot(epochs, vals, marker='x', linestyle='--', color=colors[i], label=f'{key} (Test)')
    ax1.set(xlabel='Epoch', ylabel='Squared Norm (log)', yscale='log')
    ax1.legend(); ax1.grid(True, which="both", ls="--")
    _save_and_close(fig1, save_dir, 'jacobian_norms.png')

    dot_keys = sorted([k for k in first_epoch_data.keys() if k.startswith('dot_G(')])
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    fig2.suptitle('Evolution of Jacobian Inner Products (η-weighted)', fontsize=16)
    colors = plt.cm.jet(np.linspace(0, 1, len(dot_keys)))
    for i, key in enumerate(dot_keys):
        if history_train:
            vals = [history_train.get(e, {}).get(key, np.nan) for e in epochs]
            ax2.plot(epochs, vals, marker='o', linestyle='-', color=colors[i], label=f'{key} (Train)')
        if history_test:
            vals = [history_test.get(e, {}).get(key, np.nan) for e in epochs]
            ax2.plot(epochs, vals, marker='x', linestyle='--', color=colors[i], label=f'{key} (Test)')
    ax2.set(xlabel='Epoch', ylabel='Inner Product (log)', yscale='symlog')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5)); ax2.grid(True, which="both", ls="--")
    fig2.tight_layout(rect=[0, 0, 0.8, 0.95])
    _save_and_close(fig2, save_dir, 'jacobian_inner_products.png')

    # --- 幾何学的中心のメトリクス ---
    delta_norm_keys = sorted([k for k in first_epoch_data.keys() if k.startswith('norm_Delta_S') or k.startswith('norm_Delta_L')])
    delta_dot_keys = sorted([k for k in first_epoch_data.keys() if k.startswith('dot_Delta_S_Delta_L')])
    delta_cosine_keys = sorted([k for k in first_epoch_data.keys() if k.startswith('cosine_Delta_S_Delta_L')])
    paper_norm_keys = sorted([k for k in first_epoch_data.keys() if k.startswith('paper_norm_sq_m_')])
    paper_dot_keys = sorted([k for k in first_epoch_data.keys() if k.startswith('paper_dot_m_A_m_Y')])


    # キーが存在する場合のみプロット
    if delta_norm_keys or delta_dot_keys or delta_cosine_keys or paper_norm_keys or paper_dot_keys:
        
        num_plots = 3 # Norms, Dots/PaperTerms, Cosines
        fig3, axes3 = plt.subplots(num_plots, 1, figsize=(14, 7 * num_plots))
        if num_plots == 1: axes3 = [axes3] # Make iterable
        fig3.suptitle('Evolution of Jacobian Geometric Center Metrics (S/L)', fontsize=16)

        plot_idx = 0

        # --- Delta S/L ノルム ---
        if delta_norm_keys:
            ax = axes3[plot_idx]
            colors = plt.cm.autumn(np.linspace(0, 1, len(delta_norm_keys)))
            for i, key in enumerate(delta_norm_keys):
                if history_train:
                    vals = [history_train.get(e, {}).get(key, np.nan) for e in epochs]
                    ax.plot(epochs, vals, marker='o', linestyle='-', color=colors[i], label=f'{key} (Train)')
                if history_test:
                    vals = [history_test.get(e, {}).get(key, np.nan) for e in epochs]
                    ax.plot(epochs, vals, marker='x', linestyle='--', color=colors[i], label=f'{key} (Test)')
            ax.set(xlabel='Epoch', ylabel='Norm (log)', title='Geometric Center Norms (||ΔS||, ||ΔL||)', yscale='log')
            ax.legend(); ax.grid(True, which="both", ls="--")
            plot_idx += 1

        # --- Delta S/L 内積 ---
        all_dot_keys = delta_dot_keys + paper_dot_keys + paper_norm_keys
        if all_dot_keys:
            ax = axes3[plot_idx]
            colors = plt.cm.jet(np.linspace(0, 1, len(all_dot_keys)))
            for i, key in enumerate(all_dot_keys):
                if history_train:
                    vals = [history_train.get(e, {}).get(key, np.nan) for e in epochs]
                    ax.plot(epochs, vals, marker='o', linestyle='-', color=colors[i], label=f'{key} (Train)')
                if history_test:
                    vals = [history_test.get(e, {}).get(key, np.nan) for e in epochs]
                    ax.plot(epochs, vals, marker='x', linestyle='--', color=colors[i], label=f'{key} (Test)')
            ax.set(xlabel='Epoch', ylabel='Inner Product / Norm^2 (symlog)', title='Geometric Center Inner Products & Paper Terms', yscale='symlog')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)); ax.grid(True, which="both", ls="--")
            plot_idx += 1

        # --- Delta S/L コサイン類似度 ---
        if delta_cosine_keys:
            ax = axes3[plot_idx]
            colors = plt.cm.winter(np.linspace(0, 1, len(delta_cosine_keys)))
            for i, key in enumerate(delta_cosine_keys):
                if history_train:
                    vals = [history_train.get(e, {}).get(key, np.nan) for e in epochs]
                    ax.plot(epochs, vals, marker='o', linestyle='-', color=colors[i], label=f'{key} (Train)')
                if history_test:
                    vals = [history_test.get(e, {}).get(key, np.nan) for e in epochs]
                    ax.plot(epochs, vals, marker='x', linestyle='--', color=colors[i], label=f'{key} (Test)')
            ax.set(xlabel='Epoch', ylabel='Cosine Similarity', title='Geometric Center Alignment (cos(ΔS, ΔL))', ylim=(-1.05, 1.05))
            ax.legend(); ax.grid(True, which="both", ls="--")
            plot_idx += 1
        
        fig3.tight_layout(rect=[0, 0, 0.85, 0.96])
        _save_and_close(fig3, save_dir, 'jacobian_geometric_centers.png')

# ==============================================================================
# 静的・動的分解 (項A, B, C) のプロット関数
# ==============================================================================
def plot_static_dynamic_decomposition(history_train, history_test, config, save_dir):
    """
    同じラベルを持つグループ間の性能差ダイナミクスの静的・動的分解 (項A, B, C) の
    時間発展をプロットする
    """
    if not history_train and not history_test: return
    
    epochs = sorted(history_train.keys() if history_train else history_test.keys())
    if not epochs: return
        
    first_epoch_data = history_train.get(epochs[0], {}) or history_test.get(epochs[0], {})
    
    # config からペア情報を取得
    group_pairs_config = config.get('static_dynamic_decomposition', {}).get('group_pairs', [])
    if not group_pairs_config: return

    num_pairs = len(group_pairs_config)
    fig, axes = plt.subplots(num_pairs, 1, figsize=(14, 7 * num_pairs), squeeze=False)
    fig.suptitle('Evolution of Static/Dynamic Decomposition Terms (A, B, C)', fontsize=16)

    for i, pair in enumerate(group_pairs_config):
        g_min_key = tuple(pair[0])
        g_maj_key = tuple(pair[1])
        pair_name = f"g_min_{g_min_key}_g_maj_{g_maj_key}"
        
        ax = axes[i, 0]
        
        term_keys = [f'{pair_name}_TermA', f'{pair_name}_TermB', f'{pair_name}_TermC']
        colors = ['r', 'g', 'b']
        
        for j, key in enumerate(term_keys):
            if history_train:
                vals = [history_train.get(e, {}).get(key, np.nan) for e in epochs]
                ax.plot(epochs, vals, marker='o', linestyle='-', color=colors[j], label=f'{key} (Train)')
            if history_test:
                vals = [history_test.get(e, {}).get(key, np.nan) for e in epochs]
                ax.plot(epochs, vals, marker='x', linestyle='--', color=colors[j], label=f'{key} (Test)')

        # 合計 (A+B+C) もプロット
        if history_train:
            vals_sum_train = [sum(history_train.get(e, {}).get(k, 0) for k in term_keys) for e in epochs]
            ax.plot(epochs, vals_sum_train, marker='o', linestyle='-', color='k', label=f'{pair_name}_Sum(A+B+C) (Train)')
        if history_test:
            vals_sum_test = [sum(history_test.get(e, {}).get(k, 0) for k in term_keys) for e in epochs]
            ax.plot(epochs, vals_sum_test, marker='x', linestyle='--', color='k', label=f'{pair_name}_Sum(A+B+C) (Test)')

        ax.set(xlabel='Epoch', ylabel='Inner Product (symlog)', yscale='symlog',
               title=f"Decomposition for Pair: min={g_min_key} vs maj={g_maj_key}")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, which="both", ls="--")

    fig.tight_layout(rect=[0, 0, 0.85, 0.96])
    _save_and_close(fig, save_dir, 'static_dynamic_decomposition.png')


# ==============================================================================
# モデル出力期待値のプロット関数
# ==============================================================================
def plot_model_output_expectations(history_train, history_test, save_dir):
    """各グループのモデル出力期待値 E[f(x)] の時間変化をプロット"""
    if not history_train and not history_test: return
    
    epochs = sorted(history_train.keys() if history_train else history_test.keys())
    first_epoch_data = history_train.get(epochs[0], {}) or history_test.get(epochs[0], {})
    keys = sorted([k for k in first_epoch_data.keys() if k.startswith('E[f(x)]')])
    
    # グループごとの色定義
    # (-1,-1): Cyan, (-1,1): Blue, (1,-1): Orange, (1,1): Red
    color_map = {
        '(-1,-1)': 'cyan', '(-1,1)': 'blue', 
        '(1,-1)': 'orange', '(1,1)': 'red'
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle('Evolution of Model Output Expectations E[f(x)]', fontsize=16)
    
    for key in keys:
        # キーからグループ特定 (例: "E[f(x)]_G(-1,-1)")
        group_str = key.split('_G')[-1] # "(-1,-1)"
        color = color_map.get(group_str, 'gray')
        
        if history_train:
            vals = [history_train.get(e, {}).get(key, np.nan) for e in epochs]
            ax.plot(epochs, vals, marker='o', linestyle='-', color=color, label=f'{key} (Train)')
        if history_test:
            vals = [history_test.get(e, {}).get(key, np.nan) for e in epochs]
            ax.plot(epochs, vals, marker='x', linestyle='--', color=color, label=f'{key} (Test)')
            
    ax.set(xlabel='Epoch', ylabel='E[f(x)]')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, which="both", ls="--")
    
    fig.tight_layout(rect=[0, 0, 0.85, 0.96])
    _save_and_close(fig, save_dir, 'model_output_expectations.png')


# ==============================================================================
# 全てのプロットを統括するラッパー関数
# ==============================================================================

def plot_all_results(history_df, analysis_histories, layers, save_dir, config):
    """全てのプロット関数を呼び出し，結果を保存"""
    print("\n--- Generating and saving all plots ---")
    
    plot_training_history(history_df, save_dir)
    plot_misclassification_rates(history_df.iloc[-1], config['dataset_name'], save_dir)
    
    if config.get('analyze_jacobian_norm', False):
        plot_jacobian_norm_evolution(
            analysis_histories.get('jacobian_norm_train', {}), 
            analysis_histories.get('jacobian_norm_test', {}), 
            save_dir
        )

    # 静的・動的分解のプロット
    if config.get('analyze_static_dynamic_decomposition', False):
        plot_static_dynamic_decomposition(
            analysis_histories.get('static_dynamic_decomp_train', {}),
            analysis_histories.get('static_dynamic_decomp_test', {}),
            config,
            save_dir
        )

    # モデル出力期待値のプロット
    if config.get('analyze_model_output_expectation', False):
        plot_model_output_expectations(
            analysis_histories.get('model_output_exp_train', {}),
            analysis_histories.get('model_output_exp_test', {}),
            save_dir
        )
