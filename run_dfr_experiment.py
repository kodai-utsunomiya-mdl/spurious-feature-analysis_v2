# run_dfr_experiment.py

import os
import yaml
import numpy as np
import glob
import re
import time
import datetime
import main

def parse_worst_acc_from_file(filepath):
    """
    dfr_results.txt から [DFR Model - Main Task (Predict Y)] セクションの
    Test Worst Acc を抽出する
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # セクション分け（Main Taskの結果は最初の方にあるはずだが，念のため分割）
        sections = content.split('=========================================')
        
        # "DFR Model - Main Task" を含むセクションを探す
        target_section = None
        for sec in sections:
            if "[DFR Model - Main Task (Predict Y)]" in sec:
                target_section = sec
                break
        
        if target_section is None:
            # 見つからない場合はファイルの先頭から探す（fallback）
            target_section = content

        # "Test Worst Acc: 0.1234" のようなパターンを探す
        match = re.search(r"Test Worst Acc:\s*([\d\.]+)", target_section)
        if match:
            return float(match.group(1))
        else:
            print(f"Warning: 'Test Worst Acc' not found in {filepath}")
            return None
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def run_experiment_loop():
    # ---------------------------------------------------------
    # 1. ベースとなる設定
    # ---------------------------------------------------------
    base_config_str = """
experiment_name: "test_wb_dfr_100update_identity"
dataset_name: "WaterBirds"
loss_function: "mse"
activation_function: "softplus"
initialization_method: "muP"
use_skip_connections: false
use_bias: true
use_zero_bias_initialization: true
use_grayscale: false
device: "cuda"

num_train_samples: 4795
num_test_samples: 5794
remove_minority_groups_train: true

use_feature_extractor: true
feature_extractor_model_name: "DINOv2_ViT_G_14"
feature_extractor_resnet_intermediate_layer: "avgpool"
feature_extractor_resnet_pooling_output_size: 2
feature_extractor_vit_target_block: -1
feature_extractor_vit_aggregation_mode: "mean_pool_all"

train_correlation: 0.8
train_label_marginal: 0.0
train_attribute_marginal: 0.0
test_correlation: 0.0
test_label_marginal: 0.0
test_attribute_marginal: 0.0

num_residual_blocks: 5
num_hidden_layers: 1
hidden_dim: 1024

epochs: 30
train_batch_size: 64
eval_batch_size: 512
optimizer: "Adam"
learning_rate: 0.01
momentum: 0.0
fix_final_layer: false

debias_method: "None"
use_mixup: false
mixup_alpha: 1.0

dro_eta_q: 0.01

use_kernel_regularization: false
kernel_reg_weight_sameA_diffY: 0.5
kernel_reg_weight_diffA_diffY: 0.0
kernel_reg_weight_diffA_sameY: 0.5

use_decov_regularization: false
decov_reg_weight: 0.00001
decov_target_identity: false

use_cosine_regularization: false
cosine_reg_weight_sameA_diffY: 1.0
cosine_reg_weight_diffA_diffY: 1.0
cosine_reg_weight_diffA_sameY: 1.0

regularization_end_epoch: 20000
regularization_decay_start_epoch: 300

gap_dynamics_factors_analysis_epochs:
jacobian_norm_analysis_epochs:
static_dynamic_decomposition_analysis_epochs:
umap_analysis_epochs: [0, 10, 20, 30, 50, 100, 200, 300, 400, 500, 700, 1000, 1500, 2000, 5000, 7000, 10000]

gap_dynamics_factors:
  group_pairs:
    - [[-1, 1], [-1, -1]]
    - [[1, -1], [-1, -1]]
    - [[ 1,  -1], [ 1, 1]]

static_dynamic_decomposition:
  group_pairs:
    - [[-1, 1], [-1, -1]]
    - [[1, -1], [1, 1]]

analysis_target: 'both'
jacobian_num_samples: 50000
gradient_gram_num_samples: null 
show_and_save_samples: false

visualization_method: "umap"
analyze_umap_representation: true
umap_analysis_target: 'both'
umap_num_samples: 50000
umap_n_neighbors: 15
umap_min_dist: 0.1

analyze_singular_values: true
singular_values_analysis_target: 'both'
singular_values_num_samples: 2000

tsne_perplexity: 30.0
tsne_learning_rate: 200.0

analyze_jacobian_norm: false
analyze_gradient_basis: false
analyze_gap_dynamics_factors: false
analyze_static_dynamic_decomposition: false
analyze_model_output_expectation: true

wandb:
  enable: true
  project: "sp_exp_test"
  entity: "mdl-tsukuba"

use_dfr: true
dfr_loss_type: "mse"
dfr_reg: "l2"
dfr_c_options: [2.0, 1.7, 1.5, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01]
dfr_num_retrains: 30
dfr_val_samples_per_group: 133
dfr_target_layer: "last_hidden"
dfr_method: "standard"
dfr_minimax_step_size: 0.01
dfr_minimax_iterations: 1000
dfr_minimax_num_bootstraps: 10
dfr_standardization: false
dfr_standardization_source: "validation"
"""
    
    # ベースの設定を辞書としてロード
    base_config = yaml.safe_load(base_config_str)
    
    # ---------------------------------------------------------
    # 2. 実験ループの設定
    # ---------------------------------------------------------
    methods = ["standard", "minimax"]
    num_trials = 10
    
    results = {
        "standard": [],
        "minimax": []
    }
    
    # 一時的な設定ファイル名
    temp_config_path = "temp_config_experiment.yaml"

    print(f"Starting experiment: {num_trials} trials for each method {methods}...")
    
    for method in methods:
        print(f"\n{'#'*40}")
        print(f" Running method: {method.upper()}")
        print(f"{'#'*40}")
        
        for i in range(1, num_trials + 1):
            # -----------------------------------------------------
            # 設定の更新
            # -----------------------------------------------------
            config = base_config.copy()
            
            # 実験名をユニークにする (結果ディレクトリの特定のため)
            exp_name = f"exp_{method}_trial_{i:02d}"
            config['experiment_name'] = exp_name
            
            # DFRメソッドの設定
            config['dfr_method'] = method
            
            # WandBのログ名も見分けやすくする
            # config['wandb']['enable'] = False # 必要に応じてFalseに
            
            # -----------------------------------------------------
            # 設定ファイルの保存
            # -----------------------------------------------------
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f)
            
            print(f"\n[Method: {method} | Trial: {i}/{num_trials}] Starting...")
            
            # -----------------------------------------------------
            # main.py の実行
            # -----------------------------------------------------
            try:
                # main関数を直接呼び出す
                main.main(config_path=temp_config_path)
                
                # -----------------------------------------------------
                # 結果の取得
                # -----------------------------------------------------
                # 結果ディレクトリは "results/experiment_name_timestamp" の形式
                # 作成された最新のディレクトリを探す
                search_pattern = os.path.join("results", f"{exp_name}_*")
                dirs = glob.glob(search_pattern)
                
                if not dirs:
                    print(f"Error: Result directory for {exp_name} not found.")
                    continue
                
                # タイムスタンプ順にソートして最新を取得
                latest_dir = sorted(dirs)[-1]
                result_file = os.path.join(latest_dir, "dfr_results.txt")
                
                if os.path.exists(result_file):
                    acc = parse_worst_acc_from_file(result_file)
                    if acc is not None:
                        results[method].append(acc)
                        print(f"  -> Trial {i} Result (Worst Acc): {acc:.4f}")
                    else:
                        print(f"  -> Trial {i} Failed to parse accuracy.")
                else:
                    print(f"Error: {result_file} not found.")
            
            except Exception as e:
                print(f"Error during trial {i}: {e}")
                import traceback
                traceback.print_exc()
            
            # 少し待機（ファイルIO等の安定のため）
            time.sleep(1)

    # ---------------------------------------------------------
    # 3. 集計と表示・保存
    # ---------------------------------------------------------
    summary_lines = []
    summary_lines.append("="*50)
    summary_lines.append(f" FINAL RESULTS Summary ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    summary_lines.append("="*50)
    
    for method in methods:
        accs = np.array(results[method])
        count = len(accs)
        if count > 0:
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            summary_lines.append(f"\nMethod: {method.upper()}")
            summary_lines.append(f"  Trials: {count}/{num_trials}")
            summary_lines.append(f"  Worst Group Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            summary_lines.append(f"  Raw Values: {accs.tolist()}")
        else:
            summary_lines.append(f"\nMethod: {method.upper()}")
            summary_lines.append(f"  No valid results found.")

    # まとめたテキストを作成
    summary_text = "\n".join(summary_lines)

    # 1. 画面に表示
    print("\n" + summary_text)

    # 2. ファイルに保存
    summary_filename = "final_experiment_summary.txt"
    with open(summary_filename, "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    print(f"\nSummary saved to: {summary_filename}")

    # 一時ファイルの削除
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)

if __name__ == "__main__":
    run_experiment_loop()
