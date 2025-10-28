# sp/main.py

import os
import yaml
import time
import shutil
import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import wandb

# スクリプトをインポート
import data_loader
import utils
import model as model_module
import analysis
import plotting

# --- ヘルパー関数 ---
def get_loss_function(scores, y_batch, loss_type='mse'):
    """ 損失関数を計算 """
    if loss_type == 'logistic':
        return F.softplus(-y_batch * scores).mean()
    elif loss_type == 'mse':
        return F.mse_loss(scores, y_batch)
    else:
        raise ValueError(f"Unknown loss_function: {loss_type}")

def main(config_path='config.yaml'):
    # 1. 設定ファイルの読み込み
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # --- debias_method の読み込み ---
    debias_method = config.get('debias_method', 'None')
    loss_function_name = config['loss_function']

    # wandbの初期化
    if config.get('wandb', {}).get('enable', False):
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            config=config,
            name=f"{config['experiment_name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        print("wandb is enabled and initialized.")

    # 2. 結果保存ディレクトリの作成
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_dir = os.path.join('results', f"{config['experiment_name']}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(result_dir, 'config_used.yaml'))
    print(f"Results will be saved to: {result_dir}")

    # デバイス設定
    device = config['device'] if torch.cuda.is_available() and config['device'] == 'cuda' else "cpu"
    print(f"Using device: {device}")

    # 3. データセットの準備
    print("\n--- 1. Preparing Dataset ---")
    if config['dataset_name'] == 'ColoredMNIST':
        image_size = 28

        train_y_bar = config.get('train_label_marginal', 0.0)
        train_a_bar = config.get('train_attribute_marginal', 0.0)
        test_y_bar = config.get('test_label_marginal', 0.0)
        test_a_bar = config.get('test_attribute_marginal', 0.0)

        X_train, y_train, a_train = data_loader.get_colored_mnist(
            num_samples=config['num_train_samples'],
            correlation=config['train_correlation'],
            label_marginal=train_y_bar,
            attribute_marginal=train_a_bar,
            train=True
        )
        X_test, y_test, a_test = data_loader.get_colored_mnist(
            num_samples=config['num_test_samples'],
            correlation=config['test_correlation'],
            label_marginal=test_y_bar,
            attribute_marginal=test_a_bar,
            train=False
        )

    elif config['dataset_name'] == 'WaterBirds':
        image_size = 224
        X_train, y_train, a_train, X_test, y_test, a_test = data_loader.get_waterbirds_dataset(
            num_train=config['num_train_samples'], num_test=config['num_test_samples'], image_size=image_size
        )
    else:
        raise ValueError(f"Unknown dataset: {config['dataset_name']}")

    if config['show_and_save_samples']:
        utils.show_dataset_samples(X_train, y_train, a_train, config['dataset_name'], result_dir)

    X_train = utils.l2_normalize_images(X_train)
    X_test = utils.l2_normalize_images(X_test)

    # --- グループのリストを定義 ---
    group_keys = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]

    # --- バイアス除去手法のための設定 ---
    static_weights = None
    dro_q_weights = None
    
    if debias_method == 'IW_uniform':
        print("\n--- Importance Weighting (Uniform Target) Enabled (Equivalent to v_inv) ---")
        print(f"  [Warning] 'batch_size' config is ignored. Using full-batch (per-group) gradient calculation.")
        print("  Removing both marginal bias (Term II) and spurious correlation (Term III).")

        # 理論に基づき，重みを一律 1/4 (0.25) に設定 (v_inv の勾配流)
        static_weights = {g: 0.25 for g in group_keys}

        print("  Using static weights for uniform target distribution (w_g = 0.25 for all):")
        for g, w in static_weights.items():
            print(f"  w_g{g} = {w:.6f}")
        
        train_loader = None
        
    elif debias_method == 'GroupDRO':
        print("\n--- Group DRO Enabled ---")
        print(f"  [Warning] 'batch_size' config is ignored. Using full-batch (per-group) gradient calculation.")
        
        # 動的重み q を一様分布で初期化
        dro_q_weights = torch.ones(len(group_keys), device=device) / len(group_keys)
        
        print(f"  Using dynamic weights 'q' initialized to: {dro_q_weights.cpu().numpy()}")
        print(f"  Group weight step size (eta_q): {config['dro_eta_q']}")
        
        train_loader = None

    elif debias_method == 'None':
        # 通常のERM学習
        print(f"\n--- ERM (Debias Method: None) Enabled ---")
        print(f"  Using batch_size: {config['batch_size']}")
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config['batch_size'], shuffle=True)
    
    else:
        raise ValueError(f"Unknown debias_method: {debias_method}. Must be 'None', 'IW_uniform', or 'GroupDRO'.")

    utils.display_group_distribution(y_train, a_train, "Train Set", config['dataset_name'], result_dir)
    utils.display_group_distribution(y_test, a_test, "Test Set", config['dataset_name'], result_dir)

    # 4. モデルとオプティマイザの準備
    print("\n--- 2. Setting up Model and Optimizer ---")
    input_dim = 3 * image_size * image_size
    model = model_module.MLP(
        input_dim=input_dim, hidden_dim=config['hidden_dim'],
        num_hidden_layers=config['num_hidden_layers'], activation_fn=config['activation_function'],
        use_skip_connections=config['use_skip_connections'],
        initialization_method=config['initialization_method']
    ).to(device)

    optimizer_params = model_module.apply_manual_parametrization(
        model, method=config['initialization_method'], base_lr=config['learning_rate'],
        hidden_dim=config['hidden_dim'], input_dim=input_dim,
        fix_final_layer=config.get('fix_final_layer', False)
    )

    optimizer = optim.Adam(optimizer_params) if config['optimizer'] == 'Adam' else optim.SGD(optimizer_params, momentum=config['momentum'])

    if config.get('wandb', {}).get('enable', False):
        wandb.watch(model, log='all', log_freq=100)

    all_target_layers = [f'layer_{i+1}' for i in range(config['num_hidden_layers'])]

    # 5. 学習・評価ループ
    print("\n--- 3. Starting Training & Evaluation Loop ---")
    history = {k: [] for k in ['train_avg_loss', 'test_avg_loss', 'train_worst_loss', 'test_worst_loss',
                               'train_avg_acc', 'test_avg_acc', 'train_worst_acc', 'test_worst_acc',
                               'train_group_losses', 'test_group_losses', 'train_group_accs', 'test_group_accs']}

    analysis_histories = {name: {} for name in [
        'grad_gram_train', 'grad_gram_test', 'jacobian_norm_train', 'jacobian_norm_test',
        'grad_gram_spectrum_train', 'grad_gram_spectrum_test',
        'grad_norm_ratio_train', 'grad_norm_ratio_test',
        'grad_basis_train', 'grad_basis_test',
        'prop1_terms_train', 'prop1_terms_test'
    ]}

    for epoch in range(config['epochs']):
        model.train()

        # --- 学習ステップの分岐 ---
        if debias_method == 'IW_uniform':
            # --- IW (Uniform Target) の学習ステップ (フルバッチ・グループ別勾配) ---
            optimizer.zero_grad()
            group_grads_list = {} # パラメータごとの勾配リストを格納

            # 1. グループごとに勾配を計算
            for g in group_keys:
                y_val, a_val = g
                mask = (y_train == y_val) & (a_train == a_val)
                X_g, y_g = X_train[mask].to(device), y_train[mask].to(device)

                if len(X_g) == 0:
                    continue

                # 勾配計算
                scores_g, _ = model(X_g)
                loss_g = get_loss_function(scores_g, y_g, loss_function_name)
                loss_g.backward()

                # 勾配をリストとして保存 (cloneしないと上書きされる)
                group_grads_list[g] = [p.grad.clone() for p in model.parameters() if p.grad is not None]

                # 次のグループのために勾配をリセット
                optimizer.zero_grad()

            # 2. 重み付き勾配を集約 (p.grad に設定)
            #    (static_weights には 0.25 が入っている)
            param_idx = 0
            for param in model.parameters():
                if param.requires_grad:
                    # このパラメータの最終的な勾配
                    debiased_grad = torch.zeros_like(param)
                    for g, w_g in static_weights.items(): # static_weights を使用
                        if g in group_grads_list:
                            # 対応するグループの勾配リストから勾配を取得
                            grad_g_param = group_grads_list[g][param_idx]
                            debiased_grad += w_g * grad_g_param.to(device)

                    param.grad = debiased_grad
                    param_idx += 1

            # 3. パラメータ更新
            optimizer.step()

        elif debias_method == 'GroupDRO':
            # --- Group DRO 学習ステップ (フルバッチ・グループ別勾配) ---
            optimizer.zero_grad()
            group_grads_list = {} # パラメータごとの勾配リスト
            group_losses_tensor = torch.zeros(len(group_keys), device=device)

            # 1. グループごとに勾配と損失を計算
            for i, g in enumerate(group_keys):
                y_val, a_val = g
                mask = (y_train == y_val) & (a_train == a_val)
                X_g, y_g = X_train[mask].to(device), y_train[mask].to(device)

                if len(X_g) == 0:
                    continue

                # 勾配計算
                scores_g, _ = model(X_g)
                loss_g = get_loss_function(scores_g, y_g, loss_function_name)
                
                group_losses_tensor[i] = loss_g.detach() # 損失を保存
                
                loss_g.backward()

                # 勾配をリストとして保存
                group_grads_list[g] = [p.grad.clone() for p in model.parameters() if p.grad is not None]
                optimizer.zero_grad() # 次のグループのために勾配をリセット

            # 2. グループ重み q を更新 (Exponentiated Gradient Ascent)
            with torch.no_grad():
                dro_eta_q = config['dro_eta_q']
                # q_t+1 = q_t * exp(eta * L_t)
                dro_q_weights = dro_q_weights * torch.exp(dro_eta_q * group_losses_tensor)
                # 正規化
                dro_q_weights = dro_q_weights / dro_q_weights.sum()
            
            # 100エポックごと，または最初のエポックで重みをログ出力
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1} GroupDRO weights q: {np.array2string(dro_q_weights.cpu().numpy(), precision=4)}")


            # 3. 重み付き勾配を集約 (p.grad に設定)
            param_idx = 0
            for param in model.parameters():
                if param.requires_grad:
                    debiased_grad = torch.zeros_like(param)
                    for i, g in enumerate(group_keys):
                        w_g = dro_q_weights[i] # 動的な重みを使用
                        if g in group_grads_list:
                            grad_g_param = group_grads_list[g][param_idx]
                            debiased_grad += w_g * grad_g_param.to(device)
                    
                    param.grad = debiased_grad
                    param_idx += 1

            # 4. パラメータ更新
            optimizer.step()

        elif debias_method == 'None':
            # --- 通常のERM学習ステップ (ミニバッチ) ---
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                scores, _ = model(X_batch)
                loss = get_loss_function(scores, y_batch, loss_function_name)
                loss.backward()
                optimizer.step()
        # --- 分岐終了 ---

        # --- 評価 ---
        train_metrics = utils.evaluate_model(model, X_train, y_train, a_train, device, loss_function_name)
        test_metrics = utils.evaluate_model(model, X_test, y_test, a_test, device, loss_function_name)
        for key_base in history.keys():
            if key_base.startswith('train_'):
                history[key_base].append(train_metrics[key_base.replace('train_', '')])
            else:
                history[key_base].append(test_metrics[key_base.replace('test_', '')])

        print(f"Epoch {epoch+1:5d}/{config['epochs']} | Train [Loss: {train_metrics['avg_loss']:.4f}, Worst: {train_metrics['worst_loss']:.4f}, Acc: {train_metrics['avg_acc']:.4f}, Worst: {train_metrics['worst_acc']:.4f}] | Test [Loss: {test_metrics['avg_loss']:.4f}, Worst: {test_metrics['worst_loss']:.4f}, Acc: {test_metrics['avg_acc']:.4f}, Worst: {test_metrics['worst_acc']:.4f}]")

        if config.get('wandb', {}).get('enable', False):
            log_metrics = {
                'epoch': epoch + 1,
                'train_avg_loss': train_metrics['avg_loss'],
                'train_worst_loss': train_metrics['worst_loss'],
                'train_avg_acc': train_metrics['avg_acc'],
                'train_worst_acc': train_metrics['worst_acc'],
                'test_avg_loss': test_metrics['avg_loss'],
                'test_worst_loss': test_metrics['worst_loss'],
                'test_avg_acc': test_metrics['avg_acc'],
                'test_worst_acc': test_metrics['worst_acc'],
            }
            if debias_method == 'GroupDRO':
                for i, g in enumerate(group_keys):
                    log_metrics[f'group_q_weight/q_g{g}'] = dro_q_weights[i].item()

            for i in range(4):
                log_metrics[f'train_group_{i}_loss'] = train_metrics['group_losses'][i]
                log_metrics[f'train_group_{i}_acc'] = train_metrics['group_accs'][i]
                log_metrics[f'test_group_{i}_loss'] = test_metrics['group_losses'][i]
                log_metrics[f'test_group_{i}_acc'] = test_metrics['group_accs'][i]
            wandb.log(log_metrics)

        # --- チェックポイント分析 ---
        current_epoch = epoch + 1

        def should_run(analysis_name, epoch_list_name):
            if not config.get(analysis_name, False):
                return False
            epoch_list = config.get(epoch_list_name)
            if epoch_list is None: # キーが存在しないか，値がNone（毎エポック実行）
                return True
            return current_epoch in epoch_list # リストが指定されている場合

        run_grad_gram = should_run('analyze_gradient_gram', 'gradient_gram_analysis_epochs')
        run_grad_spectrum = should_run('analyze_gradient_gram_spectrum', 'gradient_gram_spectrum_analysis_epochs')
        run_grad_norm_ratio = should_run('analyze_gradient_norm_ratio', 'gradient_norm_ratio_analysis_epochs')
        run_grad_basis = should_run('analyze_gradient_basis', 'gradient_basis_analysis_epochs')
        run_prop1_terms = should_run('analyze_proposition1_terms', 'proposition1_terms_analysis_epochs')

        run_any_analysis = run_grad_gram or run_grad_spectrum or run_grad_norm_ratio or run_grad_basis or run_prop1_terms

        if run_any_analysis:
            print(f"\n{'='*25} CHECKPOINT ANALYSIS @ EPOCH {current_epoch} {'='*25}")

            # train_outputs, test_outputs は analysis.py で使われなくなった
            train_outputs, test_outputs = (None, None)

            temp_config = config.copy()
            temp_config['analyze_gradient_gram'] = run_grad_gram
            temp_config['analyze_gradient_gram_spectrum'] = run_grad_spectrum
            temp_config['analyze_gradient_norm_ratio'] = run_grad_norm_ratio
            temp_config['analyze_gradient_basis'] = run_grad_basis
            temp_config['analyze_proposition1_terms'] = run_prop1_terms

            analysis.run_all_analyses(
                temp_config, current_epoch, all_target_layers, model, train_outputs, test_outputs,
                X_train, y_train, a_train, X_test, y_test, a_test, analysis_histories,
                optimizer.param_groups, history
            )

            if config.get('wandb', {}).get('enable', False):
                analysis_log_metrics = {}
                for history_key, history_dict in analysis_histories.items():
                    if current_epoch in history_dict:
                        epoch_data = history_dict[current_epoch]
                        if isinstance(epoch_data, dict):
                            for sub_key, value in epoch_data.items():
                                if isinstance(value, list):
                                    # スペクトル分析の固有ベクトルなどはリストなので個別にログ
                                    if 'eigenvector' in sub_key:
                                         for i, v in enumerate(value):
                                            analysis_log_metrics[f'analysis/{history_key}/{sub_key}_{i+1}'] = v
                                    # 命題1の項も辞書の辞書なので個別対応
                                    elif 'prop1_terms' in history_key:
                                        analysis_log_metrics[f'analysis/{history_key}/{sub_key}'] = value
                                    else:
                                        # 他のリスト（例: 固有値）
                                        for i, v in enumerate(value):
                                            analysis_log_metrics[f'analysis/{history_key}/{sub_key}_{i+1}'] = v
                                else:
                                    analysis_log_metrics[f'analysis/{history_key}/{sub_key}'] = value
                if analysis_log_metrics:
                    analysis_log_metrics['epoch'] = current_epoch
                    wandb.log(analysis_log_metrics)

            print(f"{'='*25} END OF ANALYSIS @ EPOCH {current_epoch} {'='*27}\n")

    # 6. 最終結果の保存とプロット
    print("\n--- 4. Saving Final Results and Plotting ---")
    history_df = pd.DataFrame(history)
    history_df.index.name = 'epoch'
    history_df.to_csv(os.path.join(result_dir, 'training_history.csv'))

    plotting.plot_all_results(history_df, analysis_histories, all_target_layers, result_dir, config)

    if config.get('wandb', {}).get('enable', False):
        wandb.finish()

    print(f"\nExperiment finished. All results saved in: {result_dir}")

if __name__ == '__main__':
    main(config_path='config.yaml')
