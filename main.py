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

def main(config_path='config.yaml'):
    # 1. 設定ファイルの読み込み
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

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
        X_train, y_train, a_train = data_loader.get_colored_mnist(
            num_samples=config['num_train_samples'], correlation=config['train_correlation'], train=True
        )
        X_test, y_test, a_test = data_loader.get_colored_mnist(
            num_samples=config['num_test_samples'], correlation=config['test_correlation'], train=False
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
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config['batch_size'], shuffle=True)

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
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            scores, _ = model(X_batch)
            loss = F.softplus(-y_batch * scores).mean() if config['loss_function'] == 'logistic' else F.mse_loss(scores, y_batch)
            loss.backward()
            optimizer.step()

        train_metrics = utils.evaluate_model(model, X_train, y_train, a_train, device, config['loss_function'])
        test_metrics = utils.evaluate_model(model, X_test, y_test, a_test, device, config['loss_function'])
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
            if epoch_list is None: # キーが存在しないか、値がNone（毎エポック実行）
                return True
            return current_epoch in epoch_list # リストが指定されている場合

        run_grad_gram = should_run('analyze_gradient_gram', 'gradient_gram_analysis_epochs')
        run_grad_spectrum = should_run('analyze_gradient_gram_spectrum', 'gradient_gram_spectrum_analysis_epochs')
        run_grad_norm_ratio = should_run('analyze_gradient_norm_ratio', 'gradient_norm_ratio_analysis_epochs')
        run_grad_basis = should_run('analyze_gradient_basis', 'gradient_basis_analysis_epochs')
        run_prop1_terms = should_run('analyze_proposition1_terms', 'proposition1_terms_analysis_epochs')
        
        # run_general_analysis を除外
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
