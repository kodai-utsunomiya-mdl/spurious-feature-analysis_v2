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
import feature_extractor
import trainer

def main(config_path='config.yaml'):
    # 1. 設定ファイルの読み込み
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # --- debias_method の読み込み ---
    debias_method = config.get('debias_method', 'None')
    loss_function_name = config['loss_function']
    
    # eval_batch_size を config から読み込む
    # 見つからない場合は None を設定 (utils.py 側でフルバッチとして扱われる)
    eval_batch_size = config.get('eval_batch_size', None)


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
    
    # キャッシュディレクトリの準備
    CACHE_DIR = config.get('features_cache_dir', 'features_cache') # config.yaml に 'features_cache_dir' を追加可能
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Using feature cache directory: {CACHE_DIR}")

    # --- 特徴抽出器のセットアップ (configに応じて) ---
    # feature_extractor.py に分離
    feat_extractor_model = None
    input_dim_for_mlp = None 
    
    # configから設定を読み込む
    use_feature_extractor = config.get('use_feature_extractor', False)
    
    # モデル名を取得
    model_name = config.get('feature_extractor_model_name', 'ResNet18') 

    # キャッシュパスの変数を初期化
    cache_path_train_X = None
    cache_path_train_y = None
    cache_path_train_a = None
    cache_path_test_X = None
    cache_path_test_y = None
    cache_path_test_a = None


    if use_feature_extractor:
        # セットアップ関数を呼び出し
        feat_extractor_model, input_dim_for_mlp = feature_extractor.setup_feature_extractor(config)
        
        # キャッシュパスの生成 (feature_extractor.py から呼び出し)
        base_name_train = feature_extractor.get_cache_filename(config['dataset_name'], model_name, config, 'train')
        base_name_test = feature_extractor.get_cache_filename(config['dataset_name'], model_name, config, 'test')
        
        cache_path_train_X = os.path.join(CACHE_DIR, base_name_train.replace('.pt', '_X.pt'))
        cache_path_train_y = os.path.join(CACHE_DIR, base_name_train.replace('.pt', '_y.pt'))
        cache_path_train_a = os.path.join(CACHE_DIR, base_name_train.replace('.pt', '_a.pt'))
        
        cache_path_test_X = os.path.join(CACHE_DIR, base_name_test.replace('.pt', '_X.pt'))
        cache_path_test_y = os.path.join(CACHE_DIR, base_name_test.replace('.pt', '_y.pt'))
        cache_path_test_a = os.path.join(CACHE_DIR, base_name_test.replace('.pt', '_a.pt'))

        print(f"Train feature cache path (X): {cache_path_train_X}")
        print(f"Test feature cache path (X): {cache_path_test_X}")
    
    elif not use_feature_extractor:
        if config['dataset_name'] == 'WaterBirds':
            print("Using raw WaterBirds images (3x224x224). OOM might occur.")
        else:
            print("Using raw image pixels.")

    # 特徴抽出の実行 or キャッシュのロード
    
    # キャッシュの存在確認
    use_cache = (
        use_feature_extractor and
        cache_path_train_X is not None and
        os.path.exists(cache_path_train_X) and
        os.path.exists(cache_path_train_y) and
        os.path.exists(cache_path_train_a) and
        os.path.exists(cache_path_test_X) and
        os.path.exists(cache_path_test_y) and
        os.path.exists(cache_path_test_a)
    )

    if use_cache:
        # --- キャッシュが存在する場合 ---
        try:
            print(f"Loading features and labels from cache...")
            X_train = torch.load(cache_path_train_X, map_location=torch.device('cpu')) # CPUにロード
            y_train = torch.load(cache_path_train_y, map_location=torch.device('cpu'))
            a_train = torch.load(cache_path_train_a, map_location=torch.device('cpu'))
            X_test = torch.load(cache_path_test_X, map_location=torch.device('cpu')) # CPUにロード
            y_test = torch.load(cache_path_test_y, map_location=torch.device('cpu'))
            a_test = torch.load(cache_path_test_a, map_location=torch.device('cpu'))
            
            print("Successfully loaded features and labels from cache.")
            print(f"Feature dimensions from cache: Train={X_train.shape}, Test={X_test.shape}")
            
            # feature_extractor はもう不要なのでメモリ解放
            if feat_extractor_model is not None:
                del feat_extractor_model
                feat_extractor_model = None

        except Exception as e:
            print(f"Warning: Failed to load features from cache: {e}. Re-extracting...")
            use_cache = False # ロード失敗
            if feat_extractor_model is None:
                 raise RuntimeError("Feature extractor was not set up, but cache load failed.") # 安全装置
    
    if not use_cache:
        # --- キャッシュが存在しない (or ロード失敗) の場合 ---
        
        # 生データをロード
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

        # 特徴抽出 (必要な場合)
        if use_feature_extractor:
            if feat_extractor_model is None:
                 raise RuntimeError("Feature extractor was not set up, but cache was not found.") # 安全装置
            
            print("--- Starting Feature Extraction (Train) ---")
            # feature_extractor.py から呼び出し
            X_train_features = feature_extractor.extract_features(feat_extractor_model, X_train, device)
            print("--- Starting Feature Extraction (Test) ---")
            X_test_features = feature_extractor.extract_features(feat_extractor_model, X_test, device)
            print(f"Feature dimensions after extraction: Train={X_train_features.shape}, Test={X_test_features.shape}")
            
            # X_train, X_test を特徴量で上書き
            X_train = X_train_features
            X_test = X_test_features
            
            # キャッシュに保存
            try:
                print(f"Saving features and labels to cache...")
                torch.save(X_train, cache_path_train_X)
                torch.save(y_train, cache_path_train_y)
                torch.save(a_train, cache_path_train_a)
                torch.save(X_test, cache_path_test_X)
                torch.save(y_test, cache_path_test_y)
                torch.save(a_test, cache_path_test_a)
                print("Successfully saved features and labels to cache.")
            except Exception as e_save:
                print(f"Warning: Failed to save features to cache: {e_save}")
        
        # (use_feature_extractor=False の場合は，生データのまま進む)


    if config['show_and_save_samples']:
        # --- データが画像の場合のみサンプル表示 ---
        if X_train.dim() == 4: # (B, C, H, W) 
            utils.show_dataset_samples(X_train, y_train, a_train, config['dataset_name'], result_dir)
        else:
            print("Skipping dataset sample visualization (data is pre-extracted features).")

    X_train = utils.l2_normalize_images(X_train)
    X_test = utils.l2_normalize_images(X_test)

    # --- グループのリストを定義 ---
    group_keys = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]

    # --- バイアス除去手法のための設定 ---
    static_weights = None
    dro_q_weights = None
    
    if debias_method == 'IW_uniform':
        print("\n--- Importance Weighting (Uniform Target) Enabled (Equivalent to v_inv) ---")
        print(f"  [Warning] 'train_batch_size' config is ignored. Using full-batch (per-group) gradient calculation.")
        print("  Removing both marginal bias (Term II) and spurious correlation (Term III).")

        # 理論に基づき，重みを一律 1/4 (0.25) に設定 (v_inv の勾配流)
        static_weights = {g: 0.25 for g in group_keys}

        print("  Using static weights for uniform target distribution (w_g = 0.25 for all):")
        for g, w in static_weights.items():
            print(f"  w_g{g} = {w:.6f}")
        
        train_loader = None
        
    elif debias_method == 'GroupDRO':
        print("\n--- Group DRO Enabled ---")
        print(f"  [Warning] 'train_batch_size' config is ignored. Using full-batch (per-group) gradient calculation.")
        
        # 動的重み q を一様分布で初期化
        dro_q_weights = torch.ones(len(group_keys), device=device) / len(group_keys)
        
        print(f"  Using dynamic weights 'q' initialized to: {dro_q_weights.cpu().numpy()}")
        print(f"  Group weight step size (eta_q): {config['dro_eta_q']}")
        
        train_loader = None

    elif debias_method == 'None':
        # 通常のERM学習
        print(f"\n--- ERM (Debias Method: None) Enabled ---")
        train_batch_size_erm = config.get('train_batch_size', 50000) 
        print(f"  Using train_batch_size: {train_batch_size_erm}")
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=train_batch_size_erm, shuffle=True)
    
    else:
        raise ValueError(f"Unknown debias_method: {debias_method}. Must be 'None', 'IW_uniform', or 'GroupDRO'.")

    utils.display_group_distribution(y_train, a_train, "Train Set", config['dataset_name'], result_dir)
    utils.display_group_distribution(y_test, a_test, "Test Set", config['dataset_name'], result_dir)

    # 4. モデルとオプティマイザの準備
    print("\n--- 2. Setting up Model and Optimizer ---")
    
    # --- input_dim の計算 (config に応じて) ---
    if use_feature_extractor:
        # input_dim_for_mlp の扱い
        # input_dim_for_mlp が設定されていない (キャッシュロードなどで) 場合，
        # ロードした X_train の次元から復元する
        if input_dim_for_mlp is None:
             if X_train.dim() == 2: # (B, D)
                 input_dim_for_mlp = X_train.shape[1]
                 print(f"Using FEATURE input_dim (inferred from cached data): {input_dim_for_mlp}")
             else:
                 raise ValueError(f"Cached feature data has unexpected dimensions: {X_train.shape}")
        
        if input_dim_for_mlp is None:
             raise ValueError("input_dim_for_mlp was not set correctly during feature extractor setup.")
        input_dim = input_dim_for_mlp
        print(f"Using FEATURE input_dim: {input_dim}")
        
    else:
        # 特徴抽出を使わない場合
        
        # X_train の形状から input_dim を計算
        # (B, C, H, W) -> C*H*W または (B, D) -> D
        if X_train.dim() == 4: # 画像データ
            input_dim = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
        elif X_train.dim() == 2: # すでに特徴量
             input_dim = X_train.shape[1]
        else:
            raise ValueError(f"Unexpected X_train dimensions: {X_train.shape}")
            
        print(f"Using RAW input_dim ({config['dataset_name']}): {input_dim}")
        
    model = model_module.MLP(
        input_dim=input_dim, hidden_dim=config['hidden_dim'],
        num_hidden_layers=config['num_hidden_layers'], activation_fn=config['activation_function'],
        use_skip_connections=config['use_skip_connections'],
        initialization_method=config['initialization_method']
    ).to(device)

    # apply_manual_parametrization はモデルの重みを直接初期化する
    model_module.apply_manual_parametrization(
        model, method=config['initialization_method'],
        hidden_dim=config['hidden_dim'],
        fix_final_layer=config.get('fix_final_layer', False)
    )

    # オプティマイザに渡すパラメータを設定
    # (fix_final_layer=True の場合, model.parameters() は
    #  requires_grad=True のパラメータのみを返すため自動的に処理される)
    optimizer_params_list = model.parameters()

    # オプティマイザの作成
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(optimizer_params_list, lr=config['learning_rate'])
    else:
        optimizer = optim.SGD(optimizer_params_list, lr=config['learning_rate'], momentum=config['momentum'])


    if config.get('wandb', {}).get('enable', False):
        wandb.watch(model, log='all', log_freq=100)

    all_target_layers = [f'layer_{i+1}' for i in range(config['num_hidden_layers'])]

    # 5. 学習・評価ループ
    print("\n--- 3. Starting Training & Evaluation Loop ---")
    print(f"Using eval_batch_size: {eval_batch_size if eval_batch_size is not None else 'Full Batch'}")
    
    history = {k: [] for k in ['train_avg_loss', 'test_avg_loss', 'train_worst_loss', 'test_worst_loss',
                               'train_avg_acc', 'test_avg_acc', 'train_worst_acc', 'test_worst_acc',
                               'train_group_losses', 'test_group_losses', 'train_group_accs', 'test_group_accs']}

    analysis_histories = {name: {} for name in [
        'jacobian_norm_train', 'jacobian_norm_test',
        'grad_basis_train', 'grad_basis_test',
        'gap_factors_train', 'gap_factors_test',
        'static_dynamic_decomp_train', 'static_dynamic_decomp_test'
    ]}
    
    for epoch in range(config['epochs']):
        
        # --- 学習ステップ ---
        updated_dro_weights = trainer.train_epoch(
            config=config, 
            model=model, 
            optimizer=optimizer, 
            debias_method=debias_method, 
            X_train=X_train, 
            y_train=y_train, 
            a_train=a_train, 
            train_loader=train_loader, 
            group_keys=group_keys, 
            static_weights=static_weights, 
            dro_q_weights=dro_q_weights, 
            device=device, 
            loss_function_name=loss_function_name, 
            epoch=epoch
        )
        
        if updated_dro_weights is not None:
            dro_q_weights = updated_dro_weights # GroupDROの場合，重みを更新

        # --- 評価 ---
        train_metrics = utils.evaluate_model(model, X_train, y_train, a_train, device, loss_function_name, eval_batch_size)
        test_metrics = utils.evaluate_model(model, X_test, y_test, a_test, device, loss_function_name, eval_batch_size)
        
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
            
            # y=-1 の損失差 (少数派 - 多数派)
            if not np.isnan(train_metrics['group_losses'][1]) and not np.isnan(train_metrics['group_losses'][0]):
                log_metrics['train_loss_gap_y_neg1'] = train_metrics['group_losses'][1] - train_metrics['group_losses'][0]
            if not np.isnan(test_metrics['group_losses'][1]) and not np.isnan(test_metrics['group_losses'][0]):
                log_metrics['test_loss_gap_y_neg1'] = test_metrics['group_losses'][1] - test_metrics['group_losses'][0]
            
            # y=+1 の損失差 (少数派 - 多数派)
            if not np.isnan(train_metrics['group_losses'][2]) and not np.isnan(train_metrics['group_losses'][3]):
                log_metrics['train_loss_gap_y_pos1'] = train_metrics['group_losses'][2] - train_metrics['group_losses'][3]
            if not np.isnan(test_metrics['group_losses'][2]) and not np.isnan(test_metrics['group_losses'][3]):
                log_metrics['test_loss_gap_y_pos1'] = test_metrics['group_losses'][2] - test_metrics['group_losses'][3]

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

        run_grad_basis = should_run('analyze_gradient_basis', 'gradient_basis_analysis_epochs')
        run_gap_factors = should_run('analyze_gap_dynamics_factors', 'gap_dynamics_factors_analysis_epochs')
        run_jacobian_norm = should_run('analyze_jacobian_norm', 'jacobian_norm_analysis_epochs')
        run_static_dynamic = should_run('analyze_static_dynamic_decomposition', 'static_dynamic_decomposition_analysis_epochs')

        run_any_analysis = run_grad_basis or run_gap_factors or run_jacobian_norm or run_static_dynamic

        if run_any_analysis:
            print(f"\n{'='*25} CHECKPOINT ANALYSIS @ EPOCH {current_epoch} {'='*25}")

            # train_outputs, test_outputs は analysis.py で使われなくなった
            train_outputs, test_outputs = (None, None)

            temp_config = config.copy()
            temp_config['analyze_gradient_basis'] = run_grad_basis
            temp_config['analyze_gap_dynamics_factors'] = run_gap_factors
            temp_config['analyze_jacobian_norm'] = run_jacobian_norm
            temp_config['analyze_static_dynamic_decomposition'] = run_static_dynamic

            analysis.run_all_analyses(
                temp_config, current_epoch, all_target_layers, model, train_outputs, test_outputs,
                X_train, y_train, a_train, X_test, y_test, a_test, analysis_histories,
                history
            )

            if config.get('wandb', {}).get('enable', False):
                analysis_log_metrics = {}
                for history_key, history_dict in analysis_histories.items():
                    if current_epoch in history_dict:
                        epoch_data = history_dict[current_epoch]
                        if isinstance(epoch_data, dict):
                            for sub_key, value in epoch_data.items():
                                # jacobian, basis, gap_factors, static_dynamic_decomp はすべて
                                # スカラ値の辞書を返すため，単純にログ記録
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
