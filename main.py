# sp_scr_v2/main.py

import os
import yaml
import shutil
import datetime
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import wandb
from torchvision import transforms

import data_loader
import utils
import model as model_module
import analysis
import plotting
import feature_extractor
import trainer
import dfr

def main(config_path='config.yaml'):
    # 1. 設定ファイルの読み込み
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    debias_method = config.get('debias_method', 'None')
    loss_function_name = config['loss_function']
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

    # 2. 結果を保存するディレクトリの作成
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_dir = os.path.join('results', f"{config['experiment_name']}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(result_dir, 'config_used.yaml'))
    print(f"Results will be saved to: {result_dir}")
    config['result_dir'] = result_dir

    device = config['device'] if torch.cuda.is_available() and config['device'] == 'cuda' else "cpu"
    print(f"Using device: {device}")

    # 3. データセットの準備
    print("\n--- 1. Preparing Dataset ---")
    
    # キャッシュディレクトリの準備
    CACHE_DIR = config.get('features_cache_dir', 'features_cache') # config.yaml に 'features_cache_dir' を追加可能
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Using feature cache directory: {CACHE_DIR}")

    # 特徴抽出器のセットアップ
    feat_extractor_model = None
    input_dim_for_mlp = None 
    use_feature_extractor = config.get('use_feature_extractor', False)
    model_name = config.get('feature_extractor_model_name', 'ResNet18') 

    # キャッシュパスの変数を初期化
    cache_path_train_X = None
    cache_path_train_y = None
    cache_path_train_a = None
    cache_path_val_X = None
    cache_path_val_y = None
    cache_path_val_a = None
    cache_path_test_X = None
    cache_path_test_y = None
    cache_path_test_a = None


    if use_feature_extractor:
        feat_extractor_model, input_dim_for_mlp = feature_extractor.setup_feature_extractor(config)

        # キャッシュパスの生成
        base_name_train = feature_extractor.get_cache_filename(config['dataset_name'], model_name, config, 'train')
        base_name_test = feature_extractor.get_cache_filename(config['dataset_name'], model_name, config, 'test')
        
        # Validation用のキャッシュ名
        dfr_n = config.get('dfr_val_samples_per_group', 100)
        base_name_val = base_name_train.replace('.pt', f'_dfr_val_n{dfr_n}.pt')
        
        cache_path_train_X = os.path.join(CACHE_DIR, base_name_train.replace('.pt', '_X.pt'))
        cache_path_train_y = os.path.join(CACHE_DIR, base_name_train.replace('.pt', '_y.pt'))
        cache_path_train_a = os.path.join(CACHE_DIR, base_name_train.replace('.pt', '_a.pt'))

        cache_path_val_X = os.path.join(CACHE_DIR, base_name_val.replace('.pt', '_X.pt'))
        cache_path_val_y = os.path.join(CACHE_DIR, base_name_val.replace('.pt', '_y.pt'))
        cache_path_val_a = os.path.join(CACHE_DIR, base_name_val.replace('.pt', '_a.pt'))

        cache_path_test_X = os.path.join(CACHE_DIR, base_name_test.replace('.pt', '_X.pt'))
        cache_path_test_y = os.path.join(CACHE_DIR, base_name_test.replace('.pt', '_y.pt'))
        cache_path_test_a = os.path.join(CACHE_DIR, base_name_test.replace('.pt', '_a.pt'))

        print(f"Train feature cache path (X): {cache_path_train_X}")
        print(f"Val feature cache path (X)  : {cache_path_val_X}")
        print(f"Test feature cache path (X) : {cache_path_test_X}")
    
    elif not use_feature_extractor:
        if config['dataset_name'] == 'WaterBirds':
            print("Using raw WaterBirds images (3x224x224). OOM might occur.")
        else:
            print("Using raw image pixels.")

    # 特徴抽出の実行 or キャッシュのロード
    
    # キャッシュの存在確認 (Train, Testは必須)
    use_cache = (
        use_feature_extractor and
        cache_path_train_X is not None and
        os.path.exists(cache_path_train_X) and
        os.path.exists(cache_path_test_X) 
    )
    # DFRを使用する場合，Valキャッシュの存在も確認
    if config.get('use_dfr', False) and use_cache:
        if not (cache_path_val_X is not None and os.path.exists(cache_path_val_X)):
             print("Validation cache not found (or config changed). Will re-extract/generate.")
             use_cache = False

    X_train, y_train, a_train = None, None, None
    X_val_dfr, y_val_dfr, a_val_dfr = None, None, None
    X_test, y_test, a_test = None, None, None

    if use_cache:
        # --- キャッシュが存在する場合 ---
        try:
            print(f"Loading features and labels from cache...")
            X_train = torch.load(cache_path_train_X, map_location=torch.device('cpu'))
            y_train = torch.load(cache_path_train_y, map_location=torch.device('cpu'))
            a_train = torch.load(cache_path_train_a, map_location=torch.device('cpu'))
            
            if config.get('use_dfr', False):
                X_val_dfr = torch.load(cache_path_val_X, map_location=torch.device('cpu'))
                y_val_dfr = torch.load(cache_path_val_y, map_location=torch.device('cpu'))
                a_val_dfr = torch.load(cache_path_val_a, map_location=torch.device('cpu'))

            X_test = torch.load(cache_path_test_X, map_location=torch.device('cpu'))
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
            use_cache = False
            if feat_extractor_model is None:
                 raise RuntimeError("Feature extractor was not set up, but cache load failed.")
    
    if not use_cache:
        # --- キャッシュが存在しない (or ロード失敗) の場合 ---
        
        # 生データをロード (Train, Val, Testの3つを受け取る)
        # 合成データセットは内部でRemainderからValを生成済み
        # Waterbirdsは公式Valを返す
        if config['dataset_name'] == 'ColoredMNIST':
            image_size = 28
            X_train, y_train, a_train, X_val_dfr, y_val_dfr, a_val_dfr, X_test, y_test, a_test = data_loader.get_colored_mnist_all(config)

        elif config['dataset_name'] == 'Dominoes':
            image_size = 224
            # Dominoesは内部で224x224にリサイズ
            X_train, y_train, a_train, X_val_dfr, y_val_dfr, a_val_dfr, X_test, y_test, a_test = data_loader.get_dominoes_all(config)

        elif config['dataset_name'] == 'WaterBirds':
            image_size = 224
            # Waterbirdsは公式Validationセットを読み込む
            X_train, y_train, a_train, X_val_raw, y_val_raw, a_val_raw, X_test, y_test, a_test = data_loader.get_waterbirds_dataset(
                num_train=config['num_train_samples'],
                num_test=config['num_test_samples'],
                image_size=image_size
            )
            
            # WaterBirdsの場合は，ここで公式Validationセットから均衡化されたDFR用データを作成
            if config.get('use_dfr', False):
                print("Balancing Waterbirds Official Validation Set for DFR...")
                target_per_group = config.get('dfr_val_samples_per_group', 100)
                X_val_dfr, y_val_dfr, a_val_dfr = data_loader.balance_dataset_by_group_size(
                    X_val_raw, y_val_raw, a_val_raw, target_per_group
                )
            else:
                X_val_dfr, y_val_dfr, a_val_dfr = None, None, None
            
        else:
            raise ValueError(f"Unknown dataset: {config['dataset_name']}")

        # Grayscale Conversion
        # Feature Extractorを使わない場合 (=Raw Pixel MLP) の次元削減用
        if config.get('use_grayscale', False):
            print("Converting images to Grayscale (1 channel)...")
            grayscale_transform = transforms.Grayscale(num_output_channels=1)
            
            if X_train is not None and X_train.dim() == 4:
                X_train = grayscale_transform(X_train)
            if X_val_dfr is not None and X_val_dfr.dim() == 4:
                X_val_dfr = grayscale_transform(X_val_dfr)
            if X_test is not None and X_test.dim() == 4:
                X_test = grayscale_transform(X_test)
                
            print(f"Data shape after grayscale conversion: {X_train.shape}")


        # 特徴抽出 (必要な場合)
        if use_feature_extractor:
            if feat_extractor_model is None:
                 raise RuntimeError("Feature extractor was not set up, but cache was not found.")
            
            print("--- Starting Feature Extraction (Train) ---")
            X_train_features = feature_extractor.extract_features(feat_extractor_model, X_train, device)
            X_train = X_train_features
            
            if X_val_dfr is not None:
                print("--- Starting Feature Extraction (DFR Validation) ---")
                X_val_dfr = feature_extractor.extract_features(feat_extractor_model, X_val_dfr, device)

            print("--- Starting Feature Extraction (Test) ---")
            X_test_features = feature_extractor.extract_features(feat_extractor_model, X_test, device)
            X_test = X_test_features
            
            print(f"Feature dimensions after extraction: Train={X_train.shape}, Test={X_test.shape}")
            
            # キャッシュに保存
            try:
                print(f"Saving features and labels to cache...")
                torch.save(X_train, cache_path_train_X)
                torch.save(y_train, cache_path_train_y)
                torch.save(a_train, cache_path_train_a)
                
                if X_val_dfr is not None:
                    torch.save(X_val_dfr, cache_path_val_X)
                    torch.save(y_val_dfr, cache_path_val_y)
                    torch.save(a_val_dfr, cache_path_val_a)

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
    
    # DFR用の分割データも正規化
    if X_val_dfr is not None:
        X_val_dfr = utils.l2_normalize_images(X_val_dfr)
        
    X_test = utils.l2_normalize_images(X_test)

    # グループのリストを定義
    group_keys = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]

    # デバイアス手法のための設定
    static_weights = None
    dro_q_weights = None
    
    if debias_method == 'IW_uniform':
        print("\n--- Importance Weighting (Uniform Target) Enabled (Equivalent to v_inv) ---")
        print(f"  [Warning] 'train_batch_size' config is ignored. Using full-batch (per-group) gradient calculation.")
        print("  Removing both marginal bias (Term II) and spurious correlation (Term III).")

        # 重みを 1/4 (0.25) に設定 (v_inv の勾配流)
        static_weights = {g: 0.25 for g in group_keys}

        print("  Using static weights for uniform target distribution (w_g = 0.25 for all):")
        for g, w in static_weights.items():
            print(f"  w_g{g} = {w:.6f}")
        
        train_loader = None
        
    elif debias_method == 'GroupDRO':
        print("\n--- Group DRO Enabled ---")
        print(f"  [Warning] 'train_batch_size' config is ignored. Using full-batch (per-group) gradient calculation.")
        
        # 動的な重み q を一様分布で初期化
        dro_q_weights = torch.ones(len(group_keys), device=device) / len(group_keys)
        
        print(f"  Using dynamic weights 'q' initialized to: {dro_q_weights.cpu().numpy()}")
        print(f"  Group weight step size (eta_q): {config['dro_eta_q']}")
        
        train_loader = None

    elif debias_method == 'None':
        # 通常のERM
        print(f"\n--- ERM (Debias Method: None) Enabled ---")
        train_batch_size_erm = config.get('train_batch_size', 50000) 
        print(f"  Using train_batch_size: {train_batch_size_erm}")
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=train_batch_size_erm, shuffle=True)
    
    else:
        raise ValueError(f"Unknown debias_method: {debias_method}. Must be 'None', 'IW_uniform', or 'GroupDRO'.")

    utils.display_group_distribution(y_train, a_train, "Train Set", config['dataset_name'], result_dir)
    if y_val_dfr is not None:
        utils.display_group_distribution(y_val_dfr, a_val_dfr, "DFR Validation Set (Balanced)", config['dataset_name'], result_dir)
    utils.display_group_distribution(y_test, a_test, "Test Set", config['dataset_name'], result_dir)

    # 4. モデルとオプティマイザの準備
    print("\n--- 2. Setting up Model and Optimizer ---")
    
    # --- input_dim の計算 (config に応じて) ---
    if use_feature_extractor:
        # input_dim_for_mlp の扱い
        # input_dim_for_mlp が設定されていない (キャッシュロードなどで) 場合，
        # ロードした X_train の次元から復元
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

    use_bias = config.get('use_bias', False)
    use_zero_bias_init = config.get('use_zero_bias_initialization', False)
    
    print(f"Using bias in MLP: {use_bias}")
    print(f"Using zero bias initialization: {use_zero_bias_init}")

    model = model_module.MLP(
        input_dim=input_dim, hidden_dim=config['hidden_dim'],
        num_hidden_layers=config['num_hidden_layers'], activation_fn=config['activation_function'],
        use_skip_connections=config['use_skip_connections'],
        initialization_method=config['initialization_method'],
        use_bias=use_bias,
        use_zero_bias_init=use_zero_bias_init
    ).to(device)

    if config.get('fix_final_layer', False):
        print("Freezing final layer weights (fix_final_layer=True)...")
        model.classifier.weight.requires_grad = False
        if model.classifier.bias is not None:
            model.classifier.bias.requires_grad = False

    # オプティマイザに渡すパラメータを設定
    optimizer_params_list = model.get_optimizer_parameters(
        optimizer_name=config['optimizer'],
        global_lr=config['learning_rate']
    )

    # オプティマイザの作成
    if config['optimizer'] == 'Adam':
        optimizer = optim.AdamW(optimizer_params_list, lr=config['learning_rate'])
    else:
        # SGD
        optimizer = optim.SGD(optimizer_params_list, lr=config['learning_rate'], momentum=config['momentum'])


    if config.get('wandb', {}).get('enable', False):
        wandb.watch(model, log='parameters', log_freq=100)

    all_target_layers = [f'layer_{i+1}' for i in range(config['num_hidden_layers'])]

    history = {k: [] for k in ['train_avg_loss', 'test_avg_loss', 'train_worst_loss', 'test_worst_loss',
                               'train_avg_acc', 'test_avg_acc', 'train_worst_acc', 'test_worst_acc',
                               'train_group_losses', 'test_group_losses', 'train_group_accs', 'test_group_accs']}

    analysis_histories = {name: {} for name in [
        'jacobian_norm_train', 'jacobian_norm_test',
        'grad_basis_train', 'grad_basis_test',
        'gap_factors_train', 'gap_factors_test',
        'static_dynamic_decomp_train', 'static_dynamic_decomp_test',
        'model_output_exp_train', 'model_output_exp_test'
    ]}

    # 初期化直後 (Epoch 0) の分析
    if any(0 in config.get(key, []) for key in [
        'umap_analysis_epochs', 
        'gradient_basis_analysis_epochs',
        'gap_dynamics_factors_analysis_epochs',
        'jacobian_norm_analysis_epochs',
        'static_dynamic_decomposition_analysis_epochs',
        'model_output_expectation_analysis_epochs'
    ]):
        print(f"\n{'='*25} INITIAL ANALYSIS (EPOCH 0) {'='*25}")
        
        # Epoch 0 用の実行フラグ判定
        current_epoch = 0
        def should_run_init(analysis_name, epoch_list_name):
            if not config.get(analysis_name, False): return False
            epoch_list = config.get(epoch_list_name)
            return epoch_list is not None and 0 in epoch_list

        run_grad_basis = should_run_init('analyze_gradient_basis', 'gradient_basis_analysis_epochs')
        run_gap_factors = should_run_init('analyze_gap_dynamics_factors', 'gap_dynamics_factors_analysis_epochs')
        run_jacobian_norm = should_run_init('analyze_jacobian_norm', 'jacobian_norm_analysis_epochs')
        run_static_dynamic = should_run_init('analyze_static_dynamic_decomposition', 'static_dynamic_decomposition_analysis_epochs')
        run_output_exp = should_run_init('analyze_model_output_expectation', 'model_output_expectation_analysis_epochs')
        run_umap = should_run_init('analyze_umap_representation', 'umap_analysis_epochs')
        run_svd = should_run_init('analyze_singular_values', 'umap_analysis_epochs')

        # 実行用のConfigを作成
        temp_config = config.copy()
        temp_config['analyze_gradient_basis'] = run_grad_basis
        temp_config['analyze_gap_dynamics_factors'] = run_gap_factors
        temp_config['analyze_jacobian_norm'] = run_jacobian_norm
        temp_config['analyze_static_dynamic_decomposition'] = run_static_dynamic
        temp_config['analyze_model_output_expectation'] = run_output_exp
        temp_config['analyze_umap_representation'] = run_umap
        temp_config['analyze_singular_values'] = run_svd

        # 分析の実行
        analysis.run_all_analyses(
            temp_config, current_epoch, all_target_layers, model, None, None,
            X_train, y_train, a_train, X_test, y_test, a_test, analysis_histories,
            history
        )
        print(f"{'='*25} END OF INITIAL ANALYSIS {'='*27}\n")


    # 5. 学習・評価ループ
    print("\n--- 3. Starting Training & Evaluation Loop ---")
    print(f"Using eval_batch_size: {eval_batch_size if eval_batch_size is not None else 'Full Batch'}")
    
    for epoch in range(config['epochs']):
        
        # 学習ステップ
        updated_dro_weights = trainer.train_epoch(
            config, model, optimizer, debias_method, 
            X_train, y_train, a_train, train_loader, 
            group_keys, static_weights, dro_q_weights, 
            device, loss_function_name, epoch
        )
        
        if updated_dro_weights is not None:
            dro_q_weights = updated_dro_weights 

        # 評価
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
            
            if not np.isnan(train_metrics['group_losses'][1]) and not np.isnan(train_metrics['group_losses'][0]):
                log_metrics['train_loss_gap_y_neg1'] = train_metrics['group_losses'][1] - train_metrics['group_losses'][0]
            if not np.isnan(test_metrics['group_losses'][1]) and not np.isnan(test_metrics['group_losses'][0]):
                log_metrics['test_loss_gap_y_neg1'] = test_metrics['group_losses'][1] - test_metrics['group_losses'][0]
            
            if not np.isnan(train_metrics['group_losses'][2]) and not np.isnan(train_metrics['group_losses'][3]):
                log_metrics['train_loss_gap_y_pos1'] = train_metrics['group_losses'][2] - train_metrics['group_losses'][3]
            if not np.isnan(test_metrics['group_losses'][2]) and not np.isnan(test_metrics['group_losses'][3]):
                log_metrics['test_loss_gap_y_pos1'] = test_metrics['group_losses'][2] - test_metrics['group_losses'][3]

            wandb.log(log_metrics)

        # チェックポイント分析
        current_epoch = epoch + 1

        def should_run(analysis_name, epoch_list_name):
            if not config.get(analysis_name, False):
                return False
            epoch_list = config.get(epoch_list_name)
            if epoch_list is None: 
                return True
            return current_epoch in epoch_list 

        run_grad_basis = should_run('analyze_gradient_basis', 'gradient_basis_analysis_epochs')
        run_gap_factors = should_run('analyze_gap_dynamics_factors', 'gap_dynamics_factors_analysis_epochs')
        run_jacobian_norm = should_run('analyze_jacobian_norm', 'jacobian_norm_analysis_epochs')
        run_static_dynamic = should_run('analyze_static_dynamic_decomposition', 'static_dynamic_decomposition_analysis_epochs')
        run_output_exp = should_run('analyze_model_output_expectation', 'model_output_expectation_analysis_epochs')
        run_umap = should_run('analyze_umap_representation', 'umap_analysis_epochs')
        run_svd = should_run('analyze_singular_values', 'umap_analysis_epochs')

        run_any_analysis = run_grad_basis or run_gap_factors or run_jacobian_norm or run_static_dynamic or run_output_exp or run_umap or run_svd

        if run_any_analysis:
            print(f"\n{'='*25} CHECKPOINT ANALYSIS @ EPOCH {current_epoch} {'='*25}")

            train_outputs, test_outputs = (None, None)

            temp_config = config.copy()
            temp_config['analyze_gradient_basis'] = run_grad_basis
            temp_config['analyze_gap_dynamics_factors'] = run_gap_factors
            temp_config['analyze_jacobian_norm'] = run_jacobian_norm
            temp_config['analyze_static_dynamic_decomposition'] = run_static_dynamic
            temp_config['analyze_model_output_expectation'] = run_output_exp
            temp_config['analyze_umap_representation'] = run_umap
            temp_config['analyze_singular_values'] = run_svd

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

    # 7. DFR (Deep Feature Reweighting) の実行
    if config.get('use_dfr', False):
        try:
            if X_val_dfr is not None:
                dfr_train_metrics, dfr_test_metrics, baseline_results, dfr_spur_train_metrics, dfr_spur_test_metrics, \
                sing_vals_y, sing_vals_a, sing_vals_y_test, sing_vals_a_test, \
                align_val, align_test = dfr.run_dfr_procedure(
                    config, model, 
                    X_train, y_train, a_train, 
                    X_test, y_test, a_test, 
                    device,
                    loss_function_name=loss_function_name,
                    X_val=X_val_dfr, y_val=y_val_dfr, a_val=a_val_dfr
                )
            
                # WandBログの記録 (ERMの指標と対応するものをプレフィックス付きで記録)
                if config.get('wandb', {}).get('enable', False):
                    dfr_log_metrics = {}
                    
                    # --- DFR Metrics (Main Task: Target Y) ---
                    dfr_log_metrics['dfr_train_avg_loss'] = dfr_train_metrics['avg_loss']
                    dfr_log_metrics['dfr_train_worst_loss'] = dfr_train_metrics['worst_loss']
                    dfr_log_metrics['dfr_train_avg_acc'] = dfr_train_metrics['avg_acc']
                    dfr_log_metrics['dfr_train_worst_acc'] = dfr_train_metrics['worst_acc']
                    
                    dfr_log_metrics['dfr_test_avg_loss'] = dfr_test_metrics['avg_loss']
                    dfr_log_metrics['dfr_test_worst_loss'] = dfr_test_metrics['worst_loss']
                    dfr_log_metrics['dfr_test_avg_acc'] = dfr_test_metrics['avg_acc']
                    dfr_log_metrics['dfr_test_worst_acc'] = dfr_test_metrics['worst_acc']

                    for i in range(4):
                        dfr_log_metrics[f'dfr_train_group_{i}_loss'] = dfr_train_metrics['group_losses'][i]
                        dfr_log_metrics[f'dfr_train_group_{i}_acc'] = dfr_train_metrics['group_accs'][i]
                        dfr_log_metrics[f'dfr_test_group_{i}_loss'] = dfr_test_metrics['group_losses'][i]
                        dfr_log_metrics[f'dfr_test_group_{i}_acc'] = dfr_test_metrics['group_accs'][i]

                    # --- DFR Metrics (Spurious Task: Target A) ---
                    dfr_log_metrics['dfr_spur_train_avg_loss'] = dfr_spur_train_metrics['avg_loss']
                    dfr_log_metrics['dfr_spur_train_worst_loss'] = dfr_spur_train_metrics['worst_loss']
                    dfr_log_metrics['dfr_spur_train_avg_acc'] = dfr_spur_train_metrics['avg_acc']
                    dfr_log_metrics['dfr_spur_train_worst_acc'] = dfr_spur_train_metrics['worst_acc']
                    
                    dfr_log_metrics['dfr_spur_test_avg_loss'] = dfr_spur_test_metrics['avg_loss']
                    dfr_log_metrics['dfr_spur_test_worst_loss'] = dfr_spur_test_metrics['worst_loss']
                    dfr_log_metrics['dfr_spur_test_avg_acc'] = dfr_spur_test_metrics['avg_acc']
                    dfr_log_metrics['dfr_spur_test_worst_acc'] = dfr_spur_test_metrics['worst_acc']
                    
                    # Gap Metrics (Main Task)
                    if not np.isnan(dfr_train_metrics['group_losses'][1]) and not np.isnan(dfr_train_metrics['group_losses'][0]):
                        dfr_log_metrics['dfr_train_loss_gap_y_neg1'] = dfr_train_metrics['group_losses'][1] - dfr_train_metrics['group_losses'][0]
                    if not np.isnan(dfr_train_metrics['group_losses'][2]) and not np.isnan(dfr_train_metrics['group_losses'][3]):
                        dfr_log_metrics['dfr_train_loss_gap_y_pos1'] = dfr_train_metrics['group_losses'][2] - dfr_train_metrics['group_losses'][3]
                    
                    if not np.isnan(dfr_test_metrics['group_losses'][1]) and not np.isnan(dfr_test_metrics['group_losses'][0]):
                        dfr_log_metrics['dfr_test_loss_gap_y_neg1'] = dfr_test_metrics['group_losses'][1] - dfr_test_metrics['group_losses'][0]
                    if not np.isnan(dfr_test_metrics['group_losses'][2]) and not np.isnan(dfr_test_metrics['group_losses'][3]):
                        dfr_log_metrics['dfr_test_loss_gap_y_pos1'] = dfr_test_metrics['group_losses'][2] - dfr_test_metrics['group_losses'][3]

                    # モデル出力の統計量
                    for k, v in dfr_train_metrics.items():
                        if 'E[f(x)]' in k or 'Std[f(x)]' in k: dfr_log_metrics[f'dfr_train_{k}'] = v
                    for k, v in dfr_test_metrics.items():
                        if 'E[f(x)]' in k or 'Std[f(x)]' in k: dfr_log_metrics[f'dfr_test_{k}'] = v

                    # --- Analysis Metrics (Principal Angles) - Validation ---
                    if len(sing_vals_y) > 0:
                        dfr_log_metrics['dfr_val_principal_angles_Y_max'] = np.max(sing_vals_y)
                        dfr_log_metrics['dfr_val_principal_angles_Y_mean'] = np.mean(sing_vals_y)
                    
                    if len(sing_vals_a) > 0:
                        dfr_log_metrics['dfr_val_principal_angles_A_max'] = np.max(sing_vals_a)
                        dfr_log_metrics['dfr_val_principal_angles_A_mean'] = np.mean(sing_vals_a)
                        
                    dfr_log_metrics['dfr_val_feature_alignment_Y_vs_A'] = align_val

                    # --- Analysis Metrics (Principal Angles) - Test ---
                    if len(sing_vals_y_test) > 0:
                        dfr_log_metrics['dfr_test_principal_angles_Y_max'] = np.max(sing_vals_y_test)
                        dfr_log_metrics['dfr_test_principal_angles_Y_mean'] = np.mean(sing_vals_y_test)
                    
                    if len(sing_vals_a_test) > 0:
                        dfr_log_metrics['dfr_test_principal_angles_A_max'] = np.max(sing_vals_a_test)
                        dfr_log_metrics['dfr_test_principal_angles_A_mean'] = np.mean(sing_vals_a_test)
                        
                    dfr_log_metrics['dfr_test_feature_alignment_Y_vs_A'] = align_test

                    # --- Baseline Metrics (Baseline regressions) ---
                    for base_name, base_metrics in baseline_results.items():
                        # prefix: baseline/erm, baseline/reg_none, baseline/reg_l1, baseline/reg_l2
                        prefix = f"baseline/{base_name}"
                        dfr_log_metrics[f'{prefix}/test_avg_loss'] = base_metrics['avg_loss']
                        dfr_log_metrics[f'{prefix}/test_worst_acc'] = base_metrics['worst_acc']
                        dfr_log_metrics[f'{prefix}/test_avg_acc'] = base_metrics['avg_acc']
                        for i in range(4):
                            dfr_log_metrics[f'{prefix}/test_group_{i}_loss'] = base_metrics['group_losses'][i]
                            dfr_log_metrics[f'{prefix}/test_group_{i}_acc'] = base_metrics['group_accs'][i]

                    wandb.log(dfr_log_metrics)
                    
                # テキスト結果保存
                with open(os.path.join(result_dir, 'dfr_results.txt'), 'w') as f:
                    f.write("DFR Evaluation Results (Same metrics as ERM)\n")
                    f.write("=========================================\n")
                    f.write(f"[DFR Model - Main Task (Predict Y)]\n")
                    f.write(f"Train Avg Loss: {dfr_train_metrics['avg_loss']:.4f}\n")
                    f.write(f"Train Worst Acc: {dfr_train_metrics['worst_acc']:.4f}\n")
                    f.write(f"Test Avg Loss: {dfr_test_metrics['avg_loss']:.4f}\n")
                    f.write(f"Test Worst Acc: {dfr_test_metrics['worst_acc']:.4f}\n")
                    f.write("\nTest Group Details:\n")
                    for i in range(4):
                        f.write(f"  Group {i}: Loss={dfr_test_metrics['group_losses'][i]:.4f}, Acc={dfr_test_metrics['group_accs'][i]:.4f}\n")
                    
                    f.write("\n=========================================\n")
                    f.write(f"[DFR Model - Spurious Task (Predict A)]\n")
                    f.write(f"Train Avg Loss: {dfr_spur_train_metrics['avg_loss']:.4f}\n")
                    f.write(f"Train Worst Acc: {dfr_spur_train_metrics['worst_acc']:.4f}\n")
                    f.write(f"Test Avg Loss: {dfr_spur_test_metrics['avg_loss']:.4f}\n")
                    f.write(f"Test Worst Acc: {dfr_spur_test_metrics['worst_acc']:.4f}\n")
                    f.write("\nTest Group Details (Aligned to Standard Groups Y, A):\n")
                    for i in range(4):
                         f.write(f"  Group {i}: Loss={dfr_spur_test_metrics['group_losses'][i]:.4f}, Acc={dfr_spur_test_metrics['group_accs'][i]:.4f}\n")
                    
                    f.write("\n=========================================\n")
                    f.write(f"[Analysis: Principal Angles (Subspace Geometry)]\n")
                    f.write(f"--- On Validation Set (Small N) ---\n")
                    f.write(f"Singular Values w.r.t Y (Label): {sing_vals_y}\n")
                    f.write(f"Singular Values w.r.t A (Spurious): {sing_vals_a}\n")
                    f.write(f"Feature Alignment (Y vs A) cos gamma_2: {align_val:.6f}\n")
                    
                    f.write(f"\n--- On Test Set (Large N) ---\n")
                    f.write(f"Singular Values w.r.t Y (Label): {sing_vals_y_test}\n")
                    f.write(f"Singular Values w.r.t A (Spurious): {sing_vals_a_test}\n")
                    f.write(f"Feature Alignment (Y vs A) cos gamma_2: {align_test:.6f}\n")

                    f.write("\n=========================================\n")
                    f.write("[Baselines & Comparisons (Main Task)]\n")
                    
                    # baseline_results: {'erm': ..., 'reg_none': ..., 'reg_l1': ..., 'reg_l2': ...}
                    print_order = ['erm', 'reg_none', 'reg_l1', 'reg_l2']
                    for base_key in print_order:
                        if base_key not in baseline_results: continue
                        
                        metrics = baseline_results[base_key]
                        f.write(f"\n--- {base_key.upper()} (Test Set) ---\n")
                        f.write(f"Avg Loss: {metrics['avg_loss']:.4f}\n")
                        f.write(f"Worst Acc: {metrics['worst_acc']:.4f}\n")
                        f.write("Group Details:\n")
                        for i in range(4):
                            f.write(f"  Group {i}: Loss={metrics['group_losses'][i]:.4f}, Acc={metrics['group_accs'][i]:.4f}\n")

            else:
                print("\n[Error] X_val_dfr is None. Cannot run DFR.")

        except Exception as e:
            print(f"\n[Error] DFR execution failed: {e}")
            import traceback
            traceback.print_exc()

    if config.get('wandb', {}).get('enable', False):
        wandb.finish()

    print(f"\nExperiment finished. All results saved in: {result_dir}")

if __name__ == '__main__':
    main(config_path='config.yaml')
