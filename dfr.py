# sp_scr_v2/dfr.py

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import utils
import analysis
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class DFRTorchModel(nn.Module):
    """
    scikit-learnで学習したロジスティック回帰モデルの重みを取り込み，
    PyTorchモデルとして振る舞うラッパー．
    これにより，utils.evaluate_model を使用してERMと完全に同じ指標を計算できるようにする．
    """
    def __init__(self, dfr_coef, dfr_intercept, scaler):
        super().__init__()
        # sklearnの重み (Classes, Features) -> (1, Features) if binary
        input_dim = dfr_coef.shape[1]
        self.linear = nn.Linear(input_dim, 1) # binary output
        
        # 重みのコピー
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(dfr_coef).float())
            self.linear.bias.copy_(torch.from_numpy(dfr_intercept).float())
            
        # Standard Scalerのパラメータ (mean, scale)
        self.scaler_mean = torch.from_numpy(scaler.mean_).float()
        self.scaler_scale = torch.from_numpy(scaler.scale_).float()
        
    def forward(self, x):
        # x: (B, D) embeddings
        
        # デバイス合わせ
        device = x.device
        if self.scaler_mean.device != device:
            self.scaler_mean = self.scaler_mean.to(device)
            self.scaler_scale = self.scaler_scale.to(device)
            
        # 1. Standard Scaling (sklearnのpreprocessingを再現)
        # z = (x - u) / s
        x_scaled = (x - self.scaler_mean) / self.scaler_scale
        
        # 2. Linear Layer
        logits = self.linear(x_scaled).squeeze(-1) # (B, 1) -> (B,)
        
        return logits, {}

def get_embeddings(model, X, target_layer_name_config, batch_size=1000, device='cpu'):
    """
    モデルから指定された層の特徴量を抽出する
    """
    model.eval()
    embeddings = []
    
    # ターゲット層の名前を決定
    target_layer_name = target_layer_name_config
    if target_layer_name == "last_hidden":
        # MLPモデルの最終隠れ層 (layer_N)
        target_layer_name = f'layer_{model.num_hidden_layers}'
    
    print(f"  [DFR] Extracting features from layer: {target_layer_name}")

    with torch.no_grad():
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            x_batch = X[batch_indices].to(device)
            
            # Forward pass
            _, outputs = model(x_batch)
            
            # 指定された層の出力を取得
            if target_layer_name in outputs:
                emb = outputs[target_layer_name]
            else:
                keys = sorted([k for k in outputs.keys() if k.startswith('layer_')])
                if target_layer_name_config == "last_hidden" and keys:
                    emb = outputs[keys[-1]]
                    print(f"  [Warning] 'last_hidden' ({target_layer_name}) not found. Using {keys[-1]} instead.")
                else:
                    raise ValueError(f"Could not find layer '{target_layer_name}' in model outputs. Available keys: {list(outputs.keys())}")
            
            embeddings.append(emb.cpu().numpy())
            
    return np.concatenate(embeddings, axis=0)

def balance_indices(y, a, random_state=None):
    """
    グループごとに最小サンプル数に合わせてダウンサンプリングするインデックスを返す
    """
    # Global seedへの副作用を防ぐため，ローカルなRandomStateを使用
    rng = np.random.RandomState(random_state)

    groups = list(zip(y, a))
    unique_groups = sorted(list(set(groups)))
    
    group_indices = defaultdict(list)
    for idx, g in enumerate(groups):
        group_indices[g].append(idx)
        
    min_size = min([len(idxs) for idxs in group_indices.values()])
    
    balanced_indices = []
    for g in unique_groups:
        idxs = group_indices[g]
        selected_idxs = rng.choice(idxs, min_size, replace=False)
        balanced_indices.extend(selected_idxs)
        
    return np.array(balanced_indices)

def dfr_tune(X_source, y_source, a_source, scaler, config):
    """
    [著者実装準拠] 
    Validationデータ (X_source) をさらに2分割してハイパーパラメータ(C)を探索する
    scaler: Trainデータでfit済みのStandardScaler
    """
    reg_type = config.get('dfr_reg', 'l1')
    c_options = config.get('dfr_c_options', [1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01])
    
    n_total = len(X_source)
    n_half = n_total // 2
    
    # ランダムに分割 (前半: 学習用, 後半: 評価用)
    indices = np.random.permutation(n_total)
    idx_train = indices[n_half:] 
    idx_eval = indices[:n_half]
    
    X_tr = X_source[idx_train]
    y_tr = y_source[idx_train]
    a_tr = a_source[idx_train]
    
    X_ev = X_source[idx_eval]
    y_ev = y_source[idx_eval]
    a_ev = a_source[idx_eval]
    
    # scalerで変換 (scalerはTrainでfit済み)
    X_tr_scaled = scaler.transform(X_tr)
    X_ev_scaled = scaler.transform(X_ev)
    
    # バランスサンプリング (学習データのみ)
    bal_idx = balance_indices(y_tr, a_tr, random_state=42)
    X_tr_bal = X_tr_scaled[bal_idx]
    y_tr_bal = y_tr[bal_idx].astype(int)
    
    best_acc = -1.0
    best_c = c_options[0]
    
    print(f"  [DFR Tune] Splitting Source ({n_total}) -> Tune-Train ({len(X_tr_bal)} balanced) / Tune-Eval ({len(X_ev)})")

    for c in c_options:
        clf = LogisticRegression(penalty=reg_type, C=c, solver='liblinear', random_state=42)
        clf.fit(X_tr_bal, y_tr_bal)
        
        preds = clf.predict(X_ev_scaled)
        
        # Worst Group Accuracy 計算
        unique_groups = sorted(list(set(zip(y_ev, a_ev))))
        accuracies = []
        for g in unique_groups:
            mask = (y_ev == g[0]) & (a_ev == g[1])
            if np.sum(mask) > 0:
                acc = np.mean(preds[mask] == y_ev[mask])
                accuracies.append(acc)
        
        wga = min(accuracies) if accuracies else 0.0
        
        if wga > best_acc:
            best_acc = wga
            best_c = c
            
    print(f"  [DFR Tune] Best C: {best_c} (Val-Split WGA: {best_acc:.4f})")
    return best_c

def dfr_train(X_source, y_source, a_source, best_c, scaler, config):
    """
    [著者実装準拠]
    Validationデータ全体 (X_source) を使って再学習 (Retraining)
    複数回行って重みを平均化する
    scaler: Trainデータでfit済みのStandardScaler
    """
    reg_type = config.get('dfr_reg', 'l1')
    num_retrains = config.get('dfr_num_retrains', 10)
    
    # 全データ変換 (scalerはTrainでfit済み)
    X_scaled = scaler.transform(X_source)
    
    coefs = []
    intercepts = []
    
    print(f"  [DFR Train] Retraining {num_retrains} times on full source dataset ({len(X_source)} samples)...")
    
    for i in range(num_retrains):
        # 毎回異なるシードでバランスサンプリング
        bal_idx = balance_indices(y_source, a_source, random_state=i)
        X_bal = X_scaled[bal_idx]
        y_bal = y_source[bal_idx].astype(int)
        
        clf = LogisticRegression(penalty=reg_type, C=best_c, solver='liblinear', random_state=i)
        clf.fit(X_bal, y_bal)
        
        coefs.append(clf.coef_)
        intercepts.append(clf.intercept_)
        
    # 重みの平均化
    avg_coef = np.mean(coefs, axis=0)
    avg_intercept = np.mean(intercepts, axis=0)
    
    return avg_coef, avg_intercept

def run_dfr_procedure(config, model, X_train, y_train, a_train, X_test, y_test, a_test, device, loss_function_name, X_val=None, y_val=None, a_val=None):
    """
    DFRの実行プロセス全体
    """
    print("\n" + "="*30)
    print(" Running Deep Feature Reweighting (DFR)")
    print("="*30)
    
    # 対象とする層の名前をConfigから取得
    target_layer_name = config.get('dfr_target_layer', 'last_hidden')

    # 1. 特徴抽出
    print("Extracting embeddings from the trained model...")
    # TrainデータはStandardScalerの学習(fit)のために必要
    train_embeddings = get_embeddings(model, X_train, target_layer_name, device=device)
    test_embeddings = get_embeddings(model, X_test, target_layer_name, device=device)

    # StandardScaler は Trainデータ で Fit させる (著者実装準拠)
    print("Fitting StandardScaler on TRAIN data...")
    scaler = StandardScaler()
    scaler.fit(train_embeddings)

    # Validationデータの準備 (DFRの学習ソース)
    val_strategy = config.get('dfr_val_split_strategy', 'original')
    val_ratio = config.get('dfr_val_ratio', 0.2)
    
    X_dfr_source = None
    y_dfr_source = None
    a_dfr_source = None

    if val_strategy == 'original':
        if X_val is not None:
            print("Using ORIGINAL Validation Set for DFR (Author's DFR_Tr^Val setting).")
            X_dfr_source = get_embeddings(model, X_val, target_layer_name, device=device)
            y_dfr_source = y_val.cpu().numpy()
            a_dfr_source = a_val.cpu().numpy()
        else:
            print("[Warning] 'original' strategy selected but no Validation set provided. Fallback to 'split_from_train'.")
            val_strategy = 'split_from_train'
            
    if val_strategy == 'split_from_train':
        print(f"Using SPLIT Validation Set from Train (Ratio: {val_ratio}).")
        n_train = len(train_embeddings)
        n_dfr = int(n_train * val_ratio)
        
        indices = np.random.permutation(n_train)
        dfr_idx = indices[:n_dfr] # 前半をDFR用(Validation代わり)に使用
        
        X_dfr_source = train_embeddings[dfr_idx]
        y_dfr_source = y_train.cpu().numpy()[dfr_idx]
        a_dfr_source = a_train.cpu().numpy()[dfr_idx]

    if X_dfr_source is None:
        raise ValueError("DFR source data could not be prepared.")

    # 2. DFR Tune (Validationを分割してCを決める) - fit済みscalerを渡す
    best_c = dfr_tune(X_dfr_source, y_dfr_source, a_dfr_source, scaler, config)
    
    # 3. DFR Train (Validation全体を使って再学習) - fit済みscalerを渡す
    avg_coef, avg_intercept = dfr_train(X_dfr_source, y_dfr_source, a_dfr_source, best_c, scaler, config)
    
    # 4. PyTorchモデル化
    dfr_torch_model = DFRTorchModel(avg_coef, avg_intercept, scaler).to(device)
    
    # 5. 評価 (ERMと完全に同一の指標を使用)
    print("\n--- DFR Evaluation (Using exact same metrics as ERM) ---")
    
    X_train_emb_tensor = torch.from_numpy(train_embeddings).to(device)
    X_test_emb_tensor = torch.from_numpy(test_embeddings).to(device)
    
    eval_bs = config.get('eval_batch_size', 5000)
    if eval_bs is None: eval_bs = 5000

    train_metrics = utils.evaluate_model(
        dfr_torch_model, X_train_emb_tensor, y_train, a_train, 
        device, loss_function_name, eval_bs
    )
    
    test_metrics = utils.evaluate_model(
        dfr_torch_model, X_test_emb_tensor, y_test, a_test, 
        device, loss_function_name, eval_bs
    )

    # モデル出力の統計量 (平均・標準偏差) の計算
    if config.get('analyze_model_output_expectation', False):
        print("  Calculating DFR model output statistics...")
        # Train
        train_out_stats = analysis.analyze_model_output_expectation(
            dfr_torch_model, X_train_emb_tensor, y_train, a_train, device, eval_bs
        )
        # Test
        test_out_stats = analysis.analyze_model_output_expectation(
            dfr_torch_model, X_test_emb_tensor, y_test, a_test, device, eval_bs
        )
        
        # 結果をメトリクスにマージ
        train_metrics.update(train_out_stats)
        test_metrics.update(test_out_stats)

    print(f"DFR Train | Loss: {train_metrics['avg_loss']:.4f}, Worst Acc: {train_metrics['worst_acc']:.4f}")
    print(f"DFR Test  | Loss: {test_metrics['avg_loss']:.4f}, Worst Acc: {test_metrics['worst_acc']:.4f}")
    
    print("DFR Test Group Details:")
    for i, loss in enumerate(test_metrics['group_losses']):
        acc = test_metrics['group_accs'][i]
        print(f"  Group {i}: Loss={loss:.4f}, Acc={acc:.4f}")

    return train_metrics, test_metrics
