# sp_scr_v2/dfr.py

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import utils
import analysis
import warnings
from scipy.linalg import LinAlgWarning

warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class DFRTorchModel(nn.Module):
    """
    scikit-learnで学習したロジスティック回帰 (または線形回帰) モデルの重みを取り込み，
    PyTorchモデルとして振る舞うラッパー．
    これにより，utils.evaluate_model を使用してERMと同じ指標を計算できるようにする．
    """
    def __init__(self, dfr_coef, dfr_intercept, scaler):
        super().__init__()
        # sklearnの重み (Classes, Features) -> (1, Features) if binary
        # Ridge/Lassoの場合も coef_ は (Features,) または (1, Features) の形になるよう調整
        if dfr_coef.ndim == 1:
            dfr_coef = dfr_coef.reshape(1, -1)
            
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
    loss_type = config.get('dfr_loss_type', 'logistic') # 'logistic' or 'mse'

    # 正則化なし('none')の場合は探索不要なため，ダミーのリストを設定
    if reg_type == 'none':
        c_options = [1.0]
    else:
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
    y_tr_bal = y_tr[bal_idx] 

    if loss_type == 'mse':
        y_tr_bal = y_tr_bal.astype(float)
    else:
        y_tr_bal = y_tr_bal.astype(int)

    best_acc = -1.0
    best_c = c_options[0]
    
    print(f"  [DFR Tune] Splitting Source ({n_total}) -> Tune-Train ({len(X_tr_bal)} balanced) / Tune-Eval ({len(X_ev)})")
    print(f"  [DFR Tune] Method: {loss_type.upper()} Regression, Reg Type: {reg_type}")

    for c in c_options:
        if reg_type == 'none':
            # 正則化なしの場合
            if loss_type == 'logistic':
                # penalty=None は solver='lbfgs' 等が必要 (liblinearは不可)
                clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=42)
            elif loss_type == 'mse':
                clf = LinearRegression()
            else:
                raise ValueError(f"Unknown dfr_loss_type: {loss_type}")
            
            clf.fit(X_tr_bal, y_tr_bal)

        else:
            # 正則化あり (l1, l2) の場合
            if loss_type == 'logistic':
                clf = LogisticRegression(penalty=reg_type, C=c, solver='liblinear', random_state=42)
                clf.fit(X_tr_bal, y_tr_bal)
            
            elif loss_type == 'mse':
                # MSE回帰 (Ridge / Lasso)
                alpha = 1.0 / (c + 1e-9)
                if reg_type == 'l2':
                    clf = Ridge(alpha=alpha, random_state=42)
                elif reg_type == 'l1':
                    clf = Lasso(alpha=alpha, random_state=42)
                else:
                    clf = Ridge(alpha=alpha, random_state=42)
                
                clf.fit(X_tr_bal, y_tr_bal)
            else:
                raise ValueError(f"Unknown dfr_loss_type: {loss_type}")

        # --- 予測と評価 ---
        if loss_type == 'logistic':
            # クラス予測 (-1 or 1)
            preds = clf.predict(X_ev_scaled)
        else:
            # 連続値予測 -> 符号を取ってクラス予測とする
            preds_raw = clf.predict(X_ev_scaled)
            preds = np.sign(preds_raw)
            preds[preds == 0] = 1.0 

        # Worst Group Accuracy 計算 (Validation Set)
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
            
    if reg_type == 'none':
        print(f"  [DFR Tune] Reg Type is 'none'. (Val-Split WGA: {best_acc:.4f})")
    else:
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
    loss_type = config.get('dfr_loss_type', 'logistic')
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
        y_bal = y_source[bal_idx]
        
        if loss_type == 'mse':
            y_bal = y_bal.astype(float)
        else:
            y_bal = y_bal.astype(int)
        
        # --- 正則化なし ('none') の場合 ---
        if reg_type == 'none':
            if loss_type == 'logistic':
                clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=i)
            elif loss_type == 'mse':
                clf = LinearRegression()
            else:
                raise ValueError(f"Unknown dfr_loss_type: {loss_type}")
            
            clf.fit(X_bal, y_bal)
            
            # 係数取得
            c = clf.coef_
            if c.ndim == 1: c = c.reshape(1, -1)
            coefs.append(c)
            
            inter = clf.intercept_
            if np.isscalar(inter): inter = np.array([inter])
            intercepts.append(inter)

        # --- 正則化あり ('l1', 'l2') の場合 ---
        else:
            if loss_type == 'logistic':
                clf = LogisticRegression(penalty=reg_type, C=best_c, solver='liblinear', random_state=i)
                clf.fit(X_bal, y_bal)
                coefs.append(clf.coef_)
                intercepts.append(clf.intercept_)
                
            elif loss_type == 'mse':
                alpha = 1.0 / (best_c + 1e-9)
                if reg_type == 'l2':
                    clf = Ridge(alpha=alpha, random_state=i)
                elif reg_type == 'l1':
                    clf = Lasso(alpha=alpha, random_state=i)
                else:
                    clf = Ridge(alpha=alpha, random_state=i)

                clf.fit(X_bal, y_bal)
                
                c = clf.coef_
                if c.ndim == 1:
                    c = c.reshape(1, -1)
                coefs.append(c)
                
                inter = clf.intercept_
                if np.isscalar(inter):
                    inter = np.array([inter])
                intercepts.append(inter)

    # 重みの平均化
    avg_coef = np.mean(coefs, axis=0)
    avg_intercept = np.mean(intercepts, axis=0)
    
    return avg_coef, avg_intercept

def train_baseline_model(X, y, loss_type, reg_type, c_options, random_state=42):
    """
    学習データ全体 (Imbalanced) を使用してベースラインの回帰モデルを学習する．
    正則化ありの場合は，学習データを分割してAvg Accに基づいてCをチューニングする．
    """
    # データをチューニング用に分割 (80/20)
    n = len(X)
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n)
    n_train = int(n * 0.8)
    idx_tr, idx_val = perm[:n_train], perm[n_train:]
    
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_val, y_val = X[idx_val], y[idx_val]

    # 型変換
    if loss_type == 'mse':
        y_tr = y_tr.astype(float); y_val = y_val.astype(float)
        y_full = y.astype(float)
    else:
        y_tr = y_tr.astype(int); y_val = y_val.astype(int)
        y_full = y.astype(int)
        
    best_c_val = 1.0 # Default
    
    # --- No Regularization ---
    if reg_type == 'none':
        if loss_type == 'logistic':
             # penalty=None for sklearn >= 1.2
             clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=random_state)
        else:
             clf = LinearRegression()
             
        # チューニングなしで全データで学習
        clf.fit(X, y_full)
        
    # --- L1 / L2 Regularization (Tune C) ---
    else:
        best_acc = -1.0
        best_c_val = c_options[0]
        
        for c in c_options:
            if loss_type == 'logistic':
                solver = 'liblinear' if reg_type=='l1' else 'lbfgs'
                clf = LogisticRegression(penalty=reg_type, C=c, solver=solver, max_iter=1000, random_state=random_state)
            else:
                alpha = 1.0 / (c + 1e-9)
                if reg_type=='l1': clf = Lasso(alpha=alpha, random_state=random_state)
                else: clf = Ridge(alpha=alpha, random_state=random_state)
            
            try:
                clf.fit(X_tr, y_tr)
                
                # Validation (Avg Acc)
                preds = clf.predict(X_val)
                if loss_type == 'mse':
                    # 符号で判定
                    preds_cls = np.sign(preds); preds_cls[preds_cls==0] = 1.0
                    acc = np.mean(preds_cls == (y_val > 0).astype(float)*2-1)
                else:
                    acc = np.mean(preds == y_val)
                
                if acc > best_acc:
                    best_acc = acc
                    best_c_val = c
            except Exception as e:
                # 収束しない場合などはスキップ
                continue
        
        # 全データで再学習 (Best C)
        if loss_type == 'logistic':
            solver = 'liblinear' if reg_type=='l1' else 'lbfgs'
            clf = LogisticRegression(penalty=reg_type, C=best_c_val, solver=solver, max_iter=1000, random_state=random_state)
        else:
            alpha = 1.0 / (best_c_val + 1e-9)
            if reg_type=='l1': clf = Lasso(alpha=alpha, random_state=random_state)
            else: clf = Ridge(alpha=alpha, random_state=random_state)
            
        clf.fit(X, y_full)

    coef = clf.coef_
    intercept = clf.intercept_
    
    # 形状調整
    if coef.ndim == 1: coef = coef.reshape(1, -1)
    if np.isscalar(intercept): intercept = np.array([intercept])
    
    return coef, intercept, (best_c_val if reg_type != 'none' else None)


def run_dfr_procedure(config, model, X_train, y_train, a_train, X_test, y_test, a_test, device, loss_function_name, X_val=None, y_val=None, a_val=None):
    """
    DFRの実行プロセス全体
    X_val, y_val, a_val は main.py で事前に分割された Held-out Validation データが渡されることを前提とする
    """
    print("\n" + "="*30)
    print(" Running Deep Feature Reweighting (DFR)")
    print("="*30)
    
    # 対象とする層の名前をConfigから取得
    target_layer_name = config.get('dfr_target_layer', 'last_hidden')
    eval_bs = config.get('eval_batch_size', 5000)
    if eval_bs is None: eval_bs = 5000

    # --- 0. DFR適用前 (ERM) の結果を計算 (比較用) ---
    print("Evaluating Original ERM Model (Before DFR)...")
    # 特徴量ではなく元の入力(X_test)を使用
    erm_test_metrics = utils.evaluate_model(
        model, X_test, y_test, a_test, 
        device, loss_function_name, eval_bs
    )
    print("ERM Test Group Details (Before DFR):")
    for i, loss in enumerate(erm_test_metrics['group_losses']):
        acc = erm_test_metrics['group_accs'][i]
        print(f"  Group {i}: Loss={loss:.4f}, Acc={acc:.4f}")


    # --- 1. 特徴抽出 ---
    print("\nExtracting embeddings from the trained model...")
    # TrainデータはStandardScalerの学習(fit)のために必要
    train_embeddings = get_embeddings(model, X_train, target_layer_name, device=device)
    test_embeddings = get_embeddings(model, X_test, target_layer_name, device=device)

    # StandardScaler は Trainデータ で Fit させる (著者実装準拠)
    print("Fitting StandardScaler on TRAIN data...")
    scaler = StandardScaler()
    scaler.fit(train_embeddings)

    # Validationデータの準備 (DFRの学習ソース)
    if X_val is None:
        raise ValueError("DFR validation data (X_val) is None. Check if use_dfr is enabled and splitting logic in main.py is correct.")
    
    print("Extracting embeddings for Validation data...")
    val_embeddings = get_embeddings(model, X_val, target_layer_name, device=device)
    
    X_dfr_source = val_embeddings
    y_dfr_source = y_val.cpu().numpy()
    a_dfr_source = a_val.cpu().numpy()

    # --- 2. DFR Tune & Train ---
    # Validationを分割してCを決める
    best_c = dfr_tune(X_dfr_source, y_dfr_source, a_dfr_source, scaler, config)
    # Validation全体を使って再学習
    avg_coef, avg_intercept = dfr_train(X_dfr_source, y_dfr_source, a_dfr_source, best_c, scaler, config)
    
    # --- 3. DFR PyTorchモデル化 & 評価 ---
    dfr_torch_model = DFRTorchModel(avg_coef, avg_intercept, scaler).to(device)
    
    print("\n--- DFR Evaluation (Using exact same metrics as ERM) ---")
    
    X_train_emb_tensor = torch.from_numpy(train_embeddings).to(device)
    X_test_emb_tensor = torch.from_numpy(test_embeddings).to(device)
    
    dfr_train_metrics = utils.evaluate_model(
        dfr_torch_model, X_train_emb_tensor, y_train, a_train, 
        device, loss_function_name, eval_bs
    )
    
    dfr_test_metrics = utils.evaluate_model(
        dfr_torch_model, X_test_emb_tensor, y_test, a_test, 
        device, loss_function_name, eval_bs
    )

    # モデル出力統計量
    if config.get('analyze_model_output_expectation', False):
        print("  Calculating DFR model output statistics...")
        train_out_stats = analysis.analyze_model_output_expectation(
            dfr_torch_model, X_train_emb_tensor, y_train, a_train, device, eval_bs
        )
        test_out_stats = analysis.analyze_model_output_expectation(
            dfr_torch_model, X_test_emb_tensor, y_test, a_test, device, eval_bs
        )
        dfr_train_metrics.update(train_out_stats)
        test_metrics_with_stats = dfr_test_metrics.copy()
        test_metrics_with_stats.update(test_out_stats)
        dfr_test_metrics = test_metrics_with_stats

    print(f"DFR Train | Loss: {dfr_train_metrics['avg_loss']:.4f}, Worst Acc: {dfr_train_metrics['worst_acc']:.4f}")
    print(f"DFR Test  | Loss: {dfr_test_metrics['avg_loss']:.4f}, Worst Acc: {dfr_test_metrics['worst_acc']:.4f}")
    
    print("DFR Test Group Details:")
    for i, loss in enumerate(dfr_test_metrics['group_losses']):
        acc = dfr_test_metrics['group_accs'][i]
        print(f"  Group {i}: Loss={loss:.4f}, Acc={acc:.4f}")


    # --- 4. Baseline Regressions (ERM on Features) ---
    # 学習データ全体(特徴量)を使って単純な回帰を行う
    # 比較: No Reg, L1, L2
    print("\n--- Baseline Regressions on Training Features (Imbalanced) ---")
    
    X_train_np = train_embeddings
    y_train_np = y_train.cpu().numpy()
    
    # Scale X_train for training (scaler is already fit on it, so transform)
    X_train_scaled = scaler.transform(X_train_np)
    
    loss_type = config.get('dfr_loss_type', 'logistic')
    c_options = config.get('dfr_c_options', [1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01])
    
    baseline_results = {}
    baseline_results['erm'] = erm_test_metrics # オリジナルモデルの結果も格納
    
    reg_configs = [('none', 'none'), ('l1', 'l1'), ('l2', 'l2')]
    
    for name, reg_type in reg_configs:
        print(f"Training Baseline: {loss_type.upper()} + {reg_type.upper()}...")
        
        coef, intercept, best_c_base = train_baseline_model(
            X_train_scaled, y_train_np, loss_type, reg_type, c_options
        )
        
        if best_c_base:
            print(f"  Best C: {best_c_base}")
            
        # Baseline Model化
        base_torch_model = DFRTorchModel(coef, intercept, scaler).to(device)
        
        # 評価
        base_test_metrics = utils.evaluate_model(
            base_torch_model, X_test_emb_tensor, y_test, a_test,
            device, loss_function_name, eval_bs
        )
        
        baseline_results[f'reg_{name}'] = base_test_metrics
        
        print(f"Baseline (Feature Space, Loss={loss_type}, Reg={reg_type.upper()}) Test Group Details:")
        for i, loss in enumerate(base_test_metrics['group_losses']):
            acc = base_test_metrics['group_accs'][i]
            print(f"  Group {i}: Loss={loss:.4f}, Acc={acc:.4f}")
            
    return dfr_train_metrics, dfr_test_metrics, baseline_results
