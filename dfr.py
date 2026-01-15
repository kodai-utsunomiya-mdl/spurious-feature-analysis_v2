# sp_scr_v2/dfr.py

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import scipy.linalg
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
            
        # 1. Standard Scaling
        # z = (x - u) / s
        # scalerが恒等変換(mean=0, scale=1)の場合は何もしないのと同じ
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
    target_layer_name = target_layer_name_config
    
    # "last_hidden" が指定された場合のみ，モデル構造に応じて実際の層名を解決する
    if target_layer_name == "last_hidden":
        if hasattr(model, 'model_type'):
            if model.model_type == 'ResNet':
                # ResNet: Input(0) -> Proj(1) -> Blocks(L個, idx 2..L+1)
                # 最終層は layer_{L+1} (L=num_blocks)
                last_idx = model.num_blocks + 1
            else:
                # MLP: Input(0) -> Hidden(H個, idx 1..H)
                # 最終層は layer_{H} (H=total_hidden_layers)
                last_idx = model.total_hidden_layers
            
            target_layer_name = f'layer_{last_idx}'
        else:
            if hasattr(model, 'num_hidden_layers'):
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
                
                # キーが存在し，かつ "last_hidden" 指定だった場合のフォールバック
                if target_layer_name_config == "last_hidden" and keys:
                    # キーをパースして最大のインデックスを探す
                    try:
                        max_idx = -1
                        for k in keys:
                            idx = int(k.split('_')[1])
                            if idx > max_idx:
                                max_idx = idx
                        actual_last_layer = f'layer_{max_idx}'
                        emb = outputs[actual_last_layer]
                        print(f"  [Warning] Calculated target '{target_layer_name}' not found. Using '{actual_last_layer}' instead.")
                    except:
                        emb = outputs[keys[-1]]
                        print(f"  [Warning] Calculated target '{target_layer_name}' not found. Using '{keys[-1]}' instead.")
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
    [原論文の実装に準拠] 
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
    
    # scalerで変換 (scalerはTrainでfit済み，あるいはダミー設定済み)
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
    [原論文の実装に準拠: Standard DFR]
    Validationデータ全体 (X_source) から均衡化サブセットをサンプリングして再学習 (Retraining)
    複数回行って重みを平均化する
    scaler: Trainデータでfit済みのStandardScaler
    """
    reg_type = config.get('dfr_reg', 'l1')
    loss_type = config.get('dfr_loss_type', 'logistic')
    num_retrains = config.get('dfr_num_retrains', 10)
    
    # 全データ変換 (scalerはTrainでfit済み，あるいはダミー)
    X_scaled = scaler.transform(X_source)
    
    coefs = []
    intercepts = []
    
    print(f"  [DFR Train (Standard)] Retraining {num_retrains} times on balanced subsets from source dataset ({len(X_source)} samples)...")
    
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

def project_simplex(v, z=1.0):
    """
    ベクトル v を確率単体 (sum(w) = z, w >= 0) に射影する．
    """
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = np.maximum(v - theta, 0)
    return w

def dfr_train_minimax(X_source, y_source, a_source, best_c, scaler, config):
    """
    [Projected Gradient Ascent for Worst-Group Risk Minimization]
    Validationデータ全体を使用し，グループ重み lambda を最適化しながら
    重み付きRidge回帰を解くことで，Minimax解 (最悪グループリスク最小化解) を求める．
    Bagging を行うことで推定を安定化させる．
    
    Args:
        best_c: dfr_tune で決定されたハイパーパラメータ (正則化係数の逆数 C)
                Ridgeの場合 alpha = 1 / best_c
    """
    # 設定の読み込み
    step_size = config.get('dfr_minimax_step_size', 0.01)
    iterations = config.get('dfr_minimax_iterations', 100)
    num_bootstraps = config.get('dfr_minimax_num_bootstraps', 1) # ブートストラップ回数 K
    
    # 損失関数は二乗誤差 (Ridge) を前提
    alpha = 1.0 / (best_c + 1e-9) # 正則化係数 lambda
    
    # 全データ変換 (scalerはTrainでfit済み，あるいはダミー)
    X_scaled = scaler.transform(X_source) # (N, m)
    
    # ラベル (float)
    y_float = y_source.astype(float)
    
    # 特徴量にバイアス項 (定数1) を追加: (N, m+1)
    N, m = X_scaled.shape
    X_aug_source = np.hstack([X_scaled, np.ones((N, 1))])
    
    # 最終的なパラメータの累積用
    w_aug_accum = np.zeros(m + 1)
    
    print(f"  [DFR Train (Minimax)] Starting optimization (T={iterations}, eta={step_size}, alpha={alpha:.4f}, K={num_bootstraps})...")
    
    for k in range(num_bootstraps):
        # --- Bootstrap Sampling ---
        # 復元抽出でインデックスを生成
        boot_indices = np.random.choice(N, N, replace=True)
        X_aug = X_aug_source[boot_indices]
        y_boot = y_float[boot_indices]
        a_boot = a_source[boot_indices]
        
        # グループごとの統計量 (共分散行列 Sigma_g, 相関ベクトル v_g) を事前計算
        # グループ定義: (-1,-1), (-1,1), (1,-1), (1,1)
        groups = sorted(list(set(zip(y_boot, a_boot))))
        group_stats = {}
        
        for g in groups:
            y_val, a_val = g
            mask = (y_boot == y_val) & (a_boot == a_val)
            X_g = X_aug[mask]
            y_g = y_boot[mask]
            n_g = len(X_g)
            
            if n_g == 0:
                continue
                
            # Sigma_g = (1/n_g) * X_g^T X_g  (Shape: m+1, m+1)
            Sigma_g = (X_g.T @ X_g) / n_g
            
            # v_g = (1/n_g) * X_g^T y_g      (Shape: m+1,)
            v_g = (X_g.T @ y_g) / n_g
            
            group_stats[g] = {
                'Sigma': Sigma_g,
                'v': v_g,
                'X': X_g, # リスク計算用
                'y': y_g
            }
        
        active_groups = list(group_stats.keys())
        num_groups = len(active_groups)
        
        if num_groups == 0:
            print(f"    [Bootstrap {k+1}/{num_bootstraps}] Warning: No active groups found.")
            continue

        # 重み lambda の初期化 (一様)
        lambdas = np.ones(num_groups) / num_groups
        
        w_aug_k = None
        
        for t in range(iterations):
            # 1. 重み付き統計量の計算
            # M(lambda) = sum lambda_g Sigma_g
            # v(lambda) = sum lambda_g v_g
            M_lambda = np.zeros((m+1, m+1))
            v_lambda = np.zeros(m+1)
            
            for i, g in enumerate(active_groups):
                lam = lambdas[i]
                M_lambda += lam * group_stats[g]['Sigma']
                v_lambda += lam * group_stats[g]['v']
                
            # 2. 正則化項を加算して線形方程式を解く (Ridge Regression)
            # (M + alpha I) w = v
            # バイアス項 (index m) の正則化係数は 0 とする
            
            Reg_matrix = np.eye(m+1) * alpha
            Reg_matrix[m, m] = 0.0 # バイアス項は正則化しない
            
            A = M_lambda + Reg_matrix
            b_vec = v_lambda
            
            # 解く: w_aug = [w; b]
            try:
                w_aug_k = scipy.linalg.solve(A, b_vec, assume_a='pos')
            except scipy.linalg.LinAlgError:
                # 特異行列対策
                w_aug_k = scipy.linalg.solve(A + np.eye(m+1)*1e-6, b_vec)
                
            # 3. 各グループのリスク (R_g) を計算
            # R_g = (1/n_g) || y_g - X_g w_aug ||^2
            group_risks = np.zeros(num_groups)
            
            for i, g in enumerate(active_groups):
                X_g = group_stats[g]['X']
                y_g = group_stats[g]['y']
                preds = X_g @ w_aug_k
                loss = np.mean((y_g - preds)**2)
                group_risks[i] = loss
                
            # 4. 重み lambda の更新 (Gradient Ascent)
            # u = lambda + eta * (R - 1) 
            # (R - 1) は勾配方向．定数 1 は Lagrangian の双対問題における項．
            
            grad = group_risks
            u = lambdas + step_size * grad
            
            # 5. 確率単体への射影
            lambdas = project_simplex(u)
            
        # ログ出力 (各ブートストラップの最終状態)
        worst_risk = np.max(group_risks) if num_groups > 0 else 0.0
        if (k + 1) % max(1, num_bootstraps // 5) == 0:
            print(f"    [Bootstrap {k+1}/{num_bootstraps}] Final Worst Risk={worst_risk:.4f}, Lambdas={np.array2string(lambdas, precision=3)}")
        
        # モデルパラメータの累積
        if w_aug_k is not None:
            w_aug_accum += w_aug_k

    # パラメータの平均化 (Bagging)
    final_w_aug = w_aug_accum / num_bootstraps

    # 最終的なパラメータの分離
    coef = final_w_aug[:m]
    intercept = final_w_aug[m]
    
    # 形状調整 (1, m) および (1,)
    coef = coef.reshape(1, -1)
    intercept = np.array([intercept])
    
    return coef, intercept


def train_baseline_model(X, y, loss_type, reg_type, c_options, random_state=42):
    """
    学習データ全体 (Imbalanced) を使用してベースラインの回帰モデルを学習する．
    正則化ありの場合は，学習データを分割してAvg Accに基づいてCをチューニングする．
    """
    # データをチューニング用に分割
    n = len(X)
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n)
    n_train = int(n * 0.8)
    idx_tr, idx_val = perm[:n_train], perm[n_train:]
    
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_val, y_val = X[idx_val], y[idx_val]

    # 型の変換
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


def _reorder_metrics_for_spurious_task(metrics):
    """
    スプリアス属性を予測するタスク (Target=A, Attr=Y) において，
    evaluate_model が返すグループ順序は 2*A + Y となるため，
    Index 0: A=-1, Y=-1 -> Group 0 (Original)
    Index 1: A=-1, Y=+1 -> Group 2 (Original)
    Index 2: A=+1, Y=-1 -> Group 1 (Original)
    Index 3: A=+1, Y=+1 -> Group 3 (Original)
    となっている．
    これを通常のグループ順序 (0, 1, 2, 3) に対応させるため，
    Index 1 と Index 2 を入れ替える．
    """
    losses = metrics['group_losses']
    accs = metrics['group_accs']
    
    if len(losses) == 4 and len(accs) == 4:
        # Swap index 1 and 2
        new_losses = [losses[0], losses[2], losses[1], losses[3]]
        new_accs = [accs[0], accs[2], accs[1], accs[3]]
        
        metrics['group_losses'] = new_losses
        metrics['group_accs'] = new_accs
        
    return metrics

def compute_geometric_metrics(Z, y_discrete, a_discrete):
    """
    正準角 (Principal Angles) と特徴間のアラインメントを計算する．
    """
    N, m = Z.shape
    tol = 1e-10

    # ターゲットベクトル (中心化前)
    y_vec = y_discrete.astype(float)
    a_vec = a_discrete.astype(float)

    # ベクトルの中心化
    # J_N v = v - mean(v)
    y_cent = y_vec - np.mean(y_vec)
    a_cent = a_vec - np.mean(a_vec)
    
    metrics = {
        'sigma_2_Y': 0.0,
        'sigma_2_A': 0.0,
        'alignment_cos_gamma_2': 0.0
    }
    
    # 特徴行列 Phi (N, m+1)
    Phi = np.hstack([Z, np.ones((N, 1))])
    
    if N > m + 1:
        # --- Case 1: N > m (通常のSVD) ---

        # 1. 特徴空間の正規直交基底 U_Phi (N, rank_Phi)
        try:
            U_Phi, S_Phi, _ = np.linalg.svd(Phi, full_matrices=False)
            rank_Phi = np.sum(S_Phi > tol)
            U_Phi = U_Phi[:, :rank_Phi]
            
            # 射影行列 P_Phi = U_Phi U_Phi^T
            # ターゲット空間 V_Y, V_A の基底
            # V_Y = span{1, Y}, V_A = span{1, A}
            # 定数項 1 を含むため，QR分解で正規直交基底を得る
            def get_subspace_basis(target_vec):
                M = np.stack([target_vec, np.ones(N)], axis=1) # (N, 2)
                Q, _ = np.linalg.qr(M)
                return Q
            
            U_Y = get_subspace_basis(y_vec)
            U_A = get_subspace_basis(a_vec)
            
            # --- 正準角 sigma_2 (Subspace Principal Angle) ---
            # sigma_k = svd(U_Phi^T U_T)
            # 第1正準角は定数項により常に1．第2正準角 (sigma_2) を取得．
            
            def compute_sigma2(U_base, U_target):
                # U_base^T U_target (rank_Phi, 2)
                M_int = np.dot(U_base.T, U_target)
                s_vals = np.linalg.svd(M_int, compute_uv=False)
                # 降順にソートされている．s_vals[0] approx 1. s_vals[1] が sigma_2.
                if len(s_vals) >= 2:
                    return s_vals[1]
                return 0.0
            
            metrics['sigma_2_Y'] = compute_sigma2(U_Phi, U_Y)
            metrics['sigma_2_A'] = compute_sigma2(U_Phi, U_A)
            
            # --- 特徴間のアラインメント ---
            # 射影: P_Phi t
            # P_Phi = U_Phi U_Phi^T
            
            # P_Phi y
            proj_y = U_Phi @ (U_Phi.T @ y_vec)
            # P_Phi a
            proj_a = U_Phi @ (U_Phi.T @ a_vec)
            
            # 中心化 (J_N P_Phi t)
            proj_y_cent = proj_y - np.mean(proj_y)
            proj_a_cent = proj_a - np.mean(proj_a)
            
            # コサイン類似度
            norm_y = np.linalg.norm(proj_y_cent)
            norm_a = np.linalg.norm(proj_a_cent)
            
            if norm_y > tol and norm_a > tol:
                cos_gamma = np.abs(np.dot(proj_y_cent, proj_a_cent)) / (norm_y * norm_a)
                metrics['alignment_cos_gamma_2'] = cos_gamma
            else:
                metrics['alignment_cos_gamma_2'] = 0.0

        except Exception as e:
            print(f"  [Error] Geometric analysis (N > m) failed: {e}")

    else:
        # --- Case 2: N <= m (カーネル法) ---
        # Moore-Penrose一般化逆行列を使用
        
        try:
            # 中心化したカーネル行列 K = J_N Phi Phi^T J_N
            # Phi Phi^T
            G = np.dot(Phi, Phi.T) # (N, N)
            
            # J_N G J_N
            # J_N は定数ベクトルの成分を除去する (行平均・列平均を引くのと等価)
            # Centering matrix J_N = I - 11^T/N
            # K = J G J
            
            # 行平均を引く
            G_centered_rows = G - np.mean(G, axis=0, keepdims=True)
            # 列平均を引く
            K = G_centered_rows - np.mean(G_centered_rows, axis=1, keepdims=True)
            
            # Kの一般化逆行列 K_dagger
            K_dagger = scipy.linalg.pinv(K)
            
            # 予測ベクトル hat_t = K K_dagger J_N t
            # J_N t は中心化されたターゲット (y_cent, a_cent)
            
            hat_y = K @ (K_dagger @ y_cent)
            hat_a = K @ (K_dagger @ a_cent)
            
            # --- 正準角 sigma_2 ---
            # sigma_2 = | (J_N t)^T hat_t | / ( ||J_N t|| ||hat_t|| )
            
            def compute_kernel_sigma2(t_cent, t_hat):
                numer = np.abs(np.dot(t_cent, t_hat))
                denom = np.linalg.norm(t_cent) * np.linalg.norm(t_hat)
                if denom > tol:
                    return numer / denom
                return 0.0
            
            metrics['sigma_2_Y'] = compute_kernel_sigma2(y_cent, hat_y)
            metrics['sigma_2_A'] = compute_kernel_sigma2(a_cent, hat_a)
            
            # --- 特徴間のアラインメント ---
            # cos gamma_2 = | hat_y^T hat_a | / ( ||hat_y|| ||hat_a|| )
            
            norm_hat_y = np.linalg.norm(hat_y)
            norm_hat_a = np.linalg.norm(hat_a)
            
            if norm_hat_y > tol and norm_hat_a > tol:
                cos_gamma = np.abs(np.dot(hat_y, hat_a)) / (norm_hat_y * norm_hat_a)
                metrics['alignment_cos_gamma_2'] = cos_gamma
            else:
                metrics['alignment_cos_gamma_2'] = 0.0
                
        except Exception as e:
            print(f"  [Error] Geometric analysis (N <= m) failed: {e}")

    return metrics

def compute_risk_decomposition_metrics(Z, y_discrete, a_discrete, w_star):
    """
    リスクの分散の幾何学的分解に関する指標 C_Y, C_A, C_YA を計算する．
    
    C_Y  = w*^T Delta_Y w*
    C_A  = w*^T Delta_A w* - 2 w*^T v_YA
    C_YA = w*^T Delta_YA w* - 2 w*^T v_A
    
    Args:
        Z (np.ndarray): 均衡化された特徴行列 (N, m).
        y_discrete (np.ndarray): クラスラベル (N,).
        a_discrete (np.ndarray): スプリアス属性 (N,).
        w_star (np.ndarray): DFRで学習された重みベクトル (m,) or (1, m).
        
    Returns:
        metrics (dict): 計算された指標を含む辞書．
    """
    N, m = Z.shape
    
    # 重みベクトルを (m,) に整形
    w = w_star.reshape(-1)
    
    # ターゲット変数 (float)
    y = y_discrete.astype(float)
    a = a_discrete.astype(float)
    ya = y * a
    
    # 特徴量の中心化 (均衡化分布上の平均を引く)
    # h_tilde = h - E[h]
    mu = np.mean(Z, axis=0)
    Z_tilde = Z - mu
    
    # 特徴量の射影 (w*^T h_tilde)
    # Shape: (N,)
    proj = Z_tilde @ w
    
    # 予測値の二乗項 (proj^2)
    proj_sq = proj ** 2
    
    # --- 1. C_Y = w*^T Delta_Y w* ---
    # Delta_Y = E[ h_tilde h_tilde^T Y ]
    # w*^T Delta_Y w* = E[ (w*^T h_tilde)^2 Y ]
    C_Y = np.mean(proj_sq * y)
    
    # --- 2. C_A = w*^T Delta_A w* - 2 w*^T v_YA ---
    # Term 1: w*^T Delta_A w* = E[ (w*^T h_tilde)^2 A ]
    term1_A = np.mean(proj_sq * a)
    
    # Term 2: 2 w*^T v_YA
    # v_YA = E[ h_tilde (YA) ]
    # w*^T v_YA = E[ (w*^T h_tilde) * (YA) ]
    term2_A = 2 * np.mean(proj * ya)
    
    C_A = term1_A - term2_A
    
    # --- 3. C_YA = w*^T Delta_YA w* - 2 w*^T v_A ---
    # Term 1: w*^T Delta_YA w* = E[ (w*^T h_tilde)^2 YA ]
    term1_YA = np.mean(proj_sq * ya)
    
    # Term 2: 2 w*^T v_A
    # v_A = E[ h_tilde A ]
    # w*^T v_A = E[ (w*^T h_tilde) * A ]
    term2_YA = 2 * np.mean(proj * a)
    
    C_YA = term1_YA - term2_YA
    
    return {
        'C_Y': C_Y,
        'C_A': C_A,
        'C_YA': C_YA
    }

def compute_nu_risk(Z, y_discrete, a_discrete, w_star, b_star, reg_lambda=0.0):
    """
    双対関数のヘッセ行列のスペクトル定数 nu(V_risk) を計算する．
    
    nu(V_risk) := min_{u in T_Delta, ||u||=1} u^T (J^T H^{-1} J) u
    
    where:
      H = 2 * (M + lambda * I)  (Primal Hessian, size (m+1)x(m+1))
      J = [grad_R_g1, ..., grad_R_g4] (Jacobian of group risks, size (m+1)x4)
      M = E_{Q_bal}[x_aug x_aug^T] (Feature correlation matrix)
      grad_R_g = nabla_{(w,b)} R_g(w*, b*)
    
    Args:
        Z (np.ndarray): 均衡化された特徴行列 (N, m).
        y_discrete (np.ndarray): クラスラベル (N,).
        a_discrete (np.ndarray): スプリアス属性 (N,).
        w_star (np.ndarray): 学習された重み (1, m) or (m,).
        b_star (np.ndarray): 学習されたバイアス (1,) or scalar.
        reg_lambda (float): Ridge正則化係数 (alpha).
        
    Returns:
        nu_val (float): 計算された最小固有値．
    """
    N, m = Z.shape
    
    # 1. 拡張特徴行列 X_aug (N, m+1) [features, 1]
    X_aug = np.hstack([Z, np.ones((N, 1))])
    
    # パラメータベクトル theta* = [w*, b*]
    w_vec = w_star.flatten()
    b_val = float(b_star)
    theta_star = np.concatenate([w_vec, [b_val]]) # (m+1,)
    
    # ターゲットベクトル (float)
    y_float = y_discrete.astype(float)
    a_float = a_discrete.astype(float)
    
    # 2. Primal Hessian H = 2 * (M + lambda * I_reg)
    # M = (1/N) X_aug^T X_aug
    M = (X_aug.T @ X_aug) / N
    
    # 正則化項行列
    Reg = np.eye(m + 1) * reg_lambda
    Reg[m, m] = 0.0 # Bias term regularization = 0
    
    H = 2 * (M + Reg)
    
    # Hの逆行列 H_inv
    # 特異に近い場合は微小値を加算
    try:
        H_inv = scipy.linalg.inv(H)
    except scipy.linalg.LinAlgError:
        H_inv = scipy.linalg.inv(H + 1e-6 * np.eye(m + 1))
        
    # 3. 各グループのリスク勾配 J (m+1, 4)
    # R_g(theta) = (1/N_g) sum_{i in g} (y_i - x_aug_i^T theta)^2
    # nabla R_g = -(2/N_g) sum_{i in g} (y_i - f(x_i)) * x_aug_i
    
    groups = sorted(list(set(zip(y_float, a_float))))
    # 想定グループ順序: (-1,-1), (-1,1), (1,-1), (1,1)
    expected_groups = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
    
    grads = []
    
    # 残差ベクトル r = y - X_aug theta
    preds = X_aug @ theta_star
    residuals = y_float - preds
    
    for g_key in expected_groups:
        mask = (y_float == g_key[0]) & (a_float == g_key[1])
        n_g = np.sum(mask)
        
        if n_g > 0:
            # -(2/N_g) * X_g^T * residuals_g
            X_g = X_aug[mask]
            res_g = residuals[mask]
            
            grad_g = -2.0 * (X_g.T @ res_g) / n_g
            grads.append(grad_g)
        else:
            # グループが存在しない場合は0ベクトル (またはNaN扱いだが，計算のため0とする)
            grads.append(np.zeros(m + 1))
            
    J = np.stack(grads, axis=1) # (m+1, 4)
    
    # 4. 双対ヘッセ行列 A_dual = J^T H^{-1} J  (4, 4)
    A_dual = J.T @ H_inv @ J
    
    # 5. 接空間 T_Delta 上での最小固有値
    # T_Delta = {u | sum(u) = 0}
    # 射影行列 P = I - (1/k) 1 1^T  (k=4)
    k = 4
    ones = np.ones((k, 1))
    P = np.eye(k) - (ones @ ones.T) / k
    
    # 射影された行列 P A_dual P の固有値を計算
    # subspace sum(u)=0 に対応する固有値を取り出す
    # P A_dual P はランク k-1 以下 (1^T P = 0 なので 1 は固有値 0 の固有ベクトル)
    # 0 以外の固有値のうち最小のものが求める nu
    
    PA_P = P @ A_dual @ P
    eigvals = np.linalg.eigvalsh(PA_P)
    
    # 固有値は昇順．最初の固有値は (数値誤差の範囲で) 0 になるはず (方向 1)
    # 0より大きい固有値の中で最小のものを探す
    
    # 数値誤差を考慮して判定
    nonzero_eigs = eigvals[eigvals > 1e-8]
    
    if len(nonzero_eigs) > 0:
        nu_val = np.min(nonzero_eigs)
    else:
        nu_val = 0.0
        
    return nu_val

def analyze_spectral_structure(Z, y_discrete, a_discrete):
    """
    特徴共分散行列 Sigma_perp のスペクトル構造と Mahalanobis 距離の分布を解析する．
    
    Args:
        Z (np.ndarray): 均衡化された特徴行列 (N, m).
        y_discrete (np.ndarray): クラスラベル (N,).
        a_discrete (np.ndarray): スプリアス属性 (N,).
        
    Returns:
        results (dict): 計算されたスペクトル指標を含む辞書．
    """
    N, m = Z.shape
    tol = 1e-10

    # ターゲットベクトル (float)
    y_vec = y_discrete.astype(float)
    a_vec = a_discrete.astype(float)

    # 1. 統計量の計算 (均衡分布 Q_bal 上の期待値に対応する標本統計量)
    # 中心化
    Z_centered = Z - np.mean(Z, axis=0)
    y_centered = y_vec - np.mean(y_vec)
    a_centered = a_vec - np.mean(a_vec)

    # 全共分散行列 Sigma_total (m, m)
    Sigma_total = (Z_centered.T @ Z_centered) / N
    
    # 信号ベクトル v_Y, v_A (m,)
    v_Y = (Z_centered.T @ y_centered) / N
    v_A = (Z_centered.T @ a_centered) / N

    # 2. Sigma_perp の構成
    # Sigma_perp = Sigma_total - v_Y v_Y^T - v_A v_A^T
    Sigma_perp = Sigma_total - np.outer(v_Y, v_Y) - np.outer(v_A, v_A)

    # 3. 固有値分解 (対称行列を前提)
    # eigenvalues: sigma_1 >= sigma_2 >= ... >= sigma_m
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma_perp)
    # 降順にソート
    idx = eigenvalues.argsort()[::-1]
    sigma = eigenvalues[idx]
    U = eigenvectors[:, idx]

    # 数値安定性のために微小な正の値を保証
    sigma_clipped = np.maximum(sigma, tol)

    # 4. 指標の計算
    # Stable Rank: sum(sigma) / sigma_max
    stable_rank = np.sum(sigma_clipped) / sigma_clipped[0]

    # Mahalanobis 距離の成分分解
    # y_i = <v_Y, u_i>, a_i = <v_A, u_i>
    y_proj = U.T @ v_Y
    a_proj = U.T @ v_A

    # d_Y^2 = sum (y_i^2 / sigma_i)
    d_Y_sq_components = (y_proj**2) / sigma_clipped
    d_A_sq_components = (a_proj**2) / sigma_clipped
    
    d_Y_sq = np.sum(d_Y_sq_components)
    d_A_sq = np.sum(d_A_sq_components)

    # Mahalanobis 空間における alignment (cos phi)
    # cos_phi = (sum y_i a_i / sigma_i) / (d_Y * d_A)
    numerator_cos_phi = np.sum((y_proj * a_proj) / sigma_clipped)
    cos_phi = numerator_cos_phi / (np.sqrt(d_Y_sq) * np.sqrt(d_A_sq) + tol)
    sin_phi_sq = 1 - cos_phi**2

    # 5. 信号のスペクトル分布 (SNRの集中度)
    # 固有値を 3 つの領域 (Top 10%, Middle 80%, Bottom 10%) に分割して SNR の寄与を計算
    m_10 = max(1, m // 10)
    
    def get_distribution(components):
        total = np.sum(components) + tol
        top = np.sum(components[:m_10]) / total
        bottom = np.sum(components[-m_10:]) / total
        return top, bottom

    y_snr_top, y_snr_bottom = get_distribution(d_Y_sq_components)
    a_snr_top, a_snr_bottom = get_distribution(d_A_sq_components)

    results = {
        'stable_rank': stable_rank,
        'd_Y_sq': d_Y_sq,
        'd_A_sq': d_A_sq,
        'cos_phi': cos_phi,
        'sin_phi_sq': sin_phi_sq,
        'y_snr_distribution': (y_snr_top, y_snr_bottom),
        'a_snr_distribution': (a_snr_top, a_snr_bottom),
        'sigma_min_max_ratio': sigma_clipped[-1] / sigma_clipped[0]
    }

    return results

def run_dfr_procedure(config, model, X_train, y_train, a_train, X_test, y_test, a_test, device, loss_function_name, X_val=None, y_val=None, a_val=None):
    """
    DFRの実行プロセス全体
    X_val, y_val, a_val は main.py で事前に分割された Held-out Validation データが渡される
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

    # Validationデータの準備 (DFRの学習ソース)
    if X_val is None:
        raise ValueError("DFR validation data (X_val) is None. Check if use_dfr is enabled and splitting logic in main.py is correct.")

    print("Extracting embeddings for Validation data...")
    val_embeddings = get_embeddings(model, X_val, target_layer_name, device=device)

    X_dfr_source = val_embeddings
    y_dfr_source = y_val.cpu().numpy()
    a_dfr_source = a_val.cpu().numpy()

    # StandardScalerの設定 (Configに基づく)
    use_standardization = config.get('dfr_standardization', False)
    std_source = config.get('dfr_standardization_source', 'train')

    scaler = StandardScaler()
    
    if use_standardization:
        if std_source == 'train':
            print("Fitting StandardScaler on TRAIN data...")
            scaler.fit(train_embeddings)
        elif std_source == 'validation':
            print("Fitting StandardScaler on VALIDATION data...")
            scaler.fit(val_embeddings)
        else:
             raise ValueError(f"Invalid dfr_standardization_source: {std_source}. Choose 'train' or 'validation'.")
    else:
        print("StandardScaler is disabled (using Identity transform).")
        # ダミーの属性を手動で設定 (平均0, 分散1 => 恒等変換)
        scaler.mean_ = np.zeros(train_embeddings.shape[1])
        scaler.scale_ = np.ones(train_embeddings.shape[1])
        scaler.var_ = np.ones(train_embeddings.shape[1])

    # --- 2. DFR Tune & Train (Main Task: Predict Y) ---
    print("\n--- DFR Main Task (Target: Y) ---")
    
    # Phase 1: Hyperparameter Search (Standard DFR Logic)
    # Validationを分割してCを決める
    best_c = dfr_tune(X_dfr_source, y_dfr_source, a_dfr_source, scaler, config)
    
    # Phase 2: Final Optimization
    dfr_method = config.get('dfr_method', 'standard')
    
    if dfr_method == 'minimax':
        # Minimax DFR
        print("  [DFR Method] Selected: Minimax Optimization")
        avg_coef, avg_intercept = dfr_train_minimax(X_dfr_source, y_dfr_source, a_dfr_source, best_c, scaler, config)
    else:
        # Standard DFR (Retrain on Balanced Subsets)
        print("  [DFR Method] Selected: Standard (Balanced Subsampling)")
        avg_coef, avg_intercept = dfr_train(X_dfr_source, y_dfr_source, a_dfr_source, best_c, scaler, config)
    
    # --- 3. DFR PyTorchモデル化 & 評価 (Main Task) ---
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

    # モデル出力の統計量
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


    # --- Spurious Attribute Prediction Task (Target: A) ---
    print("\n" + "-"*40)
    print(" Spurious Attribute Prediction (Target: A) DFR")
    print("-" * 40)
    
    # DFRのValidationソースにおけるターゲット(Y)と属性(A)を入れ替える
    # Target = a_dfr_source, Attribute for grouping = y_dfr_source
    y_dfr_spur_source = a_dfr_source
    a_dfr_spur_source = y_dfr_source
    
    # Tune C for spurious task (using same config options)
    print("  [Spurious DFR] Tuning C (predicting attribute A)...")
    best_c_spur = dfr_tune(X_dfr_source, y_dfr_spur_source, a_dfr_spur_source, scaler, config)
    
    # Train for spurious task
    print(f"  [Spurious DFR] Training with Best C: {best_c_spur}...")
    
    # Spurious Attribute Predictionは比較用のため，minimax設定でもStandard DFRを使用
    print("  [Spurious DFR] Using Standard DFR (Balanced Subsampling) for comparison regardless of dfr_method.")
    avg_coef_spur, avg_intercept_spur = dfr_train(X_dfr_source, y_dfr_spur_source, a_dfr_spur_source, best_c_spur, scaler, config)
    
    # Create Spurious DFR Model
    dfr_spur_model = DFRTorchModel(avg_coef_spur, avg_intercept_spur, scaler).to(device)
    
    print("\n--- Spurious DFR Evaluation (Predicting A) ---")
    
    dfr_spur_train_metrics = utils.evaluate_model(
        dfr_spur_model, X_train_emb_tensor, a_train, y_train, 
        device, loss_function_name, eval_bs
    )
    dfr_spur_train_metrics = _reorder_metrics_for_spurious_task(dfr_spur_train_metrics)

    dfr_spur_test_metrics = utils.evaluate_model(
        dfr_spur_model, X_test_emb_tensor, a_test, y_test, 
        device, loss_function_name, eval_bs
    )
    dfr_spur_test_metrics = _reorder_metrics_for_spurious_task(dfr_spur_test_metrics)
    
    print(f"Spurious DFR Train | Loss: {dfr_spur_train_metrics['avg_loss']:.4f}, Worst Acc: {dfr_spur_train_metrics['worst_acc']:.4f}")
    print(f"Spurious DFR Test  | Loss: {dfr_spur_test_metrics['avg_loss']:.4f}, Worst Acc: {dfr_spur_test_metrics['worst_acc']:.4f}")

    print("Spurious DFR Test Group Details (Aligned to Standard Groups Y, A):")
    for i, loss in enumerate(dfr_spur_test_metrics['group_losses']):
        acc = dfr_spur_test_metrics['group_accs'][i]
        print(f"  Group {i}: Loss={loss:.4f}, Acc={acc:.4f}")


    # --- Principal Angles Analysis ---
    print("\n" + "-"*40)
    print(" Analysis: Principal Angles (Geometric Analysis)")
    print("-" * 40)
    
    # バランスされたテストデータセットの作成
    # テストデータ (test_embeddings, y_test, a_test) から，グループサイズを均衡化したデータセットを作成する
    print(" [Analysis] Creating Balanced Test Subset for Geometric Analysis...")
    y_test_np = y_test.cpu().numpy()
    a_test_np = a_test.cpu().numpy()
    
    # balance_indicesを用いてインデックスを取得
    bal_test_idx = balance_indices(y_test_np, a_test_np, random_state=12345)
    
    X_test_bal = test_embeddings[bal_test_idx]
    y_test_bal = y_test_np[bal_test_idx]
    a_test_bal = a_test_np[bal_test_idx]
    
    print(f"  Balanced Test Subset Size: {len(X_test_bal)} (from {len(test_embeddings)})")
    
    # 指標の計算 (幾何学量)
    print(" [Analysis] Computing Geometric Metrics on Balanced Test Set...")
    geo_metrics = compute_geometric_metrics(X_test_bal, y_test_bal, a_test_bal)
    
    sigma_2_Y = geo_metrics['sigma_2_Y']
    sigma_2_A = geo_metrics['sigma_2_A']
    cos_gamma_2 = geo_metrics['alignment_cos_gamma_2']

    print(f"  Result (Balanced Test Set):")
    print(f"    sigma_2(V_Y) (Explanation of Y): {sigma_2_Y:.6f}")
    print(f"    sigma_2(V_A) (Explanation of A): {sigma_2_A:.6f}")
    print(f"    cos gamma_2 (Feature Alignment Y vs A): {cos_gamma_2:.6f}")
    
    # --- Risk Decomposition Metrics Calculation ---
    print(" [Analysis] Computing Risk Decomposition Metrics (C_Y, C_A, C_YA)...")
    risk_decomp = compute_risk_decomposition_metrics(
        X_test_bal, y_test_bal, a_test_bal, avg_coef
    )
    
    print(f"    C_Y  (Class Covariance Diff): {risk_decomp['C_Y']:.6f}")
    print(f"    C_A  (Attr Covariance Diff): {risk_decomp['C_A']:.6f}")
    print(f"    C_YA (Interaction Covariance Diff): {risk_decomp['C_YA']:.6f}")

    # --- Nu Risk Calculation (Dual Hessian Spectrum) ---
    print(" [Analysis] Computing nu(V_risk) (Dual Hessian Spectrum)...")
    # Ridgeパラメータ (alpha = 1/best_c)
    reg_alpha = 1.0 / (best_c + 1e-9)
    nu_val = compute_nu_risk(
        X_test_bal, y_test_bal, a_test_bal, avg_coef, avg_intercept, reg_lambda=reg_alpha
    )
    print(f"    nu(V_risk) (Min Eigenvalue of Dual Hessian on Simplex Tangent Space): {nu_val:.6e}")

    # --- Spectral Analysis ---
    print("\n" + "-"*40)
    print(" Analysis: Spectral Structure of Sigma_perp")
    print("-" * 40)
    
    spec_metrics = analyze_spectral_structure(X_test_bal, y_test_bal, a_test_bal)
    
    print(f"  Stable Rank (Effective Dimension): {spec_metrics['stable_rank']:.4f}")
    print(f"  Condition Number (sigma_min / sigma_max): {spec_metrics['sigma_min_max_ratio']:.6e}")
    print(f"  Mahalanobis Distance d_Y^2 (SNR_Y): {spec_metrics['d_Y_sq']:.4f}")
    print(f"  Mahalanobis Distance d_A^2 (SNR_A): {spec_metrics['d_A_sq']:.4f}")
    print(f"  Mahalanobis Alignment (cos phi): {spec_metrics['cos_phi']:.6f}")
    print(f"  Mahalanobis Orthogonality (sin^2 phi): {spec_metrics['sin_phi_sq']:.6f}")
    
    y_top, y_bot = spec_metrics['y_snr_distribution']
    a_top, a_bot = spec_metrics['a_snr_distribution']
    print(f"  SNR Distribution (Top 10% vs Bottom 10% of Spectrum):")
    print(f"    Target Y: Top={y_top:.2%}, Bottom={y_bot:.2%}")
    print(f"    Attr A:   Top={a_top:.2%}, Bottom={a_bot:.2%}")

    # 理論的な下限値の計算
    d_Y_sq_val = spec_metrics['d_Y_sq']
    d_A_sq_val = spec_metrics['d_A_sq']
    sin_phi_sq_val = spec_metrics['sin_phi_sq']
    lower_bound = (1 + d_A_sq_val) / (1 + d_Y_sq_val + d_A_sq_val + d_Y_sq_val * d_A_sq_val * sin_phi_sq_val)
    print(f"  Theoretical Lower Bound of Worst-Group Risk: {lower_bound:.6f}")


    # --- 4. Baseline Regressions (ERM on Features) ---
    print("\n--- Baseline Regressions on Training Features (Imbalanced) ---")
    
    X_train_np = train_embeddings
    y_train_np = y_train.cpu().numpy()
    
    # Scale X_train for training
    # scaler.transformを使用 (scalerがidentityならそのまま，fitting済みなら正規化される)
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

    return dfr_train_metrics, dfr_test_metrics, baseline_results, dfr_spur_train_metrics, dfr_spur_test_metrics, \
           sigma_2_Y, sigma_2_A, cos_gamma_2
