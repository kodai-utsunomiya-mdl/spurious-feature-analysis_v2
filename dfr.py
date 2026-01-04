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
    [原論文の実装に準拠]
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
    
    特徴量行列 Z (N, m) とターゲット y, a を受け取り，
    サンプルサイズ N と特徴次元 m の大小関係に応じて計算手法を選択する．
    
    定義:
      J_N = I - (1/N) 1 1^T (中心化行列)
      Phi = (Z, 1) (定数項を追加した特徴行列)
      
    1. 正準角の余弦 (sigma_2)
       - 特徴空間 V_Phi とターゲット空間 V_T (T=Y or A) の間の第2正準角の余弦 (変動成分の説明率)．
       - Case N > m: Phiの直交基底 U_Phi と Tの直交基底 U_T を用いたSVD．
       - Case N <= m: カーネル K = J_N Phi Phi^T J_N と一般化逆行列 K^dagger を用いた予測ベクトル hat_t と J_N t のコサイン類似度．
       
    2. 特徴間のアラインメント (cos gamma_2)
       - 特徴空間内における，Yの予測成分とAの予測成分の間のコサイン類似度．
       - Case N > m: 射影行列 P_Phi = U_Phi U_Phi^T を用いて射影・中心化したベクトルのコサイン類似度．
       - Case N <= m: カーネル法による予測ベクトル hat_y と hat_a のコサイン類似度．
       
    Args:
        Z (np.ndarray): 特徴行列 (N, m)．
        y_discrete (np.ndarray): クラスラベル (N,).
        a_discrete (np.ndarray): スプリアス属性 (N,).
        
    Returns:
        metrics (dict): 計算した指標を含む辞書．
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

    # StandardScaler は Trainデータ で Fit させる (原論文の実装に準拠)
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

    # --- 2. DFR Tune & Train (Main Task: Predict Y) ---
    print("\n--- DFR Main Task (Target: Y) ---")
    # Validationを分割してCを決める
    best_c = dfr_tune(X_dfr_source, y_dfr_source, a_dfr_source, scaler, config)
    # Validation全体を使って再学習
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
    
    # スケーリング (Trainで学習したScalerを適用)
    # print(" [Analysis] Scaling Balanced Test Subset (using Train scaler)...")
    # X_test_bal_scaled = scaler.transform(X_test_bal)
    
    # 指標の計算
    print(" [Analysis] Computing Geometric Metrics on Balanced Test Set...")
    geo_metrics = compute_geometric_metrics(X_test_bal, y_test_bal, a_test_bal)
    
    sigma_2_Y = geo_metrics['sigma_2_Y']
    sigma_2_A = geo_metrics['sigma_2_A']
    cos_gamma_2 = geo_metrics['alignment_cos_gamma_2']

    print(f"  Result (Balanced Test Set):")
    print(f"    sigma_2(V_Y) (Explanation of Y): {sigma_2_Y:.6f}")
    print(f"    sigma_2(V_A) (Explanation of A): {sigma_2_A:.6f}")
    print(f"    cos gamma_2 (Feature Alignment Y vs A): {cos_gamma_2:.6f}")

    # --- 4. Baseline Regressions (ERM on Features) ---
    print("\n--- Baseline Regressions on Training Features (Imbalanced) ---")
    
    X_train_np = train_embeddings
    y_train_np = y_train.cpu().numpy()
    
    # Scale X_train for training
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
