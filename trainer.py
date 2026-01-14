# sp/trainer.py

import torch
import numpy as np
from utils import get_loss_function

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''
    Mixup data augmentation
    x: input tensor
    y: target tensor
    alpha: mixup parameter (Beta distribution)
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(scores, y_a, y_b, lam, loss_function_name):
    '''
    Mixup loss calculation
    '''
    loss_a = get_loss_function(scores, y_a, loss_function_name)
    loss_b = get_loss_function(scores, y_b, loss_function_name)
    return lam * loss_a + (1 - lam) * loss_b


def compute_regularization_loss(model, X_train, y_train, a_train, group_keys, num_samples, device, kernel_reg_weights=None, cosine_reg_weights=None):
    """
    指定された条件に基づいて正則化項 (カーネル値またはコサイン類似度) を計算する．
    Args:
        model: 学習モデル
        X_train, y_train, a_train: 全学習データ
        group_keys: グループのリスト [(y, a), ...]
        num_samples: ヤコビアン計算に使用するサンプル数 (jacobian_num_samples)
        device: デバイス
        kernel_reg_weights: カーネル正則化の各条件の重みを格納した辞書 (Noneの場合は計算しない)
        cosine_reg_weights: コサイン類似度正則化の各条件の重みを格納した辞書 (Noneの場合は計算しない)
    Returns:
        total_reg_loss: 正則化損失の合計 (scalar tensor)
    """
    # 1. 各グループの勾配の平均を計算
    # m_g = E[nabla f(x)]
    group_mean_grads = {}
    
    for g in group_keys:
        y_val, a_val = g
        mask = (y_train == y_val) & (a_train == a_val)
        indices = torch.where(mask)[0]
        
        if len(indices) == 0:
            continue
            
        # サンプリング (jacobian_num_samples)
        if len(indices) > num_samples:
            # ランダムサンプリング
            perm = torch.randperm(len(indices))
            selected_indices = indices[perm[:num_samples]]
        else:
            selected_indices = indices
            
        X_sub = X_train[selected_indices].to(device)
        
        # モデル出力の平均
        scores, _ = model(X_sub)
        mean_output = scores.mean()
        
        # 出力の平均に対するパラメータ勾配を取得
        # create_graph=True にして，この勾配自体をさらに微分可能にする (正則化のため)
        grads = torch.autograd.grad(mean_output, model.parameters(), create_graph=True)
        
        # フラット化して保存
        flat_grad = torch.cat([g.view(-1) for g in grads])
        group_mean_grads[g] = flat_grad

    # 2. 正則化項の計算
    total_reg_loss = torch.tensor(0.0, device=device)
    epsilon = 1e-8 # ゼロ除算防止用
    
    # 全ペアについて計算
    existing_keys = [k for k in group_keys if k in group_mean_grads]
    
    for i in range(len(existing_keys)):
        for j in range(i + 1, len(existing_keys)):
            g_u = existing_keys[i]
            g_v = existing_keys[j]
            
            y_u, a_u = g_u
            y_v, a_v = g_v
            
            vec_u = group_mean_grads[g_u]
            vec_v = group_mean_grads[g_v]
            
            # カーネル値 (内積)
            dot_val = torch.dot(vec_u, vec_v)
            
            # --- カーネルの正則化 ---
            if kernel_reg_weights is not None:
                # 条件1: 属性が同じでラベルが異なる (Minimize)
                if (y_u != y_v) and (a_u == a_v):
                    total_reg_loss = total_reg_loss + (kernel_reg_weights.get('sameA_diffY', 0.0) * dot_val)
                # 条件2: 属性が異なりラベルも異なる (Minimize)
                elif (y_u != y_v) and (a_u != a_v):
                    total_reg_loss = total_reg_loss + (kernel_reg_weights.get('diffA_diffY', 0.0) * dot_val)
                # 条件3: 属性が異なりラベルが同じ (Maximize)
                elif (y_u == y_v) and (a_u != a_v):
                    total_reg_loss = total_reg_loss - (kernel_reg_weights.get('diffA_sameY', 0.0) * dot_val)

            # --- コサイン類似度の正則化 ---
            if cosine_reg_weights is not None:
                # ノルム計算
                norm_u = torch.norm(vec_u)
                norm_v = torch.norm(vec_v)
                
                # コサイン類似度
                cosine_val = dot_val / (norm_u * norm_v + epsilon)
                
                # 条件1: 属性が同じでラベルが異なる (Minimize -> 逆向き)
                if (y_u != y_v) and (a_u == a_v):
                    total_reg_loss = total_reg_loss + (cosine_reg_weights.get('sameA_diffY', 0.0) * cosine_val)
                # 条件2: 属性が異なりラベルも異なる (Minimize -> 逆向き)
                elif (y_u != y_v) and (a_u != a_v):
                    total_reg_loss = total_reg_loss + (cosine_reg_weights.get('diffA_diffY', 0.0) * cosine_val)
                # 条件3: 属性が異なりラベルが同じ (Maximize -> 同じ向き)
                elif (y_u == y_v) and (a_u != a_v):
                    total_reg_loss = total_reg_loss - (cosine_reg_weights.get('diffA_sameY', 0.0) * cosine_val)
                
    return total_reg_loss

def compute_decov_loss(features, target_identity=False):
    """
    DeCov Loss: 特徴量の共分散行列の非対角成分の2乗和を計算する
    
    target_identity=False (Default):
      L_DeCov = 0.5 * (||C||_F^2 - ||diag(C)||_2^2)
      (非対角成分のみを0に近づける)
      
    target_identity=True:
      L_DeCov = 0.5 * ||C - I||_F^2
      (共分散行列全体を単位行列に近づける = 非対角成分を0，対角成分を1に近づける)

    Args:
        features: 特徴量テンソル (Batch, Dim)
        target_identity: 対角成分も1に近づけるかどうか
    Returns:
        loss: DeCov損失 (scalar)
    """
    # バッチサイズ N, 特徴次元 d
    N, d = features.shape
    
    # バッチサイズが1以下の場合は計算不可
    if N <= 1:
        return torch.tensor(0.0, device=features.device)
    
    # 平均を引く (Centering)
    features_centered = features - features.mean(dim=0, keepdim=True)
    
    # 共分散行列 C = (1 / (N - 1)) * X^T X
    cov = (1.0 / (N - 1)) * torch.matmul(features_centered.T, features_centered)

    if target_identity:
        # 共分散行列全体を単位行列に近づける
        # L = 0.5 * || C - I ||_F^2 = 0.5 * sum((C - I)^2)
        I = torch.eye(d, device=features.device)
        loss = 0.5 * torch.sum((cov - I) ** 2)
    else:
        # 非対角成分のノルムの2乗: ||C||_F^2 - ||diag(C)||_2^2
        cov_fro_sq = torch.sum(cov**2)
        cov_diag_sq = torch.sum(torch.diag(cov)**2)
        
        # Loss definition from "Reducing Overfitting in Deep Networks by Decorrelating Representations"
        loss = 0.5 * (cov_fro_sq - cov_diag_sq)
    
    return loss

def get_features_from_output(outputs, target_layer_name):
    """
    モデルの出力辞書から指定された層の特徴量を取り出すヘルパー関数
    """
    if target_layer_name in outputs:
        return outputs[target_layer_name]
    elif target_layer_name == "last_hidden":
        # 'layer_X' の形式のキーを探し，最大のインデックスを持つものを返す
        keys = [k for k in outputs.keys() if k.startswith('layer_')]
        if not keys:
            return None
        # インデックスでソート
        max_idx = -1
        target_key = None
        for k in keys:
            try:
                idx = int(k.split('_')[1])
                if idx > max_idx:
                    max_idx = idx
                    target_key = k
            except:
                continue
        if target_key:
            return outputs[target_key]
    return None


def train_epoch(
    config, 
    model, 
    optimizer, 
    debias_method, 
    X_train, 
    y_train, 
    a_train, 
    train_loader, 
    group_keys, 
    static_weights, 
    dro_q_weights, 
    device, 
    loss_function_name, 
    epoch
):
    """
    1エポック分の学習を実行する．
    debias_method に応じて学習ステップを切り替える．
    GroupDROの場合，更新された dro_q_weights を返す．
    """
    model.train()

    # --- Mixup設定の読み込み ---
    use_mixup = config.get('use_mixup', False)
    mixup_alpha = config.get('mixup_alpha', 0.4)

    # --- 正則化スケジューリング (減衰係数の計算) ---
    # config.yaml から設定を読み込む
    reg_end_epoch = config.get('regularization_end_epoch', None)
    reg_decay_start_epoch = config.get('regularization_decay_start_epoch', None)
    
    # デフォルトは強度 1.0 (減衰なし)
    reg_weight_factor = 1.0
    
    if reg_end_epoch is not None:
        if epoch >= reg_end_epoch:
            # 終了エポックを超えたら無効化
            reg_weight_factor = 0.0
        elif reg_decay_start_epoch is not None and epoch >= reg_decay_start_epoch:
            # 減衰期間中: Linear Decay (1.0 -> 0.0)
            # progress: 0.0 (start) -> 1.0 (end)
            total_decay_steps = reg_end_epoch - reg_decay_start_epoch
            if total_decay_steps > 0:
                progress = (epoch - reg_decay_start_epoch) / float(total_decay_steps)
                reg_weight_factor = 1.0 - progress

                if (epoch + 1) % 10 == 0 or epoch == reg_decay_start_epoch:
                    print(f"  [Epoch {epoch+1}] Regularization decaying... factor: {reg_weight_factor:.4f}")
            else:
                reg_weight_factor = 0.0

    # --- 正則化の設定の読み込みと重みの適用 ---
    # 係数が 0 なら正則化の計算自体をスキップするためにフラグを落とす
    should_apply_grad_reg = False
    kernel_reg_weights = None
    cosine_reg_weights = None
    
    # DeCovの設定
    use_decov = config.get('use_decov_regularization', False)
    decov_weight = config.get('decov_reg_weight', 0.1)
    # 対角成分も単位行列に近づけるかどうかのフラグ
    decov_target_identity = config.get('decov_target_identity', False)
    
    if reg_weight_factor > 0:
        # 正則化の有効・無効設定を読み込み
        use_kernel_reg = config.get('use_kernel_regularization', False)
        use_cosine_reg = config.get('use_cosine_regularization', False)
        
        # カーネル正則化の重み設定 (factor を乗算して減衰させる)
        if use_kernel_reg:
            kernel_reg_weights = {
                'sameA_diffY': config.get('kernel_reg_weight_sameA_diffY', 0.0) * reg_weight_factor,
                'diffA_diffY': config.get('kernel_reg_weight_diffA_diffY', 0.0) * reg_weight_factor,
                'diffA_sameY': config.get('kernel_reg_weight_diffA_sameY', 0.0) * reg_weight_factor,
            }
            # 重みがすべて0なら無効扱い
            if not any(w != 0 for w in kernel_reg_weights.values()):
                kernel_reg_weights = None

        # コサイン類似度の正則化の重み設定 (factor を乗算して減衰させる)
        if use_cosine_reg:
            cosine_reg_weights = {
                'sameA_diffY': config.get('cosine_reg_weight_sameA_diffY', 0.0) * reg_weight_factor,
                'diffA_diffY': config.get('cosine_reg_weight_diffA_diffY', 0.0) * reg_weight_factor,
                'diffA_sameY': config.get('cosine_reg_weight_diffA_sameY', 0.0) * reg_weight_factor,
            }
            # 重みがすべて0なら無効扱い
            if not any(w != 0 for w in cosine_reg_weights.values()):
                cosine_reg_weights = None

        should_apply_grad_reg = (kernel_reg_weights is not None) or (cosine_reg_weights is not None)
        
        # DeCovの重み減衰
        if use_decov:
            decov_weight *= reg_weight_factor
    else:
        use_decov = False

    # jacobian_num_samples を config から取得
    jac_num_samples = config.get('jacobian_num_samples', 100)
    
    # DFRのターゲット層 (DeCovを適用する層)
    dfr_target_layer = config.get('dfr_target_layer', 'last_hidden')

    # --- 学習ステップの分岐 ---
    if debias_method == 'IW_uniform':
        # --- IW (Uniform Target) の学習ステップ (フルバッチ・グループ別の勾配) ---
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
            scores_g, outputs_g = model(X_g)
            # utils からインポートした関数を呼び出す (reduction='mean' がデフォルト)
            loss_g = get_loss_function(scores_g, y_g, loss_function_name)
            
            # --- DeCov正則化 ---
            if use_decov:
                features_g = get_features_from_output(outputs_g, dfr_target_layer)
                if features_g is not None:
                    # [変更] target_identity オプションを渡す
                    decov_loss = compute_decov_loss(features_g, target_identity=decov_target_identity)
                    loss_g = loss_g + decov_weight * decov_loss
            
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
                        if param_idx < len(group_grads_list[g]):
                            grad_g_param = group_grads_list[g][param_idx]
                            debiased_grad += w_g * grad_g_param.to(device)

                param.grad = debiased_grad
                param_idx += 1
        
        # --- 勾配正則化の適用 ---
        if should_apply_grad_reg:
            reg_loss = compute_regularization_loss(
                model, X_train, y_train, a_train, group_keys, jac_num_samples, device, 
                kernel_reg_weights, cosine_reg_weights
            )
            # 正則化項の勾配を計算して加算
            reg_loss.backward()

        # 3. パラメータ更新
        optimizer.step()

    elif debias_method == 'GroupDRO':
        # --- Group DRO 学習ステップ (フルバッチ・グループ別の勾配) ---
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
            scores_g, outputs_g = model(X_g)
            # utils からインポートした関数を呼び出す (reduction='mean' がデフォルト)
            loss_g = get_loss_function(scores_g, y_g, loss_function_name)
            
            group_losses_tensor[i] = loss_g.detach() # 損失を保存
            
            # --- DeCov正則化 ---
            if use_decov:
                features_g = get_features_from_output(outputs_g, dfr_target_layer)
                if features_g is not None:
                    # [変更] target_identity オプションを渡す
                    decov_loss = compute_decov_loss(features_g, target_identity=decov_target_identity)
                    loss_g = loss_g + decov_weight * decov_loss
            
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
                        if param_idx < len(group_grads_list[g]):
                            grad_g_param = group_grads_list[g][param_idx]
                            debiased_grad += w_g * grad_g_param.to(device)
                
                param.grad = debiased_grad
                param_idx += 1

        # --- 勾配正則化の適用 ---
        if should_apply_grad_reg:
            reg_loss = compute_regularization_loss(
                model, X_train, y_train, a_train, group_keys, jac_num_samples, device, 
                kernel_reg_weights, cosine_reg_weights
            )
            reg_loss.backward()

        # 4. パラメータ更新
        optimizer.step()

    elif debias_method == 'None':
        # --- 通常のERMによる学習ステップ (ミニバッチ) ---
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            if use_mixup:
                # --- Mixup Training ---
                X_mixed, y_a, y_b, lam = mixup_data(X_batch, y_batch, alpha=mixup_alpha, device=device)
                scores, outputs = model(X_mixed)
                loss = mixup_criterion(scores, y_a, y_b, lam, loss_function_name)
            else:
                # --- Standard Training ---
                scores, outputs = model(X_batch)
                loss = get_loss_function(scores, y_batch, loss_function_name)
            
            # --- DeCov正則化 ---
            if use_decov:
                features = get_features_from_output(outputs, dfr_target_layer)
                if features is not None:
                    # [変更] target_identity オプションを渡す
                    decov_loss = compute_decov_loss(features, target_identity=decov_target_identity)
                    loss = loss + decov_weight * decov_loss
            
            # --- 勾配正則化の適用 ---
            if should_apply_grad_reg:
                reg_loss = compute_regularization_loss(
                    model, X_train, y_train, a_train, group_keys, jac_num_samples, device, 
                    kernel_reg_weights, cosine_reg_weights
                )
                loss = loss + reg_loss
            
            loss.backward()
            optimizer.step()
    # --- 分岐終了 ---

    # GroupDROの場合，更新された重みを返す
    if debias_method == 'GroupDRO':
        return dro_q_weights
    else:
        return None # 他のメソッドではNoneを返す
