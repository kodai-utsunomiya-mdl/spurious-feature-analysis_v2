# sp/analysis.py

import numpy as np
from itertools import combinations, combinations_with_replacement
import torch
import torch.nn.functional as F

# ==============================================================================
# ヤコビアン計算のヘルパー関数
# ==============================================================================
def get_model_jacobian(model, X_subset, device):
    """モデルのヤコビアンの期待値を計算"""
    model.eval()
    jacobians = []
    X_subset = X_subset.to(device)
    
    for i in range(len(X_subset)):
        x_i = X_subset[i:i+1]
        x_i.requires_grad_(True)
        
        y_pred, _ = model(x_i)
        
        # モデルの全パラメータに対する勾配を計算
        grad_params = torch.autograd.grad(y_pred, model.parameters(), create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_params])
        jacobians.append(flat_grad.cpu().detach().numpy())
        
    return np.mean(jacobians, axis=0)



# ==============================================================================
# 勾配グラム行列の分析
# ==============================================================================
def analyze_gradient_gram_matrix(model, X_data, y_data, a_data, device, loss_function, dataset_type, optimizer_params):
    """
    グループ間の勾配の内積からなるグラム行列を計算（学習率を考慮）
    """
    print(f"\nAnalyzing GRADIENT GRAM MATRIX on {dataset_type} data...")
    group_keys = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    group_grads_list = {}

    for y_val, a_val in group_keys:
        mask = (y_data == y_val) & (a_data == a_val)
        if mask.sum() == 0:
            group_grads_list[(y_val, a_val)] = None
            continue
        
        X_group, y_group = X_data[mask], y_data[mask]
        
        model.zero_grad()
        scores, _ = model(X_group.to(device))
        
        if loss_function == 'logistic':
            loss = torch.nn.functional.softplus(-y_group.to(device) * scores).mean()
        else: # mse
            loss = torch.nn.functional.mse_loss(scores, y_group.to(device))
            
        loss.backward()
        
        # フラット化せず，パラメータごとの勾配テンソルのリストとして保持
        grads_list = [p.grad.cpu().numpy() for p in model.parameters() if p.grad is not None]
        group_grads_list[(y_val, a_val)] = grads_list

    gram_matrix_results = {}
    # 対角成分も計算するため combinations_with_replacement を使用
    for (y1, a1), (y2, a2) in combinations_with_replacement(group_keys, 2):
        grads1, grads2 = group_grads_list.get((y1, a1)), group_grads_list.get((y2, a2))
        
        key_name = f"G({y1},{a1})_vs_G({y2},{a2})"
        if grads1 is not None and grads2 is not None:
            # 層ごとの学習率を考慮した重み付き内積を計算
            weighted_dot_product = 0.0
            param_idx = 0
            # optimizer_params の順序は model.parameters() と一致していることを前提とする
            for group in optimizer_params:
                lr = group['lr']
                for _ in group['params']:
                    # パラメータが勾配を持つことを確認
                    if param_idx < len(grads1) and param_idx < len(grads2):
                        grad1_p = grads1[param_idx].flatten()
                        grad2_p = grads2[param_idx].flatten()
                        weighted_dot_product += lr * np.dot(grad1_p, grad2_p)
                    param_idx += 1
            gram_matrix_results[key_name] = weighted_dot_product
        else:
            gram_matrix_results[key_name] = np.nan

    return gram_matrix_results, group_grads_list

# ==============================================================================
# 勾配基底ベクトルの分析
# ==============================================================================

def _weighted_inner_product(v1_list, v2_list, optimizer_params):
    """
    2つのパラメータごと勾配リストのη-重み付き内積 <v1, v2>_η を計算
    """
    dot_product = 0.0
    param_idx = 0
    # optimizer_params の順序は model.parameters() と一致していることを前提とする
    for group in optimizer_params:
        lr = group['lr'] # lr が η_p に対応
        for _ in group['params']:
            # パラメータが勾配を持つことを確認
            if param_idx < len(v1_list) and param_idx < len(v2_list):
                v1_p = v1_list[param_idx].flatten()
                v2_p = v2_list[param_idx].flatten()
                # <v1, v2>_η = sum_p η_p * (v1_p・v2_p)
                dot_product += lr * np.dot(v1_p, v2_p)
            param_idx += 1
    return dot_product

def analyze_gradient_basis(group_grads_list, optimizer_params, dataset_type):
    """
    勾配の基底ベクトル (v_inv, v_C, v_A, v_AY) を計算し，
    そのη-重み付きノルムとコサイン類似度を計算する
    """
    print(f"\nAnalyzing GRADIENT BASIS VECTORS on {dataset_type} data...")
    basis_vectors_list = {}
    
    # 形状を取得するために，存在する最初の勾配リストを探す
    first_valid_grad_list = next((g for g in group_grads_list.values() if g is not None), None)
    if first_valid_grad_list is None:
        print("No valid group gradients found. Skipping basis analysis.")
        return {'vectors': {}} # 空の辞書を返す

    # 0で初期化
    v_inv = [np.zeros_like(g) for g in first_valid_grad_list]
    v_C   = [np.zeros_like(g) for g in first_valid_grad_list]
    v_A   = [np.zeros_like(g) for g in first_valid_grad_list]
    v_AY  = [np.zeros_like(g) for g in first_valid_grad_list]

    group_keys = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for g in group_keys:
        grads_p_list = group_grads_list.get(g)
        if grads_p_list is not None:
            y, a = g
            for i in range(len(grads_p_list)):
                v_inv[i] += grads_p_list[i]
                v_C[i]   += y * grads_p_list[i]
                v_A[i]   += a * grads_p_list[i]
                v_AY[i]  += (y * a) * grads_p_list[i]

    # 定義1に従い 1/4 で正規化
    basis_vectors_list['inv'] = [v / 4.0 for v in v_inv]
    basis_vectors_list['C']   = [v / 4.0 for v in v_C]
    basis_vectors_list['A']   = [v / 4.0 for v in v_A]
    basis_vectors_list['AY']  = [v / 4.0 for v in v_AY]

    results = {}
    basis_keys = ['inv', 'C', 'A', 'AY']
    norms_sq = {}

    # 1. ノルムを計算 ||v||_η
    for key in basis_keys:
        v_list = basis_vectors_list[key]
        # ||v||^2 = <v, v>_η
        norm_sq = _weighted_inner_product(v_list, v_list, optimizer_params)
        norms_sq[key] = norm_sq
        results[f'norm_{key}'] = np.sqrt(norm_sq) if norm_sq >= 0 else np.nan

    # 2. コサイン類似度を計算 <v1, v2>_η / (||v1||_η ||v2||_η)
    for key1, key2 in combinations(basis_keys, 2):
        v1_list, v2_list = basis_vectors_list[key1], basis_vectors_list[key2]
        inner_prod = _weighted_inner_product(v1_list, v2_list, optimizer_params)
        
        norm1 = np.sqrt(norms_sq[key1])
        norm2 = np.sqrt(norms_sq[key2])
        
        if norm1 > 1e-9 and norm2 > 1e-9:
            cosine_sim = inner_prod / (norm1 * norm2)
        else:
            cosine_sim = np.nan
        results[f'cosine_{key1}_{key2}'] = cosine_sim

    # プロットや後続の分析（命題1）のために，計算したベクトル自体も返す
    results['vectors'] = basis_vectors_list
    return results

# ==============================================================================
# 命題1の項の分析
# ==============================================================================
def analyze_proposition1_terms(group_grads_list, basis_vectors_list, optimizer_params, config, y_data, a_data, dataset_type):
    """
    命題1で定義された3つの項の大きさを，設定ファイルで指定された
    グループペア (g1, g2) ごとに計算する
    """
    print(f"\nAnalyzing PROPOSITION 1 TERMS on {dataset_type} data...")
    
    # 設定ファイルから分析対象のペアリストを取得
    group_pairs_config = config.get('proposition1_terms', {}).get('group_pairs', [])
    if not group_pairs_config:
        print("No group pairs configured for Proposition 1 analysis. Skipping.")
        return {}

    # データセットのモーメントを計算
    y_np, a_np = y_data.numpy(), a_data.numpy()
    y_bar = np.mean(y_np)
    a_bar = np.mean(a_np)
    # 共分散 Cov(Y, A) = E[YA] - E[Y]E[A]
    rho_train = np.mean(y_np * a_np) - (y_bar * a_bar)

    # 基底ベクトルを取得
    v_inv = basis_vectors_list['inv']
    v_C = basis_vectors_list['C']
    v_A = basis_vectors_list['A']
    v_AY = basis_vectors_list['AY']

    results = {}

    for pair in group_pairs_config:
        g1_key = tuple(pair[0])
        g2_key = tuple(pair[1])
        pair_name = f"g1_{g1_key}_g2_{g2_key}"
        
        grad_g1 = group_grads_list.get(g1_key)
        grad_g2 = group_grads_list.get(g2_key)

        if grad_g1 is None or grad_g2 is None:
            print(f"Skipping pair {pair_name}: gradient data missing.")
            results[f'{pair_name}_term1'] = np.nan
            results[f'{pair_name}_term2'] = np.nan
            results[f'{pair_name}_term3'] = np.nan
            continue

        # Δ∇_{2,1} = ∇g2 - ∇g1
        delta_nabla_21 = [g2 - g1 for g1, g2 in zip(grad_g1, grad_g2)]

        # --- 項(I) ---
        # <Δ∇_{2,1}, v_inv>_η
        term1 = _weighted_inner_product(delta_nabla_21, v_inv, optimizer_params)
        
        # --- 項(II) ---
        # v_term2 = y_bar*v_C + a_bar*v_A + y_bar*a_bar*v_AY
        v_term2 = [(y_bar * vc) + (a_bar * va) + (y_bar * a_bar * vay) 
                   for vc, va, vay in zip(v_C, v_A, v_AY)]
        # <Δ∇_{2,1}, v_term2>_η
        term2 = _weighted_inner_product(delta_nabla_21, v_term2, optimizer_params)
        
        # --- 項(III) ---
        # v_term3 = ρ_train * v_AY
        v_term3 = [rho_train * vay for vay in v_AY]
        # <Δ∇_{2,1}, v_term3>_η
        term3 = _weighted_inner_product(delta_nabla_21, v_term3, optimizer_params)

        results[f'{pair_name}_term1'] = term1
        results[f'{pair_name}_term2'] = term2
        results[f'{pair_name}_term3'] = term3

    return results


# ==============================================================================
# ヤコビアンノルムの分析
# ==============================================================================
def analyze_jacobian_norms(model, X_data, y_data, a_data, device, num_samples, dataset_type):
    """グループごとのヤコビアンノルムと内積を計算"""
    print(f"\nAnalyzing JACOBIAN NORMS on {dataset_type} data (using {num_samples} samples per group)...")
    group_keys = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    group_jacobians = {}

    for y_val, a_val in group_keys:
        mask = (y_data == y_val) & (a_data == a_val)
        if mask.sum() == 0:
            group_jacobians[(y_val, a_val)] = None
            continue
        
        X_group = X_data[mask]
        # サンプル数が指定数より多い場合はランダムサンプリング
        if len(X_group) > num_samples:
            indices = np.random.choice(len(X_group), num_samples, replace=False)
            X_subset = X_group[indices]
        else:
            X_subset = X_group
            
        group_jacobians[(y_val, a_val)] = get_model_jacobian(model, X_subset, device)

    jacobian_results = {}
    # ノルムの計算
    for (y, a), jacobian in group_jacobians.items():
        key_name = f"norm_G({y},{a})"
        if jacobian is not None:
            jacobian_results[key_name] = np.linalg.norm(jacobian)**2
        else:
            jacobian_results[key_name] = np.nan

    # 内積の計算
    for (y1, a1), (y2, a2) in combinations(group_keys, 2):
        jac1, jac2 = group_jacobians.get((y1, a1)), group_jacobians.get((y2, a2))
        key_name = f"dot_G({y1},{a1})_vs_G({y2},{a2})"
        if jac1 is not None and jac2 is not None:
            jacobian_results[key_name] = np.dot(jac1, jac2)
        else:
            jacobian_results[key_name] = np.nan
            
    return jacobian_results

# ==============================================================================
# 勾配グラム行列のスペクトル分析
# ==============================================================================
def analyze_gradient_gram_spectrum(gram_matrix_results, dataset_type):
    """勾配グラム行列の固有値と主固有ベクトルを計算"""
    print(f"\nAnalyzing GRADIENT GRAM SPECTRUM on {dataset_type} data...")
    group_order = [(-1,-1), (-1,1), (1,-1), (1,1)]
    G = np.zeros((4, 4))
    
    for i, g1 in enumerate(group_order):
        for j, g2 in enumerate(group_order):
            # 対称性を利用
            if i <= j:
                key = f"G({g1[0]},{g1[1]})_vs_G({g2[0]},{g2[1]})"
                val = gram_matrix_results.get(key, np.nan)
                G[i, j] = G[j, i] = val
            
    if np.isnan(G).any():
        print("Gram matrix contains NaN values. Skipping spectrum analysis.")
        return {'eigenvalues': [np.nan]*4, 'eigenvector1': [np.nan]*4, 'eigenvector2': [np.nan]*4}
        
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(G)
        # 固有値を降順にソート
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        # 上位2つの固有ベクトル
        main_eigenvector = eigenvectors[:, sorted_indices[0]]
        second_eigenvector = eigenvectors[:, sorted_indices[1]]
        
        return {
            'eigenvalues': sorted_eigenvalues.tolist(),
            'eigenvector1': main_eigenvector.tolist(),
            'eigenvector2': second_eigenvector.tolist()
        }
    except np.linalg.LinAlgError as e:
        print(f"Eigendecomposition failed: {e}")
        return {'eigenvalues': [np.nan]*4, 'eigenvector1': [np.nan]*4, 'eigenvector2': [np.nan]*4}

# ==============================================================================
# 勾配ノルム比の分析
# ==============================================================================
def analyze_gradient_norm_ratio(gram_matrix_results, config, dataset_type):
    """
    configで指定されたグループペア [g_num, g_den] ごとに
    勾配ノルムの比 ||∇g_num|| / ||∇g_den|| を計算する
    """
    print(f"\nAnalyzing GRADIENT NORM RATIO on {dataset_type} data...")
    
    # 設定ファイルから分析対象のペアリストを取得
    group_pairs_config = config.get('gradient_norm_ratio_config', {}).get('group_pairs', [])
    if not group_pairs_config:
        print("No group pairs configured for gradient norm ratio analysis. Skipping.")
        return {}

    results = {}
    for pair in group_pairs_config:
        g_num_key_tuple = tuple(pair[0]) # [y, a]
        g_den_key_tuple = tuple(pair[1]) # [y, a]
        
        # 結果格納用のキー名 (例: 'ratio_g(-1, -1)_vs_g(-1, 1)')
        pair_name = f"ratio_g{g_num_key_tuple}_vs_g{g_den_key_tuple}"
        
        # 勾配ノルムの2乗はグラム行列の対角成分
        # G(y,a)_vs_G(y,a)
        norm_sq_num_key = f"G({g_num_key_tuple[0]},{g_num_key_tuple[1]})_vs_G({g_num_key_tuple[0]},{g_num_key_tuple[1]})"
        norm_sq_den_key = f"G({g_den_key_tuple[0]},{g_den_key_tuple[1]})_vs_G({g_den_key_tuple[0]},{g_den_key_tuple[1]})"

        norm_sq_num = gram_matrix_results.get(norm_sq_num_key, np.nan)
        norm_sq_den = gram_matrix_results.get(norm_sq_den_key, np.nan)

        # NaNチェック
        if np.isnan(norm_sq_num) or np.isnan(norm_sq_den):
            print(f"Cannot calculate ratio for {pair_name}: missing norm values.")
            results[pair_name] = np.nan
            continue

        norm_num = np.sqrt(norm_sq_num)
        norm_den = np.sqrt(norm_sq_den)

        # ゼロ除算を回避
        if norm_den < 1e-9:
            ratio = np.inf if norm_num > 1e-9 else np.nan
        else:
            ratio = norm_num / norm_den
            
        results[pair_name] = ratio

    return results


# ==============================================================================
# 全ての分析を統括するラッパー関数
# ==============================================================================
def run_all_analyses(config, epoch, layers, model, train_outputs, test_outputs,
                     X_train, y_train, a_train, X_test, y_test, a_test, histories,
                     optimizer_params, history): 
    """設定に基づいてすべての分析を実行し，結果をhistory辞書に保存する"""
    analysis_target = config['analysis_target']

    # --- 勾配グラム行列関連の分析 ---
    # (analyze_gradient_basis と analyze_proposition1_terms もこれに依存するため，
    #  先に計算する必要がある)
    grad_gram_train_results, grad_gram_test_results = None, None
    group_grads_train, group_grads_test = None, None
    
    run_grad_gram_related_analysis = config.get('analyze_gradient_gram', False) or \
                                     config.get('analyze_gradient_gram_spectrum', False) or \
                                     config.get('analyze_gradient_norm_ratio', False) or \
                                     config.get('analyze_gradient_basis', False) or \
                                     config.get('analyze_proposition1_terms', False)

    if run_grad_gram_related_analysis:
        if analysis_target in ['train', 'both']:
            # グラム行列と，計算に使った勾配リストを受け取る
            grad_gram_train_results, group_grads_train = analyze_gradient_gram_matrix(
                model, X_train, y_train, a_train, config['device'], config['loss_function'], "Train", optimizer_params)
            if config.get('analyze_gradient_gram', False):
                histories['grad_gram_train'][epoch] = grad_gram_train_results
        if analysis_target in ['test', 'both']:
            grad_gram_test_results, group_grads_test = analyze_gradient_gram_matrix(
                model, X_test, y_test, a_test, config['device'], config['loss_function'], "Test", optimizer_params)
            if config.get('analyze_gradient_gram', False):
                histories['grad_gram_test'][epoch] = grad_gram_test_results

    if config.get('analyze_gradient_gram_spectrum', False):
        if grad_gram_train_results:
            histories['grad_gram_spectrum_train'][epoch] = analyze_gradient_gram_spectrum(grad_gram_train_results, "Train")
        if grad_gram_test_results:
            histories['grad_gram_spectrum_test'][epoch] = analyze_gradient_gram_spectrum(grad_gram_test_results, "Test")

    if config.get('analyze_gradient_norm_ratio', False):
        if grad_gram_train_results:
            histories['grad_norm_ratio_train'][epoch] = analyze_gradient_norm_ratio(
                grad_gram_train_results, config, "Train"
            )
        if grad_gram_test_results:
            histories['grad_norm_ratio_test'][epoch] = analyze_gradient_norm_ratio(
                grad_gram_test_results, config, "Test"
            )

    # --- 勾配基底ベクトルの分析 ---
    basis_vectors_train, basis_vectors_test = None, None
    if config.get('analyze_gradient_basis', False):
        if group_grads_train:
            basis_results_train = analyze_gradient_basis(group_grads_train, optimizer_params, "Train")
            # basis_vectors_list を取り出して，残りをhistoriesに保存
            basis_vectors_train = basis_results_train.pop('vectors', None)
            histories['grad_basis_train'][epoch] = basis_results_train
        if group_grads_test:
            basis_results_test = analyze_gradient_basis(group_grads_test, optimizer_params, "Test")
            basis_vectors_test = basis_results_test.pop('vectors', None)
            histories['grad_basis_test'][epoch] = basis_results_test

    # --- 命題1の項の分析 ---
    if config.get('analyze_proposition1_terms', False):
        # この分析は 'analyze_gradient_basis' がTrueでなくても実行される可能性がある
        # その場合，basis_vectors が None になるので，ここで再計算する
        if group_grads_train and basis_vectors_train is None:
             basis_vectors_train = analyze_gradient_basis(group_grads_train, optimizer_params, "Train").get('vectors')
        if group_grads_test and basis_vectors_test is None:
             basis_vectors_test = analyze_gradient_basis(group_grads_test, optimizer_params, "Test").get('vectors')
        
        if group_grads_train and basis_vectors_train:
            histories['prop1_terms_train'][epoch] = analyze_proposition1_terms(
                group_grads_train, basis_vectors_train, optimizer_params, config, y_train, a_train, "Train")
        if group_grads_test and basis_vectors_test:
            histories['prop1_terms_test'][epoch] = analyze_proposition1_terms(
                group_grads_test, basis_vectors_test, optimizer_params, config, y_test, a_test, "Test")

    # --- ヤコビアンノルムの分析 ---
    if config.get('analyze_jacobian_norm', False):
        if analysis_target in ['train', 'both']:
            histories['jacobian_norm_train'][epoch] = analyze_jacobian_norms(
                model, X_train, y_train, a_train, config['device'], config['jacobian_num_samples'], "Train")
        if analysis_target in ['test', 'both']:
            histories['jacobian_norm_test'][epoch] = analyze_jacobian_norms(
                model, X_test, y_test, a_test, config['device'], config['jacobian_num_samples'], "Test")
