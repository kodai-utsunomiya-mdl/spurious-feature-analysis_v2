# sp/analysis.py

import numpy as np
from itertools import combinations, combinations_with_replacement
import torch
import torch.nn.functional as F
from torch.func import vmap, grad, functional_call

# ==============================================================================
# ヤコビアン計算のヘルパー関数 (vmap を使用してバッチ化)
# ==============================================================================
def get_model_jacobian(model, X_subset, device):
    """
    モデルのヤコビアンの期待値を計算 (パラメータごとのリストとして)
    vmapを使用してバッチ処理で高速化
    """
    model.eval()
    X_subset = X_subset.to(device)

    # 1. モデルのパラメータとバッファを「辞書」として取得
    params_dict = dict(model.named_parameters())
    buffers_dict = dict(model.named_buffers()) # このモデルでは通常は空

    # 2. gradが微分対象とする「パラメータの値のタプル」を取得
    #    (gradは辞書ではなく，タプルやテンソルを引数に取るため)
    params_values_tuple = tuple(params_dict.values())
    
    # 3. パラメータの「キーのタプル」を保持 (後で辞書を再構築するため)
    params_keys = tuple(params_dict.keys())

    def compute_output_scalar(params_values_tuple_inner, x_sample):
        """
        1サンプル(x_sample)と「パラメータ値のタプル」を受け取り，
        モデルのスカラ出力を計算する
        
        Note: buffers_dict は外側のスコープからキャプチャされ、定数として扱われる
        """
        
        # 4. functional_call のために、(キー, 値)タプルからパラメータ辞書を再構築
        param_dict_reconstructed = {
            key: value for key, value in zip(params_keys, params_values_tuple_inner)
        }
        
        # モデルはバッチ入力を想定しているため，(C, H, W) -> (1, C, H, W) のように
        # バッチ次元 (1) を追加して渡す
        x_batch = x_sample.unsqueeze(0)
        
        # 5. functional_call を (param_dict, buffer_dict) の形式で呼び出す
        return torch.func.functional_call(
            model,
            (param_dict_reconstructed, buffers_dict), # (param_dict, buffer_dict)
            x_batch
        )[0].squeeze(0) # [0]で output_scalar を取得

    # 6. 1サンプルのヤコビアンを計算する関数を定義
    #    compute_output_scalar の第0引数 (params_values_tuple_inner) で微分
    compute_jacobian_per_sample = torch.func.grad(compute_output_scalar, argnums=0)

    # 7. vmap を使ってバッチ全体でヤコビアン計算を並列化
    #    compute_jacobian_per_sample を
    #    params_values_tuple (in_dims=None, バッチ処理せず共有)
    #    X_subset (in_dims=0, 0次元目でバッチ処理)
    #    に適用する
    
    # batched_jacobians_tuple は (params_values_tuple と同じ構造の) タプル
    # 各要素の形状は (N, *param_shape), Nはバッチサイズ
    try:
        batched_jacobians_tuple = torch.func.vmap(
            compute_jacobian_per_sample, in_dims=(None, 0)
        )(params_values_tuple, X_subset)
        
    except Exception as e:
        print(f"Error during vmap execution: {e}")
        print("This might be due to CUDA OOM or an issue with the functionalized model.")
        return [] # エラー時は空リストを返す

    # 8. バッチ次元 (dim=0) で平均をとり，期待ヤコビアンを計算
    # (None が返る可能性があるため，None でないかチェックしながら平均化)
    avg_jacobian_tuple = tuple(
        jac_batch.mean(dim=0) if jac_batch is not None else None
        for jac_batch in batched_jacobians_tuple
    )

    # 9. Numpy配列のリストに変換
    avg_jacobian_list = [
        avg_grad.cpu().detach().numpy() if avg_grad is not None else None
        for avg_grad in avg_jacobian_tuple
    ]

    # 10. None (勾配が計算されなかったパラメータ,例: requires_grad=False) を除外
    avg_jacobian_list_filtered = [g for g in avg_jacobian_list if g is not None]

    # 11. 警告チェック
    num_grad_params = sum(1 for p in model.parameters() if p.requires_grad)
    
    if len(avg_jacobian_list_filtered) != num_grad_params:
         # grad() は requires_grad=False のパラメータに対して None を返すため，
         # filtered と num_grad_params は一致するはずだが，念のため警告を残す
         print(f"Warning: Final averaged Jacobian list length ({len(avg_jacobian_list_filtered)}) differs from expected ({num_grad_params}).")

    return avg_jacobian_list_filtered


# ==============================================================================
# 勾配グラム行列の分析
# ==============================================================================
def analyze_gradient_gram_matrix(model, X_data, y_data, a_data, device, loss_function, dataset_type, optimizer_params, num_samples):
    """
    グループ間の勾配の内積からなるグラム行列を計算（学習率を考慮）
    """
    print(f"\nAnalyzing GRADIENT GRAM MATRIX on {dataset_type} data...")
    if num_samples is not None:
        print(f"  (using {num_samples} samples per group)...")
    else:
        print(f"  (using all available samples per group)...")
        
    group_keys = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    group_grads_list = {}

    for y_val, a_val in group_keys:
        mask = (y_data == y_val) & (a_data == a_val)
        if mask.sum() == 0:
            group_grads_list[(y_val, a_val)] = None
            continue
        
        X_group_all, y_group_all = X_data[mask], y_data[mask]
        
        # --- サンプリング ---
        if num_samples is not None and len(X_group_all) > num_samples:
            indices = np.random.choice(len(X_group_all), num_samples, replace=False)
            X_group, y_group = X_group_all[indices], y_group_all[indices]
        else:
            X_group, y_group = X_group_all, y_group_all
        
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
    if v1_list is None or v2_list is None:
        return np.nan
        
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
    
    if param_idx != len(v1_list) or param_idx != len(v2_list):
        # `fix_final_layer` などで optimizer_params と grad_list の長さが
        # 一致しない場合がある．param_idx は optimizer_params に基づく
        # grad_list は model.parameters() からの勾配に基づく
        # _weighted_inner_product は optimizer_params に含まれる
        # パラメータのみの内積を計算するため，これで正しい．
        pass

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
            # first_valid_grad_list と長さが異なる場合がある (fix_layer)
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
        # grad_g1 と grad_g2 の長さが一致することを期待
        if len(grad_g1) != len(grad_g2):
             print(f"Skipping pair {pair_name}: gradient list length mismatch.")
             continue
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
def analyze_jacobian_norms(model, X_data, y_data, a_data, device, num_samples, optimizer_params, dataset_type):
    """
    グループごとのヤコビアンのη-ノルムとη-内積，および
    幾何学的中心ベクトル (Delta C, Delta L) のノルムとアラインメントを計算
    """
    print(f"\nAnalyzing JACOBIAN (η-weighted) NORMS on {dataset_type} data (using {num_samples} samples per group)...")
    group_keys = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    group_jacobians_list = {}

    for y_val, a_val in group_keys:
        mask = (y_data == y_val) & (a_data == a_val)
        if mask.sum() == 0:
            group_jacobians_list[(y_val, a_val)] = None
            continue
        
        X_group = X_data[mask]
        # サンプル数が指定数より多い場合はランダムサンプリング
        if len(X_group) > num_samples:
            indices = np.random.choice(len(X_group), num_samples, replace=False)
            X_subset = X_group[indices]
        else:
            X_subset = X_group
            
        # ここで vmap 化された get_model_jacobian が呼び出される
        group_jacobians_list[(y_val, a_val)] = get_model_jacobian(model, X_subset, device)

    jacobian_results = {}
    group_norms_sq = {} # ノルムの2乗をキャッシュ

    # --- 1. 元のヤコビアンノルム (η-ノルムで計算) ---
    for (y, a), jac_list in group_jacobians_list.items():
        key_name = f"norm_G({y},{a})"
        if jac_list is not None:
            # ||J||^2 = <J, J>_η
            norm_sq = _weighted_inner_product(jac_list, jac_list, optimizer_params)
            group_norms_sq[(y, a)] = norm_sq
            jacobian_results[key_name] = norm_sq # 2乗のまま保存
        else:
            group_norms_sq[(y, a)] = np.nan
            jacobian_results[key_name] = np.nan

    # --- 2. 元のヤコビアン内積 (η-内積で計算) ---
    for (y1, a1), (y2, a2) in combinations(group_keys, 2):
        jac1_list = group_jacobians_list.get((y1, a1))
        jac2_list = group_jacobians_list.get((y2, a2))
        key_name = f"dot_G({y1},{a1})_vs_G({y2},{a2})"
        if jac1_list is not None and jac2_list is not None:
            inner_prod = _weighted_inner_product(jac1_list, jac2_list, optimizer_params)
            jacobian_results[key_name] = inner_prod
        else:
            jacobian_results[key_name] = np.nan
    
    # --- 3. 幾何学的中心ベクトルの分析 ---
    # C+1 = 1/2 * (Phi_(+1,+1) + Phi_(-1,+1))
    # C-1 = 1/2 * (Phi_(+1,-1) + Phi_(-1,-1))
    # L+1 = 1/2 * (Phi_(+1,+1) + Phi_(+1,-1))
    # L-1 = 1/2 * (Phi_(-1,-1) + Phi_(-1,+1))
    #
    # Delta_C = C+1 - C-1 = 1/2 * (Phi_pp + Phi_np - Phi_pn - Phi_nn)
    # Delta_L = L+1 - L-1 = 1/2 * (Phi_pp + Phi_pn - Phi_nn - Phi_np)
    #
    # m_A = 2 * Delta_C
    # m_Y = 2 * Delta_L
    
    Phi_pp = group_jacobians_list.get((1, 1))
    Phi_pn = group_jacobians_list.get((1, -1))
    Phi_np = group_jacobians_list.get((-1, 1))
    Phi_nn = group_jacobians_list.get((-1, -1))

    # 必要なグループがすべて存在するかチェック
    if all(v is not None for v in [Phi_pp, Phi_pn, Phi_np, Phi_nn]):
        try:
            # パラメータリストの長さを取得 (すべて同じ長さと仮定)
            num_params = len(Phi_pp)
            if not all(len(v) == num_params for v in [Phi_pn, Phi_np, Phi_nn]):
                raise ValueError("Jacobian lists have mismatched parameter counts.")

            # Delta_C = 1/2 * ( (Phi_pp - Phi_pn) + (Phi_np - Phi_nn) )
            delta_C_list = [0.5 * (Phi_pp[i] - Phi_pn[i] + Phi_np[i] - Phi_nn[i]) for i in range(num_params)]
            
            # Delta_L = 1/2 * ( (Phi_pp - Phi_np) + (Phi_pn - Phi_nn) )
            delta_L_list = [0.5 * (Phi_pp[i] + Phi_pn[i] - Phi_np[i] - Phi_nn[i]) for i in range(num_params)]

            # ノルムの2乗 ||Delta C||_η^2 と ||Delta L||_η^2
            norm_sq_delta_C = _weighted_inner_product(delta_C_list, delta_C_list, optimizer_params)
            norm_sq_delta_L = _weighted_inner_product(delta_L_list, delta_L_list, optimizer_params)
            
            # ノルム ||Delta C||_η と ||Delta L||_η
            norm_delta_C = np.sqrt(norm_sq_delta_C) if norm_sq_delta_C >= 0 else np.nan
            norm_delta_L = np.sqrt(norm_sq_delta_L) if norm_sq_delta_L >= 0 else np.nan
            
            jacobian_results["norm_Delta_C"] = norm_delta_C
            jacobian_results["norm_Delta_L"] = norm_delta_L
            
            # 内積 <Delta C, Delta L>_η
            inner_prod_CL = _weighted_inner_product(delta_C_list, delta_L_list, optimizer_params)
            jacobian_results["dot_Delta_C_Delta_L"] = inner_prod_CL

            # コサイン類似度
            if norm_delta_C > 1e-9 and norm_delta_L > 1e-9:
                cosine_sim_CL = inner_prod_CL / (norm_delta_C * norm_delta_L)
            else:
                cosine_sim_CL = np.nan
            jacobian_results["cosine_Delta_C_Delta_L"] = cosine_sim_CL

            # ||m_A||^2 = 4 * ||Delta_C||^2
            # ||m_Y||^2 = 4 * ||Delta_L||^2
            # <m_A, m_Y> = 4 * <Delta_C, Delta_L>
            jacobian_results["paper_norm_sq_mu_A"] = 4.0 * norm_sq_delta_C
            jacobian_results["paper_norm_sq_mu_Y"] = 4.0 * norm_sq_delta_L
            jacobian_results["paper_dot_mu_A_mu_Y"] = 4.0 * inner_prod_CL

        except Exception as e:
            print(f"Error during geometric center analysis: {e}")
            keys_to_nan = ["norm_Delta_C", "norm_Delta_L", "dot_Delta_C_Delta_L", 
                           "cosine_Delta_C_Delta_L", "paper_norm_sq_mu_A", 
                           "paper_norm_sq_mu_Y", "paper_dot_mu_A_mu_Y"]
            for k in keys_to_nan:
                jacobian_results[k] = np.nan
    else:
        print(f"Skipping geometric center (Delta C, Delta L) analysis: missing group Jacobian data.")
        keys_to_nan = ["norm_Delta_C", "norm_Delta_L", "dot_Delta_C_Delta_L", 
                       "cosine_Delta_C_Delta_L", "paper_norm_sq_mu_A", 
                       "paper_norm_sq_mu_Y", "paper_dot_mu_A_mu_Y"]
        for k in keys_to_nan:
            jacobian_results[k] = np.nan
            
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
        # --- サンプル数をconfigから取得 ---
        num_samples_grad = config.get('gradient_gram_num_samples', None) # Noneの場合は全サンプル使用
        
        if analysis_target in ['train', 'both']:
            # グラム行列と，計算に使った勾配リストを受け取る
            grad_gram_train_results, group_grads_train = analyze_gradient_gram_matrix(
                model, X_train, y_train, a_train, config['device'], config['loss_function'], "Train", optimizer_params,
                num_samples_grad
            )
            if config.get('analyze_gradient_gram', False):
                histories['grad_gram_train'][epoch] = grad_gram_train_results
        if analysis_target in ['test', 'both']:
            grad_gram_test_results, group_grads_test = analyze_gradient_gram_matrix(
                model, X_test, y_test, a_test, config['device'], config['loss_function'], "Test", optimizer_params,
                num_samples_grad
            )
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
        # ヤコビアンのサンプル数を取得
        num_samples_jac = config.get('jacobian_num_samples', 100)
        
        if analysis_target in ['train', 'both']:
            histories['jacobian_norm_train'][epoch] = analyze_jacobian_norms(
                model, X_train, y_train, a_train, config['device'], 
                num_samples_jac, optimizer_params, "Train")
        if analysis_target in ['test', 'both']:
            histories['jacobian_norm_test'][epoch] = analyze_jacobian_norms(
                model, X_test, y_test, a_test, config['device'], 
                num_samples_jac, optimizer_params, "Test")
