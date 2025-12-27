# sp/analysis.py

import numpy as np
from itertools import combinations
import torch
import plotting

# ==============================================================================
# ヤコビアン計算のヘルパー関数
# ==============================================================================
def get_model_jacobian(model, X_subset, device):
    """
    モデルのヤコビアンの期待値 (勾配の平均) を計算
    """
    model.eval()
    model.zero_grad() # 念のためリセット
    X_subset = X_subset.to(device)
    
    # 1. バッチ全体の出力を計算
    scores, _ = model(X_subset) # (N,)
    
    # 2. 出力の平均をとる (効率化)
    # 線形性により E[∇f(x)] = ∇E[f(x)] なので，先に平均をとってから微分しても結果は同じ
    mean_output = scores.mean()
    
    # 3. 平均出力に対する勾配を計算 (標準的なBackprop)
    # requires_grad=True のパラメータのみ対象
    params = [p for p in model.parameters() if p.requires_grad]
    
    # create_graph=False (推論/分析用なので計算グラフは保持しなくてよい)
    grads = torch.autograd.grad(mean_output, params, create_graph=False)
    
    # 4. Numpy配列のリストに変換
    avg_jacobian_list = [g.cpu().detach().numpy() for g in grads]

    return avg_jacobian_list


# ==============================================================================
# グループ勾配の計算
# ==============================================================================
def calculate_group_gradients(model, X_data, y_data, a_data, device, loss_function, dataset_type, num_samples):
    """
    グループごとの勾配を計算 (パラメータごとのリストとして) 
    """
    print(f"\nCalculating GROUP GRADIENTS on {dataset_type} data...")
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

    return group_grads_list

# ==============================================================================
# 勾配基底ベクトルの分析
# ==============================================================================

def _standard_inner_product(v1_list, v2_list):
    """
    2つのパラメータごと勾配リストの標準ユークリッド内積 <v1, v2> を計算
    """
    if v1_list is None or v2_list is None:
        return np.nan
        
    dot_product = 0.0
    
    if len(v1_list) != len(v2_list):
        print(f"Warning: Standard inner product list length mismatch ({len(v1_list)} vs {len(v2_list)}).")
        
    min_len = min(len(v1_list), len(v2_list))

    for i in range(min_len):
        v1_p = v1_list[i].flatten()
        v2_p = v2_list[i].flatten()
        dot_product += np.dot(v1_p, v2_p)

    return dot_product

def analyze_gradient_basis(group_grads_list, dataset_type):
    """
    勾配の基底ベクトル (v_inv, v_Y, v_A, v_AY) を計算し，
    その標準ノルムとコサイン類似度を計算する
    """
    print(f"\nAnalyzing GRADIENT BASIS VECTORS on {dataset_type} data...")
    basis_vectors_list = {}
    
    first_valid_grad_list = next((g for g in group_grads_list.values() if g is not None), None)
    if first_valid_grad_list is None:
        print("No valid group gradients found. Skipping basis analysis.")
        return {'vectors': {}} 

    v_inv = [np.zeros_like(g) for g in first_valid_grad_list]
    v_Y   = [np.zeros_like(g) for g in first_valid_grad_list]
    v_A   = [np.zeros_like(g) for g in first_valid_grad_list]
    v_AY  = [np.zeros_like(g) for g in first_valid_grad_list]

    group_keys = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for g in group_keys:
        grads_p_list = group_grads_list.get(g)
        if grads_p_list is not None:
            y, a = g
            for i in range(len(v_inv)):
                if i < len(grads_p_list):
                    v_inv[i] += grads_p_list[i]
                    v_Y[i]   += y * grads_p_list[i]
                    v_A[i]   += a * grads_p_list[i]
                    v_AY[i]  += (y * a) * grads_p_list[i]

    basis_vectors_list['inv'] = [v / 4.0 for v in v_inv]
    basis_vectors_list['Y']   = [v / 4.0 for v in v_Y]  
    basis_vectors_list['A']   = [v / 4.0 for v in v_A]
    basis_vectors_list['AY']  = [v / 4.0 for v in v_AY]

    results = {}
    basis_keys = ['inv', 'Y', 'A', 'AY']
    norms_sq = {}

    for key in basis_keys:
        v_list = basis_vectors_list[key]
        norm_sq = _standard_inner_product(v_list, v_list)
        norms_sq[key] = norm_sq
        results[f'norm_{key}'] = np.sqrt(norm_sq) if norm_sq >= 0 else np.nan

    for key1, key2 in combinations(basis_keys, 2):
        v1_list, v2_list = basis_vectors_list[key1], basis_vectors_list[key2]
        inner_prod = _standard_inner_product(v1_list, v2_list)
        norm1 = np.sqrt(norms_sq[key1])
        norm2 = np.sqrt(norms_sq[key2])
        if norm1 > 1e-9 and norm2 > 1e-9:
            cosine_sim = inner_prod / (norm1 * norm2)
        else:
            cosine_sim = np.nan
        results[f'cosine_{key1}_{key2}'] = cosine_sim

    results['vectors'] = basis_vectors_list
    return results

# ==============================================================================
# 性能差のダイナミクスの要因の分析
# ==============================================================================
def analyze_gap_dynamics_factors(group_grads_list, basis_vectors_list, config, y_data, a_data, dataset_type):
    """
    性能差のダイナミクスの3つの要因の大きさを計算する
    """
    print(f"\nAnalyzing GAP DYNAMICS FACTORS on {dataset_type} data...")
    group_pairs_config = config.get('gap_dynamics_factors', {}).get('group_pairs', [])
    if not group_pairs_config:
        return {}

    y_np, a_np = y_data.numpy(), a_data.numpy()
    y_bar = np.mean(y_np)
    a_bar = np.mean(a_np)
    rho_train = np.mean(y_np * a_np) - (y_bar * a_bar)

    v_inv = basis_vectors_list['inv']
    v_Y = basis_vectors_list['Y']
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
            results[f'{pair_name}_term1'] = np.nan
            results[f'{pair_name}_term2'] = np.nan
            results[f'{pair_name}_term3'] = np.nan
            continue

        min_len = min(len(grad_g1), len(grad_g2), len(v_inv))
        delta_nabla_21 = [grad_g2[i] - grad_g1[i] for i in range(min_len)]
        v_inv_trunc = v_inv[:min_len]
        v_Y_trunc = v_Y[:min_len]
        v_A_trunc = v_A[:min_len]
        v_AY_trunc = v_AY[:min_len]

        term1 = _standard_inner_product(delta_nabla_21, v_inv_trunc)
        v_term2 = [(y_bar * vy) + (a_bar * va) + (y_bar * a_bar * vay) 
                   for vy, va, vay in zip(v_Y_trunc, v_A_trunc, v_AY_trunc)]
        term2 = _standard_inner_product(delta_nabla_21, v_term2)
        v_term3 = [rho_train * vay for vay in v_AY_trunc]
        term3 = _standard_inner_product(delta_nabla_21, v_term3)

        results[f'{pair_name}_term1'] = term1
        results[f'{pair_name}_term2'] = term2
        results[f'{pair_name}_term3'] = term3

    return results


# ==============================================================================
# グループヤコビアンの計算
# ==============================================================================
def calculate_group_jacobians(model, X_data, y_data, a_data, device, num_samples, dataset_type):
    """
    グループごとのヤコビアンの期待値 (平均埋め込み m_g(t)) を計算
    """
    print(f"\nCalculating GROUP JACOBIANS on {dataset_type} data (using {num_samples} samples per group)...")
    group_keys = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    group_jacobians_list = {}

    for y_val, a_val in group_keys:
        mask = (y_data == y_val) & (a_data == a_val)
        if mask.sum() == 0:
            group_jacobians_list[(y_val, a_val)] = None
            continue
        
        X_group = X_data[mask]
        if len(X_group) > num_samples:
            indices = np.random.choice(len(X_group), num_samples, replace=False)
            X_subset = X_group[indices]
        else:
            X_subset = X_group

        group_jacobians_list[(y_val, a_val)] = get_model_jacobian(model, X_subset, device)
    
    return group_jacobians_list

# ==============================================================================
# ヤコビアンノルムの分析
# ==============================================================================
def analyze_jacobian_norms(group_jacobians_list, dataset_type):
    """
    ヤコビアンノルムと幾何学的中心ベクトルの分析
    """
    print(f"\nAnalyzing JACOBIAN (standard) NORMS on {dataset_type} data...")
    group_keys = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    jacobian_results = {}
    group_norms_sq = {}

    for (y, a), jac_list in group_jacobians_list.items():
        key_name = f"norm_G({y},{a})"
        if jac_list is not None:
            norm_sq = _standard_inner_product(jac_list, jac_list)
            group_norms_sq[(y, a)] = norm_sq
            jacobian_results[key_name] = norm_sq 
        else:
            group_norms_sq[(y, a)] = np.nan
            jacobian_results[key_name] = np.nan

    for (y1, a1), (y2, a2) in combinations(group_keys, 2):
        jac1_list = group_jacobians_list.get((y1, a1))
        jac2_list = group_jacobians_list.get((y2, a2))
        key_name_dot = f"dot_G({y1},{a1})_vs_G({y2},{a2})"
        key_name_cosine = f"cosine_G({y1},{a1})_vs_G({y2},{a2})"
        key_name_dist_sq = f"dist_sq_G({y1},{a1})_vs_G({y2},{a2})"

        if jac1_list is not None and jac2_list is not None:
            inner_prod = _standard_inner_product(jac1_list, jac2_list)
            jacobian_results[key_name_dot] = inner_prod
            norm1_sq = group_norms_sq.get((y1, a1), np.nan)
            norm2_sq = group_norms_sq.get((y2, a2), np.nan)
            if not np.isnan(norm1_sq) and not np.isnan(norm2_sq):
                norm1, norm2 = np.sqrt(norm1_sq), np.sqrt(norm2_sq)
                jacobian_results[key_name_cosine] = inner_prod / (norm1 * norm2) if norm1*norm2 > 1e-9 else np.nan
                jacobian_results[key_name_dist_sq] = norm1_sq + norm2_sq - 2 * inner_prod
            else:
                jacobian_results[key_name_cosine] = np.nan
                jacobian_results[key_name_dist_sq] = np.nan
        else:
            jacobian_results[key_name_dot] = np.nan
            jacobian_results[key_name_cosine] = np.nan
            jacobian_results[key_name_dist_sq] = np.nan
    
    Phi_pp = group_jacobians_list.get((1, 1))
    Phi_pn = group_jacobians_list.get((1, -1))
    Phi_np = group_jacobians_list.get((-1, 1))
    Phi_nn = group_jacobians_list.get((-1, -1))

    keys_to_nan = ["norm_Delta_S", "norm_Delta_L", "ratio_Delta_S_L", 
                   "dot_Delta_S_Delta_L", "cosine_Delta_S_Delta_L", 
                   "paper_norm_sq_m_A", "paper_norm_sq_m_Y", "paper_dot_m_A_m_Y"]

    if all(v is not None for v in [Phi_pp, Phi_pn, Phi_np, Phi_nn]):
        try:
            num_params = len(Phi_pp)
            delta_S_list = [0.5 * (Phi_pp[i] - Phi_pn[i] + Phi_np[i] - Phi_nn[i]) for i in range(num_params)]
            delta_L_list = [0.5 * (Phi_pp[i] + Phi_pn[i] - Phi_np[i] - Phi_nn[i]) for i in range(num_params)]
            norm_sq_delta_S = _standard_inner_product(delta_S_list, delta_S_list)
            norm_sq_delta_L = _standard_inner_product(delta_L_list, delta_L_list)
            norm_delta_S, norm_delta_L = np.sqrt(norm_sq_delta_S), np.sqrt(norm_sq_delta_L)
            jacobian_results["norm_Delta_S"] = norm_delta_S
            jacobian_results["norm_Delta_L"] = norm_delta_L
            jacobian_results["ratio_Delta_S_L"] = norm_delta_S / norm_delta_L if norm_delta_L > 1e-9 else np.nan
            inner_prod_SL = _standard_inner_product(delta_S_list, delta_L_list)
            jacobian_results["dot_Delta_S_Delta_L"] = inner_prod_SL
            jacobian_results["cosine_Delta_S_Delta_L"] = inner_prod_SL / (norm_delta_S * norm_delta_L) if norm_delta_S*norm_delta_L > 1e-9 else np.nan
            jacobian_results["paper_norm_sq_m_A"] = 0.25 * norm_sq_delta_S
            jacobian_results["paper_norm_sq_m_Y"] = 0.25 * norm_sq_delta_L
            jacobian_results["paper_dot_m_A_m_Y"] = 0.25 * inner_prod_SL
        except Exception as e:
            for k in keys_to_nan: jacobian_results[k] = np.nan
    else:
        for k in keys_to_nan: jacobian_results[k] = np.nan
            
    return jacobian_results

# ==============================================================================
# 静的・動的分解の分析
# ==============================================================================
def analyze_static_dynamic_decomposition(group_grads_list, group_jacobians_list, config, y_data, a_data, dataset_type):
    """
    同じラベルを持つグループ間の性能差ダイナミクスの静的・動的分解 (項A, B, C) を計算
    """
    print(f"\nAnalyzing STATIC/DYNAMIC DECOMPOSITION on {dataset_type} data...")
    group_pairs_config = config.get('static_dynamic_decomposition', {}).get('group_pairs', [])
    if not group_pairs_config: return {}

    loss_function = config['loss_function']
    if loss_function == 'logistic': c0_map = {-1.0: 0.5, 1.0: -0.5}
    elif loss_function == 'mse': c0_map = {-1.0: 2.0, 1.0: -2.0}
    else: return {}
    
    group_keys = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    nabla_static, nabla_dynamic = {}, {}
    first_valid_grad_list = next((g for g in group_grads_list.values() if g is not None), None)
    if first_valid_grad_list is None: return {}
    
    for g in group_keys:
        y_val, a_val = g
        nabla_g, m_g = group_grads_list.get(g), group_jacobians_list.get(g)
        if nabla_g is None or m_g is None:
            nabla_static[g] = None; nabla_dynamic[g] = None; continue
        min_len = min(len(nabla_g), len(m_g))
        nabla_g, m_g = nabla_g[:min_len], m_g[:min_len]
        c0_val = c0_map[y_val]
        nabla_static[g] = [c0_val * m_p for m_p in m_g]
        nabla_dynamic[g] = [nabla_p - static_p for nabla_p, static_p in zip(nabla_g, nabla_static[g])]

    pi_g = {}
    total_samples = len(y_data)
    if total_samples == 0: return {}
    for g in group_keys:
        mask = (y_data == g[0]) & (a_data == g[1])
        pi_g[g] = mask.sum().item() / total_samples

    nabla_train_static = [np.zeros_like(g) for g in first_valid_grad_list]
    nabla_train_dynamic = [np.zeros_like(g) for g in first_valid_grad_list]
    for g in group_keys:
        pi_val = pi_g[g]
        if nabla_static[g] is not None:
            for i in range(min(len(nabla_train_static), len(nabla_static[g]))):
                nabla_train_static[i] += pi_val * nabla_static[g][i]
        if nabla_dynamic[g] is not None:
            for i in range(min(len(nabla_train_dynamic), len(nabla_dynamic[g]))):
                nabla_train_dynamic[i] += pi_val * nabla_dynamic[g][i]

    results = {}
    for pair in group_pairs_config:
        g_min_key, g_maj_key = tuple(pair[0]), tuple(pair[1])
        pair_name = f"g_min_{g_min_key}_g_maj_{g_maj_key}"
        nms, nms_min, nmd, nmd_min = nabla_static.get(g_maj_key), nabla_static.get(g_min_key), nabla_dynamic.get(g_maj_key), nabla_dynamic.get(g_min_key)
        if any(v is None for v in [nms, nms_min, nmd, nmd_min]):
            for k in ['TermA', 'TermB', 'TermC']: results[f'{pair_name}_{k}'] = np.nan
            continue
        min_len = min(len(nms), len(nms_min), len(nmd), len(nmd_min), len(nabla_train_static))
        delta_static = [nms[i] - nms_min[i] for i in range(min_len)]
        delta_dynamic = [nmd[i] - nmd_min[i] for i in range(min_len)]
        term_A = _standard_inner_product(delta_static, nabla_train_static[:min_len])
        term_B = _standard_inner_product(delta_static, nabla_train_dynamic[:min_len]) + _standard_inner_product(delta_dynamic, nabla_train_static[:min_len])
        term_C = _standard_inner_product(delta_dynamic, nabla_train_dynamic[:min_len])
        results[f'{pair_name}_TermA'], results[f'{pair_name}_TermB'], results[f'{pair_name}_TermC'] = term_A, term_B, term_C
    
    return results

# ==============================================================================
# モデル出力期待値の分析
# ==============================================================================
def analyze_model_output_expectation(model, X_data, y_data, a_data, device, batch_size=None):
    """
    各グループにおけるモデル出力の期待値 E[f(x)] および標準偏差 Std[f(x)] を計算する
    """
    model.eval()
    group_keys = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    results = {}
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_data), batch_size=(batch_size or len(X_data)), shuffle=False)
    all_scores = torch.cat([model(batch_X.to(device))[0].cpu() for (batch_X,) in loader])

    for y_val, a_val in group_keys:
        mask = (y_data == y_val) & (a_data == a_val)
        if mask.sum() > 0:
            results[f'E[f(x)]_G({y_val},{a_val})'], results[f'Std[f(x)]_G({y_val},{a_val})'] = all_scores[mask].mean().item(), all_scores[mask].std().item()
        else:
            results[f'E[f(x)]_G({y_val},{a_val})'], results[f'Std[f(x)]_G({y_val},{a_val})'] = np.nan, np.nan
    return results


# ==============================================================================
# 可視化 (UMAP / t-SNE) / 特異値解析用 特徴量抽出
# ==============================================================================
def get_layer_representations(model, X_data, y_data, a_data, device, max_samples=2000):
    """
    各層の表現を取得する
    """
    N = len(X_data)
    if max_samples and N > max_samples:
        indices_list = []
        for y_v, a_v in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            mask = (y_data.cpu().numpy() == y_v) & (a_data.cpu().numpy() == a_v)
            group_indices = np.where(mask)[0]
            if len(group_indices) > 0: indices_list.append(np.random.choice(group_indices, min(len(group_indices), max_samples//4), replace=False))
        indices = np.sort(np.concatenate(indices_list)) if indices_list else np.arange(N)
    else: indices = np.arange(N)
    
    X_sub = X_data[indices].to(device)
    model.eval()
    with torch.no_grad():
        logits, outputs_dict = model(X_sub)
    layers_dict = {'Input': X_sub.view(len(X_sub), -1).cpu().numpy()}
    for k in sorted([k for k in outputs_dict.keys() if k.startswith('layer_')]):
        layers_dict[k.replace('layer_', 'Layer ').replace('_', ' ')] = outputs_dict[k].cpu().numpy()
    layers_dict['Output (Logit)'] = logits.cpu().unsqueeze(1).numpy()
    return layers_dict, indices

def run_umap_analysis(config, model, X_train, y_train, a_train, X_test, y_test, a_test, epoch, save_dir):
    """
    可視化を実行する
    """
    if not plotting.HAS_ANY_VIS_LIB: return
    method = config.get('visualization_method', 'umap')
    print(f"\nRunning {method.upper()} Visualization Analysis at Epoch {epoch}...")
    target, n_samples = config.get('umap_analysis_target', 'both'), config.get('umap_num_samples', 2000)
    train_layers, test_layers, train_indices, test_indices = {}, {}, [], []
    if target in ['train', 'both']: train_layers, train_indices = get_layer_representations(model, X_train, y_train, a_train, config['device'], n_samples)
    if target in ['test', 'both']: test_layers, test_indices = get_layer_representations(model, X_test, y_test, a_test, config['device'], n_samples)
    plotting.plot_umap_grid(train_layers, (y_train[train_indices].cpu().numpy() if len(train_indices)>0 else None), (a_train[train_indices].cpu().numpy() if len(train_indices)>0 else None),
                            test_layers, (y_test[test_indices].cpu().numpy() if len(test_indices)>0 else None), (a_test[test_indices].cpu().numpy() if len(test_indices)>0 else None), epoch, save_dir, config)

def run_singular_value_analysis(config, model, X_train, y_train, a_train, X_test, y_test, a_test, epoch, save_dir):
    """
    各層の特徴行列の特異値を計算しプロットする
    """
    if not config.get('analyze_singular_values', False): return
    print(f"\nRunning Singular Value Analysis at Epoch {epoch}...")
    
    target = config.get('singular_values_analysis_target', 'both')
    n_samples = config.get('singular_values_num_samples', 2000)
    
    train_sv_dict = None
    test_sv_dict = None
    
    if target in ['train', 'both']:
        # Train
        layers_dict, _ = get_layer_representations(model, X_train, y_train, a_train, config['device'], n_samples)
        sv_dict = {}
        for name, data in layers_dict.items():
            # data: (N, D) numpy array
            t_data = torch.from_numpy(data).float().to(config['device'])
            # 特異値計算
            try:
                s = torch.linalg.svdvals(t_data)
                sv_dict[name] = s.cpu().numpy()
            except Exception as e:
                print(f"  Warning: SVD failed for layer {name}: {e}")
        train_sv_dict = sv_dict

    if target in ['test', 'both']:
        # Test
        layers_dict, _ = get_layer_representations(model, X_test, y_test, a_test, config['device'], n_samples)
        sv_dict = {}
        for name, data in layers_dict.items():
            t_data = torch.from_numpy(data).float().to(config['device'])
            try:
                s = torch.linalg.svdvals(t_data)
                sv_dict[name] = s.cpu().numpy()
            except Exception as e:
                print(f"  Warning: SVD failed for layer {name}: {e}")
        test_sv_dict = sv_dict

    plotting.plot_singular_values_across_layers(train_sv_dict, test_sv_dict, epoch, save_dir)


# ==============================================================================
# 全ての分析を統括するラッパー関数
# ==============================================================================
def run_all_analyses(config, epoch, layers, model, train_outputs, test_outputs, X_train, y_train, a_train, X_test, y_test, a_test, histories, history): 
    result_dir, analysis_target = config.get('result_dir', '.'), config['analysis_target']
    run_grad_basis, run_gap_factors, run_jacobian_norm, run_static_dynamic, run_output_exp = config.get('analyze_gradient_basis', False), config.get('analyze_gap_dynamics_factors', False), config.get('analyze_jacobian_norm', False), config.get('analyze_static_dynamic_decomposition', False), config.get('analyze_model_output_expectation', False)
    
    # UMAPと特異値解析は epoch リストで制御
    is_umap_epoch = (epoch in (config.get('umap_analysis_epochs', []) or []))
    run_umap = config.get('analyze_umap_representation', False) and is_umap_epoch
    run_svd = config.get('analyze_singular_values', False) and is_umap_epoch

    group_grads_train, group_grads_test = None, None
    if run_grad_basis or run_gap_factors or run_static_dynamic:
        num_s = config.get('gradient_gram_num_samples', None) 
        if analysis_target in ['train', 'both']: group_grads_train = calculate_group_gradients(model, X_train, y_train, a_train, config['device'], config['loss_function'], "Train", num_s)
        if analysis_target in ['test', 'both']: group_grads_test = calculate_group_gradients(model, X_test, y_test, a_test, config['device'], config['loss_function'], "Test", num_s)

    group_jacobians_train, group_jacobians_test = None, None
    if run_jacobian_norm or run_static_dynamic:
        num_s = config.get('jacobian_num_samples', 100)
        if analysis_target in ['train', 'both']: group_jacobians_train = calculate_group_jacobians(model, X_train, y_train, a_train, config['device'], num_s, "Train")
        if analysis_target in ['test', 'both']: group_jacobians_test = calculate_group_jacobians(model, X_test, y_test, a_test, config['device'], num_s, "Test")

    basis_v_train, basis_v_test = None, None
    if run_grad_basis:
        if group_grads_train:
            res = analyze_gradient_basis(group_grads_train, "Train")
            basis_v_train = res.pop('vectors', None); histories['grad_basis_train'][epoch] = res
        if group_grads_test:
            res = analyze_gradient_basis(group_grads_test, "Test")
            basis_v_test = res.pop('vectors', None); histories['grad_basis_test'][epoch] = res

    if run_gap_factors:
        if group_grads_train:
            if basis_v_train is None: basis_v_train = analyze_gradient_basis(group_grads_train, "Train").get('vectors')
            histories['gap_factors_train'][epoch] = analyze_gap_dynamics_factors(group_grads_train, basis_v_train, config, y_train, a_train, "Train")
        if group_grads_test:
            if basis_v_test is None: basis_v_test = analyze_gradient_basis(group_grads_test, "Test").get('vectors')
            histories['gap_factors_test'][epoch] = analyze_gap_dynamics_factors(group_grads_test, basis_v_test, config, y_test, a_test, "Test")

    if run_jacobian_norm:
        if group_jacobians_train: histories['jacobian_norm_train'][epoch] = analyze_jacobian_norms(group_jacobians_train, "Train")
        if group_jacobians_test: histories['jacobian_norm_test'][epoch] = analyze_jacobian_norms(group_jacobians_test, "Test")

    if run_static_dynamic:
        if group_grads_train and group_jacobians_train: histories['static_dynamic_decomp_train'][epoch] = analyze_static_dynamic_decomposition(group_grads_train, group_jacobians_train, config, y_train, a_train, "Train")
        if group_grads_test and group_jacobians_test: histories['static_dynamic_decomp_test'][epoch] = analyze_static_dynamic_decomposition(group_grads_test, group_jacobians_test, config, y_test, a_test, "Test")

    if run_output_exp:
        eval_bs = config.get('eval_batch_size', None)
        if analysis_target in ['train', 'both']: histories['model_output_exp_train'][epoch] = analyze_model_output_expectation(model, X_train, y_train, a_train, config['device'], eval_bs)
        if analysis_target in ['test', 'both']: histories['model_output_exp_test'][epoch] = analyze_model_output_expectation(model, X_test, y_test, a_test, config['device'], eval_bs)
    
    if run_umap: 
        run_umap_analysis(config, model, X_train, y_train, a_train, X_test, y_test, a_test, epoch, config.get('result_dir', '.'))
        
    if run_svd:
        run_singular_value_analysis(config, model, X_train, y_train, a_train, X_test, y_test, a_test, epoch, config.get('result_dir', '.'))
