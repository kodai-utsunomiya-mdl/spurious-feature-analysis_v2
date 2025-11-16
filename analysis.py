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
        
        Note: buffers_dict は外側のスコープからキャプチャされ，定数として扱われる
        """
        
        # 4. functional_call のために，(キー, 値)タプルからパラメータ辞書を再構築
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
# グループ勾配の計算
# ==============================================================================
def calculate_group_gradients(model, X_data, y_data, a_data, device, loss_function, dataset_type, num_samples):
    """
    グループごとの勾配を計算（パラメータごとのリストとして）
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
        # `fix_final_layer` などで requires_grad=False のパラメータがあると
        # get_model_jacobian と group_grads_list でリスト長が異なる可能性がある
        # この場合，短い方に合わせるのが安全
        # (ただし，通常は v1, v2 は同じソースから来るので長さは一致するはず)
        print(f"Warning: Standard inner product list length mismatch ({len(v1_list)} vs {len(v2_list)}).")
        
    min_len = min(len(v1_list), len(v2_list))

    for i in range(min_len):
        v1_p = v1_list[i].flatten()
        v2_p = v2_list[i].flatten()
        # <v1, v2> = sum_p (v1_p・v2_p)
        dot_product += np.dot(v1_p, v2_p)

    return dot_product

def analyze_gradient_basis(group_grads_list, dataset_type):
    """
    勾配の基底ベクトル (v_inv, v_Y, v_A, v_AY) を計算し，
    その標準ノルムとコサイン類似度を計算する
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
    v_Y   = [np.zeros_like(g) for g in first_valid_grad_list]
    v_A   = [np.zeros_like(g) for g in first_valid_grad_list]
    v_AY  = [np.zeros_like(g) for g in first_valid_grad_list]

    group_keys = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for g in group_keys:
        grads_p_list = group_grads_list.get(g)
        if grads_p_list is not None:
            y, a = g
            # first_valid_grad_list と長さが異なる場合がある (fix_layer)
            # v_inv など (first_valid_grad_list基準) の長さに合わせる
            for i in range(len(v_inv)):
                if i < len(grads_p_list):
                    v_inv[i] += grads_p_list[i]
                    v_Y[i]   += y * grads_p_list[i]
                    v_A[i]   += a * grads_p_list[i]
                    v_AY[i]  += (y * a) * grads_p_list[i]

    # 定義1に従い 1/4 で正規化
    basis_vectors_list['inv'] = [v / 4.0 for v in v_inv]
    basis_vectors_list['Y']   = [v / 4.0 for v in v_Y]  
    basis_vectors_list['A']   = [v / 4.0 for v in v_A]
    basis_vectors_list['AY']  = [v / 4.0 for v in v_AY]

    results = {}
    basis_keys = ['inv', 'Y', 'A', 'AY']
    norms_sq = {}

    # 1. ノルムを計算 ||v||
    for key in basis_keys:
        v_list = basis_vectors_list[key]
        # ||v||^2 = <v, v>
        norm_sq = _standard_inner_product(v_list, v_list)
        norms_sq[key] = norm_sq
        results[f'norm_{key}'] = np.sqrt(norm_sq) if norm_sq >= 0 else np.nan

    # 2. コサイン類似度を計算 <v1, v2> / (||v1|| ||v2||)
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

    # プロットや後続の分析のために，計算したベクトル自体も返す
    results['vectors'] = basis_vectors_list
    return results

# ==============================================================================
# 性能差ダイナミクスの要因の分析
# ==============================================================================
def analyze_gap_dynamics_factors(group_grads_list, basis_vectors_list, config, y_data, a_data, dataset_type):
    """
    性能差ダイナミクスの3つの要因（分布非依存，周辺分布バイアス，相関バイアス）の
    大きさを，設定ファイルで指定されたグループペア (g1, g2) ごとに計算する
    """
    print(f"\nAnalyzing GAP DYNAMICS FACTORS on {dataset_type} data...")
    
    # 設定ファイルから分析対象のペアリストを取得
    group_pairs_config = config.get('gap_dynamics_factors', {}).get('group_pairs', [])
    if not group_pairs_config:
        print("No group pairs configured for gap dynamics factor analysis. Skipping.")
        return {}

    # データセットのモーメントを計算
    y_np, a_np = y_data.numpy(), a_data.numpy()
    y_bar = np.mean(y_np)
    a_bar = np.mean(a_np)
    # 共分散 Cov(Y, A) = E[YA] - E[Y]E[A]
    rho_train = np.mean(y_np * a_np) - (y_bar * a_bar)

    # 基底ベクトルを取得
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
            print(f"Skipping pair {pair_name}: gradient data missing.")
            results[f'{pair_name}_term1'] = np.nan
            results[f'{pair_name}_term2'] = np.nan
            results[f'{pair_name}_term3'] = np.nan
            continue

        # Δ∇_{2,1} = ∇g2 - ∇g1
        # grad_g1 と grad_g2 の長さが一致することを期待
        if len(grad_g1) != len(grad_g2):
             print(f"Warning: Skipping pair {pair_name}: gradient list length mismatch.")
             # v_inv などと長さを合わせるため，短い方に切り捨てる
        
        min_len = min(len(grad_g1), len(grad_g2), len(v_inv))
        
        delta_nabla_21 = [grad_g2[i] - grad_g1[i] for i in range(min_len)]
        # v_inv なども長さを合わせる
        v_inv_trunc = v_inv[:min_len]
        v_Y_trunc = v_Y[:min_len]
        v_A_trunc = v_A[:min_len]
        v_AY_trunc = v_AY[:min_len]


        # --- 項(I) ---
        # <Δ∇_{2,1}, v_inv>
        term1 = _standard_inner_product(delta_nabla_21, v_inv_trunc)
        
        # --- 項(II) ---
        # v_term2 = y_bar*v_Y + a_bar*v_A + y_bar*a_bar*v_AY (v_C -> v_Y)
        v_term2 = [(y_bar * vy) + (a_bar * va) + (y_bar * a_bar * vay) 
                   for vy, va, vay in zip(v_Y_trunc, v_A_trunc, v_AY_trunc)]
        # <Δ∇_{2,1}, v_term2>
        term2 = _standard_inner_product(delta_nabla_21, v_term2)
        
        # --- 項(III) ---
        # v_term3 = ρ_train * v_AY
        v_term3 = [rho_train * vay for vay in v_AY_trunc]
        # <Δ∇_{2,1}, v_term3>
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
    グループごとのヤコビアンの期待値（平均埋め込み m_g(t)）を計算
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
        # サンプル数が指定数より多い場合はランダムサンプリング
        if len(X_group) > num_samples:
            indices = np.random.choice(len(X_group), num_samples, replace=False)
            X_subset = X_group[indices]
        else:
            X_subset = X_group

        # get_model_jacobian は m_g(t) = E[nabla_theta f(X)] を計算
        group_jacobians_list[(y_val, a_val)] = get_model_jacobian(model, X_subset, device)
    
    return group_jacobians_list

# ==============================================================================
# ヤコビアンノルムの分析
# ==============================================================================
def analyze_jacobian_norms(group_jacobians_list, dataset_type):
    """
    グループごとのヤコビアンの標準ノルムと標準内積，および
    幾何学的中心ベクトル (Delta S, Delta L) のノルムとアラインメントを計算
    (入力: 事前に計算されたヤコビアンのリスト)
    """
    print(f"\nAnalyzing JACOBIAN (standard) NORMS on {dataset_type} data...")
    group_keys = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    jacobian_results = {}
    group_norms_sq = {} # ノルムの2乗をキャッシュ

    # --- 1. 元のヤコビアンノルム ---
    for (y, a), jac_list in group_jacobians_list.items():
        key_name = f"norm_G({y},{a})"
        if jac_list is not None:
            # ||J||^2 = <J, J>
            norm_sq = _standard_inner_product(jac_list, jac_list)
            group_norms_sq[(y, a)] = norm_sq
            jacobian_results[key_name] = norm_sq # 2乗のまま保存
        else:
            group_norms_sq[(y, a)] = np.nan
            jacobian_results[key_name] = np.nan

    # --- 2. 元のヤコビアン内積とコサイン類似度 ---
    for (y1, a1), (y2, a2) in combinations(group_keys, 2):
        jac1_list = group_jacobians_list.get((y1, a1))
        jac2_list = group_jacobians_list.get((y2, a2))
        
        key_name_dot = f"dot_G({y1},{a1})_vs_G({y2},{a2})"
        key_name_cosine = f"cosine_G({y1},{a1})_vs_G({y2},{a2})"
        
        if jac1_list is not None and jac2_list is not None:
            # 内積を計算
            inner_prod = _standard_inner_product(jac1_list, jac2_list)
            jacobian_results[key_name_dot] = inner_prod
            
            # --- コサイン類似度の計算 ---
            norm1_sq = group_norms_sq.get((y1, a1), np.nan)
            norm2_sq = group_norms_sq.get((y2, a2), np.nan)
            
            if not np.isnan(norm1_sq) and not np.isnan(norm2_sq):
                norm1 = np.sqrt(norm1_sq)
                norm2 = np.sqrt(norm2_sq)
                if norm1 > 1e-9 and norm2 > 1e-9:
                    cosine_sim = inner_prod / (norm1 * norm2)
                    jacobian_results[key_name_cosine] = cosine_sim
                else:
                    jacobian_results[key_name_cosine] = np.nan # ゼロノルムの場合
            else:
                jacobian_results[key_name_cosine] = np.nan
            
        else:
            jacobian_results[key_name_dot] = np.nan
            jacobian_results[key_name_cosine] = np.nan
    
    # --- 3. 幾何学的中心ベクトルの分析 ---
    # Phi_g = Φ̄_g (グループgのヤコビアンの期待値)
    #
    # S_+1 (S_p1) = 1/2 * (Phi_(+1,+1) + Phi_(-1,+1))
    # S_-1 (S_m1) = 1/2 * (Phi_(+1,-1) + Phi_(-1,-1))
    # L_+1 (L_p1) = 1/2 * (Phi_(+1,+1) + Phi_(+1,-1))
    # L_-1 (L_m1) = 1/2 * (Phi_(-1,-1) + Phi_(-1,+1))
    #
    # Delta_S = S_+1 - S_-1 = 1/2 * (Phi_pp + Phi_np - Phi_pn - Phi_nn)
    # Delta_L = L_+1 - L_-1 = 1/2 * (Phi_pp + Phi_pn - Phi_nn - Phi_np)
    
    Phi_pp = group_jacobians_list.get((1, 1))
    Phi_pn = group_jacobians_list.get((1, -1))
    Phi_np = group_jacobians_list.get((-1, 1))
    Phi_nn = group_jacobians_list.get((-1, -1))

    # エラー時に NaN を設定するキーのリスト
    keys_to_nan = ["norm_Delta_S", "norm_Delta_L", "ratio_Delta_S_L", 
                   "dot_Delta_S_Delta_L", "cosine_Delta_S_Delta_L", 
                   "paper_norm_sq_m_A", "paper_norm_sq_m_Y", 
                   "paper_dot_m_A_m_Y"]

    # 必要なグループがすべて存在するかチェック
    if all(v is not None for v in [Phi_pp, Phi_pn, Phi_np, Phi_nn]):
        try:
            # パラメータリストの長さを取得 (すべて同じ長さと仮定)
            num_params = len(Phi_pp)
            if not all(len(v) == num_params for v in [Phi_pn, Phi_np, Phi_nn]):
                raise ValueError("Jacobian lists have mismatched parameter counts.")

            # Delta_S = 1/2 * ( (Phi_pp - Phi_pn) + (Phi_np - Phi_nn) )
            delta_S_list = [0.5 * (Phi_pp[i] - Phi_pn[i] + Phi_np[i] - Phi_nn[i]) for i in range(num_params)]
            
            # Delta_L = 1/2 * ( (Phi_pp - Phi_np) + (Phi_pn - Phi_nn) )
            delta_L_list = [0.5 * (Phi_pp[i] + Phi_pn[i] - Phi_np[i] - Phi_nn[i]) for i in range(num_params)]

            # ノルムの2乗 ||Delta S||^2 と ||Delta L||^2
            norm_sq_delta_S = _standard_inner_product(delta_S_list, delta_S_list)
            norm_sq_delta_L = _standard_inner_product(delta_L_list, delta_L_list)
            
            # ノルム ||Delta S|| と ||Delta L||
            norm_delta_S = np.sqrt(norm_sq_delta_S) if norm_sq_delta_S >= 0 else np.nan
            norm_delta_L = np.sqrt(norm_sq_delta_L) if norm_sq_delta_L >= 0 else np.nan
            
            jacobian_results["norm_Delta_S"] = norm_delta_S
            jacobian_results["norm_Delta_L"] = norm_delta_L
            
            # 比 (Ratio) の計算
            if not np.isnan(norm_delta_S) and not np.isnan(norm_delta_L) and norm_delta_L > 1e-9:
                ratio_Delta_S_L = norm_delta_S / norm_delta_L
            else:
                ratio_Delta_S_L = np.nan
            jacobian_results["ratio_Delta_S_L"] = ratio_Delta_S_L

            # 内積 <Delta S, Delta L>
            inner_prod_SL = _standard_inner_product(delta_S_list, delta_L_list)
            jacobian_results["dot_Delta_S_Delta_L"] = inner_prod_SL

            # コサイン類似度
            if norm_delta_S > 1e-9 and norm_delta_L > 1e-9:
                cosine_sim_SL = inner_prod_SL / (norm_delta_S * norm_delta_L)
            else:
                cosine_sim_SL = np.nan
            jacobian_results["cosine_Delta_S_Delta_L"] = cosine_sim_SL

            # ||m_A||^2 = 0.25 * ||Delta_S||^2
            # ||m_Y||^2 = 0.25 * ||Delta_L||^2
            # <m_A, m_Y> = 0.25 * <Delta_S, Delta_L>
            jacobian_results["paper_norm_sq_m_A"] = 0.25 * norm_sq_delta_S
            jacobian_results["paper_norm_sq_m_Y"] = 0.25 * norm_sq_delta_L
            jacobian_results["paper_dot_m_A_m_Y"] = 0.25 * inner_prod_SL

        except Exception as e:
            print(f"Error during geometric center analysis: {e}")
            for k in keys_to_nan:
                jacobian_results[k] = np.nan
    else:
        print(f"Skipping geometric center (Delta S, Delta L) analysis: missing group Jacobian data.")
        for k in keys_to_nan:
            jacobian_results[k] = np.nan
            
    return jacobian_results

# ==============================================================================
# 静的・動的分解の分析
# ==============================================================================
def analyze_static_dynamic_decomposition(group_grads_list, group_jacobians_list, config, y_data, a_data, dataset_type):
    """
    同じラベルを持つグループ間の性能差ダイナミクスの静的・動的分解 (項A, B, C) を計算
    group_grads_list = {g: nabla_g(t)}
    group_jacobians_list = {g: m_g(t)}
    """
    print(f"\nAnalyzing STATIC/DYNAMIC DECOMPOSITION on {dataset_type} data...")
    
    group_pairs_config = config.get('static_dynamic_decomposition', {}).get('group_pairs', [])
    if not group_pairs_config:
        print("No group pairs configured for static/dynamic decomposition. Skipping.")
        return {}

    loss_function = config['loss_function']
    
    # 1. c_0(y) の決定 (損失関数の原点での勾配)
    if loss_function == 'logistic':
        # c_0(y) = -y/2
        c0_map = {-1.0: 0.5, 1.0: -0.5}
    elif loss_function == 'mse':
        # c_0(y) = -2y
        c0_map = {-1.0: 2.0, 1.0: -2.0}
    else:
        print(f"Warning: c_0(y) not defined for loss '{loss_function}'. Skipping decomposition.")
        return {}
    
    # 2. グループごとの静的・動的勾配を計算
    group_keys = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    nabla_static = {}
    nabla_dynamic = {}
    
    first_valid_grad_list = next((g for g in group_grads_list.values() if g is not None), None)
    if first_valid_grad_list is None:
        print("No valid group gradients found.")
        return {}
    
    for g in group_keys:
        y_val, a_val = g
        nabla_g = group_grads_list.get(g) # 損失勾配 nabla_g(t)
        m_g = group_jacobians_list.get(g)  # ヤコビアンの期待値 m_g(t)
        
        if nabla_g is None or m_g is None:
            nabla_static[g] = None
            nabla_dynamic[g] = None
            continue
            
        min_len = min(len(nabla_g), len(m_g))
        nabla_g = nabla_g[:min_len]
        m_g = m_g[:min_len]
        
        # nabla_static[g] = c_0(y_g) * m_g(t)
        c0_val = c0_map[y_val]
        nabla_static[g] = [c0_val * m_p for m_p in m_g]
        
        # nabla_dynamic[g] = nabla_g(t) - nabla_static[g]
        nabla_dynamic[g] = [nabla_p - static_p for nabla_p, static_p in zip(nabla_g, nabla_static[g])]

    # 3. 訓練分布のモーメント pi_g を計算
    pi_g = {}
    total_samples = len(y_data)
    if total_samples == 0:
        print("Warning: 0 samples. Skipping train decomposition.")
        return {}
        
    for g in group_keys:
        y_val, a_val = g
        mask = (y_data == y_val) & (a_data == a_val)
        pi_g[g] = mask.sum().item() / total_samples

    # 4. 訓練勾配の静的・動的成分を計算
    # nabla_train^static = sum(pi_g * nabla_static[g])
    nabla_train_static = [np.zeros_like(g) for g in first_valid_grad_list]
    nabla_train_dynamic = [np.zeros_like(g) for g in first_valid_grad_list]
    
    min_len_train = len(nabla_train_static)

    for g in group_keys:
        pi_val = pi_g[g]
        if nabla_static[g] is not None:
            for i in range(min(min_len_train, len(nabla_static[g]))):
                nabla_train_static[i] += pi_val * nabla_static[g][i]
        
        if nabla_dynamic[g] is not None:
            for i in range(min(min_len_train, len(nabla_dynamic[g]))):
                nabla_train_dynamic[i] += pi_val * nabla_dynamic[g][i]

    # 5. ペアごとに項A, B, C を計算
    results = {}
    for pair in group_pairs_config:
        # config では [g_min, g_maj] の順で渡されると仮定
        g_min_key = tuple(pair[0])
        g_maj_key = tuple(pair[1])
        pair_name = f"g_min_{g_min_key}_g_maj_{g_maj_key}"
        
        # d/dt (R_min - R_maj) = <nabla_maj - nabla_min, nabla_train>
        # Delta_nabla_y = nabla_maj - nabla_min
        
        nabla_maj_static = nabla_static.get(g_maj_key)
        nabla_min_static = nabla_static.get(g_min_key)
        nabla_maj_dynamic = nabla_dynamic.get(g_maj_key)
        nabla_min_dynamic = nabla_dynamic.get(g_min_key)
        
        if nabla_maj_static is None or nabla_min_static is None or \
           nabla_maj_dynamic is None or nabla_min_dynamic is None:
            print(f"Skipping pair {pair_name}: missing gradient data.")
            results[f'{pair_name}_TermA'] = np.nan
            results[f'{pair_name}_TermB'] = np.nan
            results[f'{pair_name}_TermC'] = np.nan
            continue
        
        min_len = min(len(nabla_maj_static), len(nabla_min_static), 
                      len(nabla_maj_dynamic), len(nabla_min_dynamic),
                      len(nabla_train_static))
        
        # Delta_nabla_y^static = nabla_maj^static - nabla_min^static
        delta_nabla_static = [nabla_maj_static[i] - nabla_min_static[i] for i in range(min_len)]
        
        # Delta_nabla_y^dynamic = nabla_maj^dynamic - nabla_min^dynamic
        delta_nabla_dynamic = [nabla_maj_dynamic[i] - nabla_min_dynamic[i] for i in range(min_len)]
        
        nabla_train_static_trunc = [p for p in nabla_train_static[:min_len]]
        nabla_train_dynamic_trunc = [p for p in nabla_train_dynamic[:min_len]]

        # 項A = <Delta_nabla_y^static, nabla_train^static>
        term_A = _standard_inner_product(delta_nabla_static, nabla_train_static_trunc)
        
        # 項B = <Delta_nabla_y^static, nabla_train^dynamic> + <Delta_nabla_y^dynamic, nabla_train^static>
        term_B1 = _standard_inner_product(delta_nabla_static, nabla_train_dynamic_trunc)
        term_B2 = _standard_inner_product(delta_nabla_dynamic, nabla_train_static_trunc)
        term_B = term_B1 + term_B2
        
        # 項C = <Delta_nabla_y^dynamic, nabla_train^dynamic>
        term_C = _standard_inner_product(delta_nabla_dynamic, nabla_train_dynamic_trunc)

        results[f'{pair_name}_TermA'] = term_A
        results[f'{pair_name}_TermB'] = term_B
        results[f'{pair_name}_TermC'] = term_C
        results[f'{pair_name}_TermB1_static_dyn'] = term_B1 # 参考
        results[f'{pair_name}_TermB2_dyn_static'] = term_B2 # 参考
    
    return results


# ==============================================================================
# 全ての分析を統括するラッパー関数
# ==============================================================================
def run_all_analyses(config, epoch, layers, model, train_outputs, test_outputs,
                     X_train, y_train, a_train, X_test, y_test, a_test, histories,
                     history): 
    """設定に基づいてすべての分析を実行し，結果をhistory辞書に保存する"""
    analysis_target = config['analysis_target']

    # --- フラグの決定 ---
    run_grad_basis = config.get('analyze_gradient_basis', False)
    run_gap_factors = config.get('analyze_gap_dynamics_factors', False)
    run_jacobian_norm = config.get('analyze_jacobian_norm', False)
    run_static_dynamic = config.get('analyze_static_dynamic_decomposition', False)

    # --- グループ勾配の計算 (Basis, GapFactors, StaticDynamic が依存) ---
    group_grads_train, group_grads_test = None, None
    run_grad_basis_or_gap_factors_or_static_dynamic = run_grad_basis or run_gap_factors or run_static_dynamic
    
    if run_grad_basis_or_gap_factors_or_static_dynamic:
        num_samples_grad = config.get('gradient_gram_num_samples', None) 
        
        if analysis_target in ['train', 'both']:
            group_grads_train = calculate_group_gradients(
                model, X_train, y_train, a_train, config['device'], config['loss_function'], "Train",
                num_samples_grad
            )
        if analysis_target in ['test', 'both']:
            group_grads_test = calculate_group_gradients(
                model, X_test, y_test, a_test, config['device'], config['loss_function'], "Test",
                num_samples_grad
            )

    # --- グループヤコビアンの計算 (JacobianNorm, StaticDynamic が依存) ---
    group_jacobians_train, group_jacobians_test = None, None
    run_jacobian_or_static_dynamic = run_jacobian_norm or run_static_dynamic

    if run_jacobian_or_static_dynamic:
        num_samples_jac = config.get('jacobian_num_samples', 100)
        
        if analysis_target in ['train', 'both']:
            group_jacobians_train = calculate_group_jacobians(
                model, X_train, y_train, a_train, config['device'], 
                num_samples_jac, "Train")
        if analysis_target in ['test', 'both']:
            group_jacobians_test = calculate_group_jacobians(
                model, X_test, y_test, a_test, config['device'], 
                num_samples_jac, "Test")

    # --- 勾配基底ベクトルの分析 ---
    basis_vectors_train, basis_vectors_test = None, None
    if run_grad_basis:
        if group_grads_train:
            basis_results_train = analyze_gradient_basis(group_grads_train, "Train")
            basis_vectors_train = basis_results_train.pop('vectors', None)
            histories['grad_basis_train'][epoch] = basis_results_train
        if group_grads_test:
            basis_results_test = analyze_gradient_basis(group_grads_test, "Test")
            basis_vectors_test = basis_results_test.pop('vectors', None)
            histories['grad_basis_test'][epoch] = basis_results_test

    # --- 性能差ダイナミクスの要因の分析 ---
    if run_gap_factors:
        # この分析は 'analyze_gradient_basis' がTrueでなくても実行される可能性がある
        # その場合，basis_vectors が None になるので，ここで再計算する
        if group_grads_train and basis_vectors_train is None:
             basis_vectors_train = analyze_gradient_basis(group_grads_train, "Train").get('vectors')
        if group_grads_test and basis_vectors_test is None:
             basis_vectors_test = analyze_gradient_basis(group_grads_test, "Test").get('vectors')
        
        if group_grads_train and basis_vectors_train:
            histories['gap_factors_train'][epoch] = analyze_gap_dynamics_factors(
                group_grads_train, basis_vectors_train, config, y_train, a_train, "Train")
        if group_grads_test and basis_vectors_test:
            histories['gap_factors_test'][epoch] = analyze_gap_dynamics_factors(
                group_grads_test, basis_vectors_test, config, y_test, a_test, "Test")

    # --- ヤコビアンノルムの分析 ---
    if run_jacobian_norm:
        if group_jacobians_train:
            histories['jacobian_norm_train'][epoch] = analyze_jacobian_norms(
                group_jacobians_train, "Train")
        if group_jacobians_test:
            histories['jacobian_norm_test'][epoch] = analyze_jacobian_norms(
                group_jacobians_test, "Test")

    # --- 静的・動的分解の分析 ---
    if run_static_dynamic:
        if group_grads_train and group_jacobians_train:
            histories['static_dynamic_decomp_train'][epoch] = analyze_static_dynamic_decomposition(
                group_grads_train, group_jacobians_train, config, y_train, a_train, "Train")
        if group_grads_test and group_jacobians_test:
            histories['static_dynamic_decomp_test'][epoch] = analyze_static_dynamic_decomposition(
                group_grads_test, group_jacobians_test, config, y_test, a_test, "Test")
