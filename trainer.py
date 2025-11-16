# sp/trainer.py

import torch
import numpy as np
from utils import get_loss_function

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

    # --- 学習ステップの分岐 ---
    if debias_method == 'IW_uniform':
        # --- IW (Uniform Target) の学習ステップ (フルバッチ・グループ別勾配) ---
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
            scores_g, _ = model(X_g)
            # utils からインポートした関数を呼び出す (reduction='mean' がデフォルト)
            loss_g = get_loss_function(scores_g, y_g, loss_function_name)
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

        # 3. パラメータ更新
        optimizer.step()

    elif debias_method == 'GroupDRO':
        # --- Group DRO 学習ステップ (フルバッチ・グループ別勾配) ---
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
            scores_g, _ = model(X_g)
            # utils からインポートした関数を呼び出す (reduction='mean' がデフォルト)
            loss_g = get_loss_function(scores_g, y_g, loss_function_name)
            
            group_losses_tensor[i] = loss_g.detach() # 損失を保存
            
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

        # 4. パラメータ更新
        optimizer.step()

    elif debias_method == 'None':
        # --- 通常のERM学習ステップ (ミニバッチ) ---
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            scores, _ = model(X_batch)
            loss = get_loss_function(scores, y_batch, loss_function_name)
            loss.backward()
            optimizer.step()
    # --- 分岐終了 ---

    # GroupDROの場合，更新された重みを返す
    if debias_method == 'GroupDRO':
        return dro_q_weights
    else:
        return None # 他のメソッドではNoneを返す
