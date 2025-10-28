# sp/model.py

import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, activation_fn='relu', use_skip_connections=False, initialization_method='SP'):
        """
        MLPモデルの定義
        Args:
            input_dim (int): 入力次元数
            hidden_dim (int): 隠れ層の次元数
            num_hidden_layers (int): 隠れ層の数
            activation_fn (str): 活性化関数名 ('relu', 'gelu', 'tanh', 'identity')
            use_skip_connections (bool): Skip Connectionを使用するかどうかのフラグ
            initialization_method (str): パラメータ化の手法 ('SP', 'NTP', 'muP', 'mf')
        """
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1.")

        self.use_skip_connections = use_skip_connections
        self.initialization_method = initialization_method
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()

        # 入力層
        self.layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        # 隠れ層
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))

        self.classifier = nn.Linear(hidden_dim, 1, bias=False)

        if activation_fn == 'relu':
            self.activation = nn.ReLU()
        elif activation_fn == 'gelu':
            self.activation = nn.GELU()
        elif activation_fn == 'tanh':
            self.activation = nn.Tanh()
        elif activation_fn == 'identity':
            self.activation = nn.Identity()
        elif activation_fn == 'silu':  # SiLU (Sigmoid Linear Unit)
            self.activation = nn.SiLU()
        elif activation_fn == 'softplus':  # Smoothed ReLU (Softplus)
            self.activation = nn.Softplus()
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")


    def forward(self, x):
        outputs = {}
        z = x.view(x.shape[0], -1)

        # 最初の層 (Skip Connectionなし)
        z = self.layers[0](z)
        z = self.activation(z)
        outputs['layer_1'] = z

        # 2層目以降の隠れ層
        for i in range(1, len(self.layers)):
            identity = z  # Skip Connectionのための入力を保持

            # 線形変換
            z_pre_activation = self.layers[i](z)

            # Skip Connectionを適用 (設定が有効な場合)
            if self.use_skip_connections:
                z_pre_activation = z_pre_activation + identity

            # 活性化関数
            z = self.activation(z_pre_activation)
            outputs[f'layer_{i+1}'] = z

        # 分類器
        output_scalar = self.classifier(z).squeeze(-1)

        # 'mf' (mean-field) の場合，出力層で 1/n のスケーリングを適用
        if self.initialization_method == 'mf':
            output_scalar = output_scalar / float(self.hidden_dim)

        outputs['logit'] = output_scalar

        return output_scalar, outputs

def apply_manual_parametrization(model, method, base_lr, hidden_dim, input_dim, fix_final_layer=False):
    """
    Args:
        model (nn.Module): 対象のMLPモデル
        method (str): 'SP', 'NTP', 'muP', 'mf'のいずれか
        base_lr (float): 基本学習率 (η)
        hidden_dim (int): 隠れ層の幅 (n)
        input_dim (int): 入力次元 (d)
        fix_final_layer (bool): 最終層の重みを固定するかのフラグ

    Returns:
        list: オプティマイザに渡すためのパラメータグループのリスト
    """
    n = float(hidden_dim)
    d = float(input_dim)
    optimizer_param_groups = []

    print(f"\nApplying manual parametrization for '{method}'...")
    print(f"        - Input Dim (d): {input_dim}, Hidden Dim (n): {hidden_dim}, Base LR (η): {base_lr}")
    if fix_final_layer:
        print("        - Final layer weights are FROZEN.")

    if method == 'mf' and len(model.layers) != 1:
        raise ValueError("The 'mf' (mean-field) parametrization is only supported for models with exactly one hidden layer.")


    with torch.no_grad():
        # --- 入力層 ---
        # Note: 入力データがL2正規化されているため，dによるスケーリングは削除済み
        input_layer = model.layers[0]
        const = 1.0 # 2.0

        if method == 'SP':
            init_var = const
            lr = base_lr / n
        elif method == 'NTP':
            init_var = const
            lr = base_lr
        elif method == 'muP':
            init_var = const
            lr = base_lr * n
        elif method == 'mf':
            init_var = const
            lr = base_lr * n
        else:
            raise ValueError(f"Unknown method: {method}")

        std = np.sqrt(init_var)
        input_layer.weight.normal_(0, std)
        optimizer_param_groups.append({'params': input_layer.weight, 'lr': lr})
        print(f"        - Input Layer (W^1):        Init Var = {init_var:.2f}, LR = {lr:.2e}")

        # --- 隠れ層 ---
        for i in range(1, len(model.layers)):
            hidden_layer = model.layers[i]

            if method == 'SP':
                init_var = const / n
                lr = base_lr / n
            elif method == 'NTP':
                init_var = const / n
                lr = base_lr / n
            elif method == 'muP':
                init_var = const / n
                lr = base_lr
            else:
                raise ValueError(f"Unknown method: {method}")

            std = np.sqrt(init_var)
            hidden_layer.weight.normal_(0, std)
            optimizer_param_groups.append({'params': hidden_layer.weight, 'lr': lr})
            print(f"        - Hidden Layer {i+1} (W^{i+1}): Init Var = 1/{int(n)}, LR = {lr:.2e}")

        # --- 出力層 ---
        output_layer = model.classifier

        if method == 'SP':
            init_var = const / n
            lr = base_lr / n
        elif method == 'NTP':
            init_var = const / n
            lr = base_lr / n
        elif method == 'muP':
            init_var = const / (n**2)
            lr = base_lr / n
        elif method == 'mf':
            init_var = const
            lr = base_lr * n
        else:
            raise ValueError(f"Unknown method: {method}")

        std = np.sqrt(init_var)
        output_layer.weight.normal_(0, std)
        
        if not fix_final_layer:
            optimizer_param_groups.append({'params': output_layer.weight, 'lr': lr})
            if method == 'mf':
                print(f"        - Output Layer (W^L+1): Init Var = {init_var:.2f}, LR = {lr:.2e}")
            else:
                print(f"        - Output Layer (W^L+1): Init Var = 1/{int(n*n) if method=='muP' else int(n)}, LR = {lr:.2e}")
        else:
            output_layer.weight.requires_grad = False
            if method == 'mf':
                print(f"        - Output Layer (W^L+1): Init Var = {init_var:.2f}, LR = 0.00 (Fixed)")
            else:
                print(f"        - Output Layer (W^L+1): Init Var = 1/{int(n*n) if method=='muP' else int(n)}, LR = 0.00 (Fixed)")


    return optimizer_param_groups
