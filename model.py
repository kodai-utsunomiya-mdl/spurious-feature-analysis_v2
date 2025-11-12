# sp/model.py

import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, activation_fn='relu', use_skip_connections=False, initialization_method='muP'):
        """
        MLPモデルの定義
        Args:
            input_dim (int): 入力次元数
            hidden_dim (int): 隠れ層の次元数 (m)
            num_hidden_layers (int): 隠れ層の数
            activation_fn (str): 活性化関数名 ('relu', 'gelu', 'tanh', 'identity')
            use_skip_connections (bool): Skip Connectionを使用するかどうかのフラグ
            initialization_method (str): パラメータ化の手法 ('muP', 'NTP')
        """
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1.")
        
        if initialization_method not in ['muP', 'NTP']:
             raise ValueError(f"Unknown initialization_method: {initialization_method}. Only 'muP' and 'NTP' are supported.")

        self.use_skip_connections = use_skip_connections
        self.initialization_method = initialization_method
        self.hidden_dim = float(hidden_dim) # スケーリングのために float に (m)
        self.num_hidden_layers = num_hidden_layers
        self.layers = nn.ModuleList()

        # 入力層
        self.layers.append(nn.Linear(input_dim, int(self.hidden_dim), bias=False))
        # 隠れ層
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(int(self.hidden_dim), int(self.hidden_dim), bias=False))

        self.classifier = nn.Linear(int(self.hidden_dim), 1, bias=False)

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
        m = self.hidden_dim
        sqrt_m = np.sqrt(m)

        # --- 1. 入力層 (W^1) ---
        z = self.layers[0](z)
        if self.initialization_method == 'muP':
            # muP: f(x) = ... φ( √m * W^1 * x ) ...
            z = z * sqrt_m
        # NTP: f_NTK(x) = ... φ( W^1 * x ) ... (スケーリングなし)
        
        z = self.activation(z)
        outputs['layer_1'] = z

        # --- 2. 隠れ層 (W^l, l=2...L-1) ---
        # num_hidden_layers が 1 の場合はループしない (len(self.layers) == 1)
        for i in range(1, len(self.layers)): 
            identity = z  # Skip Connectionのための入力を保持

            # 線形変換
            z_pre_activation = self.layers[i](z)

            if self.initialization_method == 'NTP':
                 # NTP: f_NTK(x) = ... φ( (1/√m) * W^l * ... )
                z_pre_activation = z_pre_activation / sqrt_m
            # muP: f(x) = ... φ( W^l * ... ) ... (スケーリングなし)

            # Skip Connectionを適用 (設定が有効な場合)
            if self.use_skip_connections:
                z_pre_activation = z_pre_activation + identity

            # 活性化関数
            z = self.activation(z_pre_activation)
            outputs[f'layer_{i+1}'] = z

        # --- 3. 出力層 (w^L) ---
        # f(x) = (1/√m) * w^L * ... (muP, NTP共通)
        output_scalar = self.classifier(z).squeeze(-1)
        output_scalar = output_scalar / sqrt_m
        
        outputs['logit'] = output_scalar

        return output_scalar, outputs

def apply_manual_parametrization(model, method, hidden_dim, fix_final_layer=False):
    """
    モデルの重みを初期化する．
    Args:
        model (nn.Module): 対象のMLPモデル
        method (str): 'muP', 'NTP' のいずれか
        hidden_dim (int): 隠れ層の幅 (m)
        fix_final_layer (bool): 最終層の重みを固定するかのフラグ
    """
    m = float(hidden_dim)

    if method not in ['muP', 'NTP']:
        raise ValueError(f"Unknown initialization_method: {method}. Only 'muP' and 'NTP' are supported.")

    print(f"\nApplying manual initialization for '{method}'...")
    print(f"        - Hidden Dim (m): {hidden_dim}")
    if fix_final_layer:
        print("        - Final layer weights will be FROZEN.")

    with torch.no_grad():
        
        # --- 初期化分散 ---
        if method == 'muP':
            # 各要素を N(0, 1/m) で初期化
            init_var_input = 1.0 / m
            init_var_hidden = 1.0 / m
            init_var_output = 1.0 / m
            print(f"        - Using muP initialization: N(0, 1/m) for all layers.")
        else: # NTP
            # 各要素を N(0, 1) で初期化
            init_var_input = 1.0
            init_var_hidden = 1.0
            init_var_output = 1.0
            print(f"        - Using NTP initialization: N(0, 1) for all layers.")

        # --- 入力層 (W^1) ---
        std_input = np.sqrt(init_var_input)
        model.layers[0].weight.normal_(0, std_input)
        print(f"        - Input Layer (W^1):        Init Var = {init_var_input:.2e}")

        # --- 隠れ層 (W^l, l=2...L-1) ---
        std_hidden = np.sqrt(init_var_hidden)
        # len(model.layers) は隠れ層の総数 (入力層 + 中間層)
        for i in range(1, len(model.layers)):
            hidden_layer = model.layers[i]
            hidden_layer.weight.normal_(0, std_hidden)
            print(f"        - Hidden Layer {i+1} (W^{i+1}): Init Var = {init_var_hidden:.2e}")

        # --- 出力層 (w^L) ---
        std_output = np.sqrt(init_var_output)
        output_layer = model.classifier
        output_layer.weight.normal_(0, std_output)
        
        if not fix_final_layer:
            print(f"        - Output Layer (w^L):     Init Var = {init_var_output:.2e}")
        else:
            output_layer.weight.requires_grad = False
            print(f"        - Output Layer (w^L):     Init Var = {init_var_output:.2e} (Fixed)")

    return
