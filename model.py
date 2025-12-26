# # sp/model.py

# import torch
# import torch.nn as nn
# import numpy as np

# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_hidden_layers, activation_fn='relu', use_skip_connections=False, initialization_method='muP'):
#         """
#         MLPモデルの定義
#         Args:
#             input_dim (int): 入力次元数
#             hidden_dim (int): 隠れ層の次元数 (m)
#             num_hidden_layers (int): 隠れ層の数
#             activation_fn (str): 活性化関数名 ('relu', 'gelu', 'tanh', 'identity')
#             use_skip_connections (bool): Skip Connectionを使用するかどうかのフラグ
#             initialization_method (str): パラメータ化の手法 ('muP', 'NTP')
#         """
#         super().__init__()
#         if num_hidden_layers < 1:
#             raise ValueError("num_hidden_layers must be at least 1.")
        
#         if initialization_method not in ['muP', 'NTP']:
#              raise ValueError(f"Unknown initialization_method: {initialization_method}. Only 'muP' and 'NTP' are supported.")

#         self.use_skip_connections = use_skip_connections
#         self.initialization_method = initialization_method
#         self.hidden_dim = float(hidden_dim) # スケーリングのために float に (m)
#         self.num_hidden_layers = num_hidden_layers
#         self.layers = nn.ModuleList()

#         # 入力層
#         self.layers.append(nn.Linear(input_dim, int(self.hidden_dim), bias=False))
#         # 隠れ層
#         for _ in range(num_hidden_layers - 1):
#             self.layers.append(nn.Linear(int(self.hidden_dim), int(self.hidden_dim), bias=False))

#         self.classifier = nn.Linear(int(self.hidden_dim), 1, bias=False)

#         if activation_fn == 'relu':
#             self.activation = nn.ReLU()
#         elif activation_fn == 'gelu':
#             self.activation = nn.GELU()
#         elif activation_fn == 'tanh':
#             self.activation = nn.Tanh()
#         elif activation_fn == 'identity':
#             self.activation = nn.Identity()
#         elif activation_fn == 'silu':  # SiLU (Sigmoid Linear Unit)
#             self.activation = nn.SiLU()
#         elif activation_fn == 'softplus':  # Smoothed ReLU (Softplus)
#             self.activation = nn.Softplus()
#         else:
#             raise ValueError(f"Unknown activation function: {activation_fn}")


#     def forward(self, x):
#         outputs = {}
#         z = x.view(x.shape[0], -1)
#         m = self.hidden_dim
#         sqrt_m = np.sqrt(m)

#         # --- 1. 入力層 (W^1) ---
#         z = self.layers[0](z)
#         if self.initialization_method == 'muP':
#             # muP: f(x) = ... φ( √m * W^1 * x ) ...
#             z = z * sqrt_m
#         # NTP: f_NTK(x) = ... φ( W^1 * x ) ... (スケーリングなし)
        
#         z = self.activation(z)
#         outputs['layer_1'] = z

#         # --- 2. 隠れ層 (W^l, l=2...L-1) ---
#         # num_hidden_layers が 1 の場合はループしない (len(self.layers) == 1)
#         for i in range(1, len(self.layers)): 
#             identity = z  # Skip Connectionのための入力を保持

#             # 線形変換
#             z_pre_activation = self.layers[i](z)

#             if self.initialization_method == 'NTP':
#                  # NTP: f_NTK(x) = ... φ( (1/√m) * W^l * ... )
#                 z_pre_activation = z_pre_activation / sqrt_m
#             # muP: f(x) = ... φ( W^l * ... ) ... (スケーリングなし)

#             # Skip Connectionを適用 (設定が有効な場合)
#             if self.use_skip_connections:
#                 z_pre_activation = z_pre_activation + identity

#             # 活性化関数
#             z = self.activation(z_pre_activation)
#             outputs[f'layer_{i+1}'] = z

#         # --- 3. 出力層 (w^L) ---
#         # f(x) = (1/√m) * w^L * ... (muP, NTP共通)
#         output_scalar = self.classifier(z).squeeze(-1)
#         output_scalar = output_scalar / sqrt_m
        
#         outputs['logit'] = output_scalar

#         return output_scalar, outputs

# def apply_manual_parametrization(model, method, hidden_dim, fix_final_layer=False):
#     """
#     モデルの重みを初期化する．
#     Args:
#         model (nn.Module): 対象のMLPモデル
#         method (str): 'muP', 'NTP' のいずれか
#         hidden_dim (int): 隠れ層の幅 (m)
#         fix_final_layer (bool): 最終層の重みを固定するかのフラグ
#     """
#     m = float(hidden_dim)

#     if method not in ['muP', 'NTP']:
#         raise ValueError(f"Unknown initialization_method: {method}. Only 'muP' and 'NTP' are supported.")

#     print(f"\nApplying manual initialization for '{method}'...")
#     print(f"        - Hidden Dim (m): {hidden_dim}")
#     if fix_final_layer:
#         print("        - Final layer weights will be FROZEN.")

#     with torch.no_grad():
        
#         # --- 初期化分散 ---
#         if method == 'muP':
#             # 各要素を N(0, 1/m) で初期化
#             init_var_input = 1.0 / m
#             init_var_hidden = 1.0 / m
#             init_var_output = 1.0 / m
#             print(f"        - Using muP initialization: N(0, 1/m) for all layers.")
#         else: # NTP
#             # 各要素を N(0, 1) で初期化
#             init_var_input = 1.0
#             init_var_hidden = 1.0
#             init_var_output = 1.0
#             print(f"        - Using NTP initialization: N(0, 1) for all layers.")

#         # --- 入力層 (W^1) ---
#         std_input = np.sqrt(init_var_input)
#         model.layers[0].weight.normal_(0, std_input)
#         print(f"        - Input Layer (W^1):        Init Var = {init_var_input:.2e}")

#         # --- 隠れ層 (W^l, l=2...L-1) ---
#         std_hidden = np.sqrt(init_var_hidden)
#         # len(model.layers) は隠れ層の総数 (入力層 + 中間層)
#         for i in range(1, len(model.layers)):
#             hidden_layer = model.layers[i]
#             hidden_layer.weight.normal_(0, std_hidden)
#             print(f"        - Hidden Layer {i+1} (W^{i+1}): Init Var = {init_var_hidden:.2e}")

#         # --- 出力層 (w^L) ---
#         std_output = np.sqrt(init_var_output)
#         output_layer = model.classifier
#         output_layer.weight.normal_(0, std_output)
        
#         if not fix_final_layer:
#             print(f"        - Output Layer (w^L):     Init Var = {init_var_output:.2e}")
#         else:
#             output_layer.weight.requires_grad = False
#             print(f"        - Output Layer (w^L):     Init Var = {init_var_output:.2e} (Fixed)")

#     return






# sp/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CustomLinear(nn.Module):
    """
    重みとバイアスを直接定義し、指定されたmultiplierと初期化分散でParametrizationを行う線形層
    """
    def __init__(self, in_features, out_features, bias=True, 
                 weight_mult=1.0, bias_mult=1.0, 
                 weight_init_std=1.0, bias_init_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mult = weight_mult
        self.bias_mult = bias_mult
        self.weight_init_std = weight_init_std
        self.bias_init_std = bias_init_std
        
        # 重みパラメータの定義 (Shape: out x in)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # バイアスパラメータの定義 (Shape: out)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # パラメータの初期化を実行
        self.reset_parameters()
            
    def reset_parameters(self):
        """
        指定された標準偏差でパラメータを初期化する
        """
        # 重みの初期化: N(0, weight_init_std^2)
        nn.init.normal_(self.weight, mean=0.0, std=self.weight_init_std)
        
        # バイアスの初期化: N(0, bias_init_std^2)
        if self.bias is not None:
            nn.init.normal_(self.bias, mean=0.0, std=self.bias_init_std)

    def forward(self, input):
        # Linear計算: input @ weight.T
        # F.linear(input, weight) は input @ weight.T を行う
        out = F.linear(input, self.weight, None)
        
        # 重みのスケーリング適用
        if self.weight_mult != 1.0:
            out = out * self.weight_mult
            
        # バイアスの加算とスケーリング
        if self.bias is not None:
            out = out + (self.bias * self.bias_mult)
            
        return out
        
    def extra_repr(self):
        return (f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, '
                f'weight_mult={self.weight_mult}, bias_mult={self.bias_mult}, '
                f'weight_init_std={self.weight_init_std:.2e}, bias_init_std={self.bias_init_std:.2e}')

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, activation_fn='relu', use_skip_connections=False, initialization_method='muP', use_bias=False):
        """
        MLPモデルの定義
        Args:
            input_dim (int): 入力次元数
            hidden_dim (int): 隠れ層の次元数 (m)
            num_hidden_layers (int): 隠れ層の数
            activation_fn (str): 活性化関数名 ('relu', 'gelu', 'tanh', 'identity')
            use_skip_connections (bool): Skip Connectionを使用するかどうかのフラグ
            initialization_method (str): パラメータ化の手法 ('muP', 'NTP')
            use_bias (bool): バイアス項を使用するかどうかのフラグ
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
        self.use_bias = use_bias
        
        # スケーリング係数と初期化分散の計算
        m = self.hidden_dim
        sqrt_m = np.sqrt(m)
        
        print(f"\nInitializing MLP with method: '{initialization_method}'")
        print(f"        - Hidden Dim (m): {int(m)}")
        if use_bias:
            print(f"        - Bias: Enabled")

        # --- 各層の設定 (Multiplier & Init Std) ---
        
        # 1. 入力層 (Input Layer)
        # muP: 
        #   - W mult: sqrt(m)
        #   - b mult: sqrt(m)
        #   - W init: N(0, 1/m) -> std = 1/sqrt(m)
        #   - b init: N(0, 1/m) -> std = 1/sqrt(m)
        # NTP: 
        #   - W mult: 1.0
        #   - W init: N(0, 1) -> std = 1.0
        if initialization_method == 'muP':
            in_w_mult = sqrt_m
            in_b_mult = sqrt_m
            in_w_init_std = 1.0 / sqrt_m
            in_b_init_std = 1.0 / sqrt_m
            in_bias_flag = use_bias
            print(f"        - Input Layer:  W mult={in_w_mult:.2f}, W init std={in_w_init_std:.2e} (Var=1/m)")
        else: # NTP
            in_w_mult = 1.0
            in_b_mult = 1.0
            in_w_init_std = 1.0
            in_b_init_std = 1.0
            in_bias_flag = False
            print(f"        - Input Layer:  W mult={in_w_mult:.2f}, W init std={in_w_init_std:.2e} (Var=1)")

        self.input_layer = CustomLinear(
            input_dim, int(m), 
            bias=in_bias_flag, 
            weight_mult=in_w_mult, 
            bias_mult=in_b_mult,
            weight_init_std=in_w_init_std,
            bias_init_std=in_b_init_std
        )

        # 2. 隠れ層 (Hidden Layers)
        # muP: 
        #   - W mult: 1.0
        #   - b mult: sqrt(m)
        #   - W init: N(0, 1/m) -> std = 1/sqrt(m)
        #   - b init: N(0, 1/m) -> std = 1/sqrt(m)
        # NTP: 
        #   - W mult: 1/sqrt(m)
        #   - W init: N(0, 1) -> std = 1.0
        self.hidden_layers = nn.ModuleList()
        
        if initialization_method == 'muP':
            hid_w_mult = 1.0
            hid_b_mult = sqrt_m
            hid_w_init_std = 1.0 / sqrt_m
            hid_b_init_std = 1.0 / sqrt_m
            hid_bias_flag = use_bias
            print(f"        - Hidden Layers: W mult={hid_w_mult:.2f}, W init std={hid_w_init_std:.2e} (Var=1/m)")
        else: # NTP
            hid_w_mult = 1.0 / sqrt_m
            hid_b_mult = 1.0
            hid_w_init_std = 1.0
            hid_b_init_std = 1.0
            hid_bias_flag = False
            print(f"        - Hidden Layers: W mult={hid_w_mult:.2e}, W init std={hid_w_init_std:.2e} (Var=1)")

        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(CustomLinear(
                int(m), int(m),
                bias=hid_bias_flag,
                weight_mult=hid_w_mult,
                bias_mult=hid_b_mult,
                weight_init_std=hid_w_init_std,
                bias_init_std=hid_b_init_std
            ))

        # 3. 出力層 (Output Layer)
        # muP: 
        #   - W mult: 1/sqrt(m)
        #   - b mult: 1.0 (sqrt(fan_out)=1)
        #   - W init: N(0, 1/m) -> std = 1/sqrt(m)
        #   - b init: N(0, 1) -> std = 1.0 (fan_out=1)
        # NTP: 
        #   - W mult: 1/sqrt(m)
        #   - W init: N(0, 1) -> std = 1.0
        if initialization_method == 'muP':
            out_w_mult = 1.0 / sqrt_m
            out_b_mult = 1.0
            out_w_init_std = 1.0 / sqrt_m
            out_b_init_std = 1.0
            out_bias_flag = use_bias
            print(f"        - Output Layer: W mult={out_w_mult:.2e}, W init std={out_w_init_std:.2e} (Var=1/m)")
        else: # NTP
            out_w_mult = 1.0 / sqrt_m
            out_b_mult = 1.0
            out_w_init_std = 1.0
            out_b_init_std = 1.0
            out_bias_flag = False
            print(f"        - Output Layer: W mult={out_w_mult:.2e}, W init std={out_w_init_std:.2e} (Var=1)")
            
        self.classifier = CustomLinear(
            int(m), 1,
            bias=out_bias_flag,
            weight_mult=out_w_mult,
            bias_mult=out_b_mult,
            weight_init_std=out_w_init_std,
            bias_init_std=out_b_init_std
        )

        # 活性化関数の設定
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

        # --- 1. 入力層 ---
        # CustomLinear内でスケーリングが行われるため、ここでは単に呼び出すだけ
        z = self.input_layer(z)
        z = self.activation(z)
        outputs['layer_1'] = z

        # --- 2. 隠れ層 ---
        # 隠れ層リストは input_layer の次から始まるため、インデックスに注意
        # loop index i=0 -> layer name "layer_2"
        for i, layer in enumerate(self.hidden_layers): 
            identity = z 

            # 線形変換 (スケーリング込み)
            z_pre_activation = layer(z)

            # Skip Connection
            if self.use_skip_connections:
                z_pre_activation = z_pre_activation + identity

            # 活性化関数
            z = self.activation(z_pre_activation)
            outputs[f'layer_{i+2}'] = z

        # --- 3. 出力層 ---
        # CustomLinear内でスケーリング (W/sqrt_m など) が行われる
        output_scalar = self.classifier(z).squeeze(-1)
        outputs['logit'] = output_scalar

        return output_scalar, outputs

    def get_optimizer_parameters(self, optimizer_name, global_lr):
        """
        オプティマイザに渡すパラメータグループを作成する．
        muPかつAdamの場合のみ層ごとの学習率スケーリングを行う．
        """
        if self.initialization_method == 'NTP' or optimizer_name == 'SGD':
            # スケーリングなし: すべてのパラメータに対してglobal_lrを使用
            return [{'params': self.parameters(), 'lr': global_lr}]
        
        # --- muP + Adam の場合: 層ごとのLR設定 ---
        groups = []
        m = self.hidden_dim
        sqrt_m = np.sqrt(m)
        
        print(f"Configuring Adam parameters for muP (m={int(m)})...")

        # 1. Input Layer
        # Weights (fan_out=m): lr_scale = 1/sqrt(fan_out) = 1/sqrt(m)
        # Biases (fan_out=m): lr_scale = 1/sqrt(fan_out) = 1/sqrt(m)
        input_lr_scale = 1.0 / sqrt_m
        input_params = []
        input_params.append(self.input_layer.weight)
        if self.input_layer.bias is not None:
            input_params.append(self.input_layer.bias)
        
        groups.append({
            'params': input_params, 
            'lr': global_lr * input_lr_scale, 
            'name': 'input_layer (W, b)'
        })
        print(f"  - Input Layer (W, b): LR scale = 1/sqrt(m) = {input_lr_scale:.6f}")
        
        # 2. Hidden Layers
        # Weights (fan_in=m): lr_scale = 1/fan_in = 1/m
        # Biases (fan_out=m): lr_scale = 1/sqrt(fan_out) = 1/sqrt(m)
        hidden_weights = []
        hidden_biases = []
        
        for layer in self.hidden_layers:
            hidden_weights.append(layer.weight)
            if layer.bias is not None:
                hidden_biases.append(layer.bias)
        
        if hidden_weights:
            hid_w_scale = 1.0 / m
            groups.append({
                'params': hidden_weights, 
                'lr': global_lr * hid_w_scale, 
                'name': 'hidden_weights'
            })
            print(f"  - Hidden Weights:     LR scale = 1/m       = {hid_w_scale:.6f}")

        if hidden_biases:
            hid_b_scale = 1.0 / sqrt_m
            groups.append({
                'params': hidden_biases, 
                'lr': global_lr * hid_b_scale, 
                'name': 'hidden_biases'
            })
            print(f"  - Hidden Biases:      LR scale = 1/sqrt(m) = {hid_b_scale:.6f}")

        # 3. Output Layer
        # Weights (fan_out=1): lr_scale = 1/sqrt(fan_out) = 1.0
        # Biases (fan_out=1): lr_scale = 1/sqrt(fan_out) = 1.0
        output_params = []
        output_params.append(self.classifier.weight)
        if self.classifier.bias is not None:
            output_params.append(self.classifier.bias)
        
        output_scale = 1.0
        groups.append({
            'params': output_params, 
            'lr': global_lr * output_scale, 
            'name': 'output_layer (W, b)'
        })
        print(f"  - Output Layer (W, b): LR scale = 1.0")
        
        return groups
