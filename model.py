# sp/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""_muP (TP5論文 Table 9)_

|  | Input weights & all biases | Output weights | Hidden weights |
| --- | --- | --- | --- |
| Init. Var. | 1/fan_out | 1/fan_in | 1/fan_in |
| Multiplier | √fan_out | 1/√fan_in | 1 |
| SGD LR | 1 | 1 | 1 |
| Adam LR | 1/√fan_out | 1/√fan_in | 1/fan_in |

"""

class ParametrizedLinear(nn.Module):
    """
    重みとバイアスを定義し，特定のmultiplierと初期化分散でparametrizationを行う線形層．
    TP5論文の "Table 9: muP Formulation in the Style of [57]..." の形式に従う．
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
        特定の標準偏差でパラメータを初期化する
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
    def __init__(self, input_dim, hidden_dim, 
                 num_residual_blocks=None, num_hidden_layers=None,
                 activation_fn='relu', 
                 use_skip_connections=False, initialization_method='muP', 
                 use_bias=False, use_zero_bias_init=False):
        """
        MLPモデルの定義
        Args:
            input_dim (int): 入力次元数
            hidden_dim (int): 隠れ層の次元数 (m)
            num_residual_blocks (int, optional): 残差ブロックの数 L (use_skip_connections=True の場合に使用)
            num_hidden_layers (int, optional): 隠れ層の総数 H (use_skip_connections=False の場合に使用)
            activation_fn (str): 活性化関数名 ('relu', 'gelu', 'tanh', 'identity')
            use_skip_connections (bool): Skip Connectionを使用するかどうかのフラグ
            initialization_method (str): パラメータ化の手法 ('muP', 'NTP')
            use_bias (bool): バイアス項を使用するかどうかのフラグ
            use_zero_bias_init (bool): バイアスを0で初期化するかどうかのフラグ (True: 0, False: ランダム)
        """
        super().__init__()
        
        if initialization_method not in ['muP', 'NTP']:
             raise ValueError(f"Unknown initialization_method: {initialization_method}. Only 'muP' and 'NTP' are supported.")

        self.use_skip_connections = use_skip_connections
        self.initialization_method = initialization_method
        self.hidden_dim = float(hidden_dim) # スケーリングのために float に (m)
        self.use_bias = use_bias
        self.use_zero_bias_init = use_zero_bias_init

        # 構成の決定 (ResNet/MLP)
        # additional_layers_count: Input Layerの後に追加するブロック・層の数
        # ResNet (Skipあり): L = num_residual_blocks
        # MLP (Skipなし)   : H-1 = num_hidden_layers - 1 
        if self.use_skip_connections:
            # ResNet
            if num_residual_blocks is None:
                if num_hidden_layers is not None:
                    print(f"[Warning] use_skip_connections=True but num_residual_blocks is None. Using num_hidden_layers ({num_hidden_layers}) as L.")
                    self.num_blocks = num_hidden_layers
                else:
                    raise ValueError("num_residual_blocks must be specified when use_skip_connections=True.")
            else:
                self.num_blocks = num_residual_blocks
            
            additional_layers_count = self.num_blocks
            self.model_type = "ResNet"
            depth_scaling_L = self.num_blocks
            
        else:
            # MLP
            if num_hidden_layers is None:
                raise ValueError("num_hidden_layers must be specified when use_skip_connections=False.")
            
            if num_hidden_layers < 1:
                raise ValueError("num_hidden_layers must be at least 1.")
            
            self.total_hidden_layers = num_hidden_layers
            additional_layers_count = self.total_hidden_layers - 1
            self.model_type = "MLP"
            depth_scaling_L = 1

        self.depth_scaling_L = depth_scaling_L

        # スケーリング係数と初期化分散の計算
        m = self.hidden_dim
        sqrt_m = np.sqrt(m)
        
        print(f"\nInitializing {self.model_type} with method: '{initialization_method}'")
        print(f"        - Hidden Dim (m): {int(m)}")
        if self.model_type == "ResNet":
            print(f"        - Depth L (num_residual_blocks): {self.num_blocks} (Total Hidden Layers: {1 + self.num_blocks})")
        else:
            print(f"        - Total Hidden Layers (H): {self.total_hidden_layers} (Structure: Raw Input + {1 + additional_layers_count} Layers)")

        if use_bias:
            init_type = "Zero" if use_zero_bias_init else "Random"
            print(f"        - Bias: Enabled (Initialization: {init_type})")

        # --- 各層の設定 (Multiplier & Init Std) ---
        
        # 1. 入力層 (Input Layer / Projection W_in)
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
        else: # NTP
            in_w_mult = 1.0
            in_b_mult = 1.0
            in_w_init_std = 1.0
            in_b_init_std = 1.0
            in_bias_flag = use_bias

        # バイアスの初期化stdを0に上書き
        if self.use_zero_bias_init:
            in_b_init_std = 0.0

        if initialization_method == 'muP':
            print(f"        - Input Layer:  W mult={in_w_mult:.2f}, W init std={in_w_init_std:.2e}, b init std={in_b_init_std:.2e}")
        else:
            print(f"        - Input Layer:  W mult={in_w_mult:.2f}, W init std={in_w_init_std:.2e}, b init std={in_b_init_std:.2e}")

        self.input_layer = ParametrizedLinear(
            input_dim, int(m), 
            bias=in_bias_flag, 
            weight_mult=in_w_mult, 
            bias_mult=in_b_mult,
            weight_init_std=in_w_init_std,
            bias_init_std=in_b_init_std
        )

        # 2. 隠れ層 (Hidden Layers / Residual Blocks)
        # muP (TP5 + TP6 Depth): 
        #   - W mult: 1.0 (TP5 Table 9)
        #   - b mult: sqrt(m) (TP5 Table 9)
        #   - Depth Multiplier (Branch): 1/sqrt(L) (TP6) -> forwardで適用
        #   - W init: N(0, 1/m) -> std = 1/sqrt(m)
        #   - b init: N(0, 1/m) -> std = 1/sqrt(m)
        # NTP: 
        #   - W mult: 1/sqrt(m)
        #   - W init: N(0, 1) -> std = 1.0
        self.hidden_layers = nn.ModuleList()

        # --- Depth-muP Scaling の計算 ---
        self.depth_mult = 1.0
        if initialization_method == 'muP' and use_skip_connections and depth_scaling_L > 0:
            self.depth_mult = 1.0 / np.sqrt(depth_scaling_L)
            print(f"        - Depth-muP: Scaling residual branches by 1/sqrt(L) = 1/sqrt({depth_scaling_L}) = {self.depth_mult:.4f}")
        
        if initialization_method == 'muP':
            hid_w_mult = 1.0
            hid_b_mult = sqrt_m

            hid_w_init_std = 1.0 / sqrt_m
            hid_b_init_std = 1.0 / sqrt_m
            hid_bias_flag = use_bias
        else: # NTP
            hid_w_mult = 1.0 / sqrt_m
            hid_b_mult = 1.0
            hid_w_init_std = 1.0
            hid_b_init_std = 1.0
            hid_bias_flag = use_bias

        # バイアスの初期化stdを0に上書き
        if self.use_zero_bias_init:
            hid_b_init_std = 0.0

        if additional_layers_count > 0:
            if initialization_method == 'muP':
                print(f"        - Hidden Layers: W mult={hid_w_mult:.2f}, W init std={hid_w_init_std:.2e}, b init std={hid_b_init_std:.2e}")
            else:
                print(f"        - Hidden Layers: W mult={hid_w_mult:.2e}, W init std={hid_w_init_std:.2e}, b init std={hid_b_init_std:.2e}")

        # 層を追加
        for _ in range(additional_layers_count):
            self.hidden_layers.append(ParametrizedLinear(
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
        else: # NTP
            out_w_mult = 1.0 / sqrt_m
            out_b_mult = 1.0
            out_w_init_std = 1.0
            out_b_init_std = 1.0
            out_bias_flag = use_bias
            
        # バイアスの初期化stdを0に上書き
        if self.use_zero_bias_init:
            out_b_init_std = 0.0

        if initialization_method == 'muP':
            print(f"        - Output Layer: W mult={out_w_mult:.2e}, W init std={out_w_init_std:.2e}, b init std={out_b_init_std:.2e}")
        else:
            print(f"        - Output Layer: W mult={out_w_mult:.2e}, W init std={out_w_init_std:.2e}, b init std={out_b_init_std:.2e}")
            
        self.classifier = ParametrizedLinear(
            int(m), 1,
            bias=out_bias_flag,
            weight_mult=out_w_mult,
            bias_mult=out_b_mult,
            weight_init_std=out_w_init_std,
            bias_init_std=out_b_init_std
        )

        # 活性化関数
        if activation_fn == 'relu':
            self.activation = nn.ReLU()
        elif activation_fn == 'gelu':
            self.activation = nn.GELU()
        elif activation_fn == 'tanh':
            self.activation = nn.Tanh()
        elif activation_fn == 'identity':
            self.activation = nn.Identity()
        elif activation_fn == 'silu':
            self.activation = nn.SiLU()
        elif activation_fn == 'softplus':  # Smoothed ReLU
            self.activation = nn.Softplus()
        elif activation_fn == 'abs':
            self.activation = torch.abs
        elif activation_fn == 'elu':
            self.activation = nn.ELU()
        elif activation_fn == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_fn == 'logsigmoid':
            self.activation = nn.LogSigmoid()
        elif activation_fn == 'exp':
            self.activation = torch.exp
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")


    def forward(self, x):
        outputs = {}
        z = x.view(x.shape[0], -1)
        
        # --- 0. Input Layer (Raw Input) ---
        outputs['layer_0'] = z 

        # --- 1. 最初の隠れ層 (Projection / First Hidden) ---
        # z = W_in x + b
        # ResNet: Y_0 = W_in x
        # MLP: Hidden 1 = W_in x
        z = self.input_layer(z)

        # ResNetの場合，残差ブロックへの入力は Post-activation (h^1)
        # h^1 = phi(tilde_h^1)
        if self.use_skip_connections:
            z = self.activation(z)

        outputs['layer_1'] = z
        
        # --- 2. 追加の隠れ層 ---
        # ResNet (TP6): Y_l = Y_{l-1} + 1/sqrt(L) * MS(phi(W_l Y_{l-1}))
        # MLP: h_l = W_l phi(h_{l-1})
        for i, layer in enumerate(self.hidden_layers): 
            identity = z 
            
            if self.use_skip_connections:
                # [ResNet] TP6 Structure: Post-Nonlin
                # 入力 z は既に post-activation (h^{l-1})

                # 1. Linear Transform (W * h^{l-1})
                branch = layer(z) 
                
                # 2. Activation
                branch = self.activation(branch)
                
                # 3. Mean Subtraction (TP6)
                # Feature Diversityを維持するため
                if self.initialization_method == 'muP':
                     branch = branch - branch.mean(dim=1, keepdim=True)

                # 4. Depth Scaling (Branch Multiplier)
                # Apply 1/sqrt(L) to the branch
                if self.initialization_method == 'muP':
                    branch = branch * self.depth_mult

                # 5. Skip Connection
                # h^l = h^{l-1} + branch
                z = identity + branch
                
            else:
                # [MLP] Standard Pre-Nonlin
                # 入力 z は pre-activation (tilde_h^{l-1})
                z_act = self.activation(z) 
                branch = layer(z_act)
                z = branch 
            
            # i=0 (追加層1つ目) -> layer_2
            outputs[f'layer_{i+2}'] = z

        # --- 3. 出力層 (Output Layer) ---
        if self.use_skip_connections:
            # ResNetの場合，z は既に post-activation (h^{L-1}) なのでそのまま分類器に入力 (w^T h^{L-1})
            z_final = z
        else:
            # [MLP] z は pre-activation (tilde_h^{L-1}) なので
            # 活性化関数を通してから分類器に入力する
            z_final = self.activation(z)

        output_scalar = self.classifier(z_final).squeeze(-1)
        outputs['logit'] = output_scalar

        return output_scalar, outputs

    def get_optimizer_parameters(self, optimizer_name, global_lr):
        """
        オプティマイザに渡すパラメータグループを作成する．
        muPかつAdamの場合のみ層ごとの学習率スケーリングを行う．
        """
        if self.initialization_method == 'NTP' or optimizer_name == 'SGD':
            return [{'params': self.parameters(), 'lr': global_lr}]
        
        # --- muP + Adam の場合 ---
        groups = []
        m = self.hidden_dim
        sqrt_m = np.sqrt(m)
        
        # Adamにおける深さ方向のLRスケーリング (1/sqrt(L))
        depth_lr_scale = 1.0
        if self.use_skip_connections and self.depth_scaling_L > 0:
            depth_lr_scale = 1.0 / np.sqrt(self.depth_scaling_L)
        
        print(f"Configuring Adam parameters for muP (m={int(m)})...")
        if depth_lr_scale != 1.0:
            print(f"  - Depth-muP: Scaling hidden layer LRs by 1/sqrt(L) = {depth_lr_scale:.4f}")

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
        # Weights (fan_in=m): lr_scale = 1/fan_in = 1/m  (* 1/sqrt(L) for Depth-muP)
        # Biases (fan_out=m): lr_scale = 1/sqrt(fan_out) = 1/sqrt(m) (* 1/sqrt(L) for Depth-muP)
        hidden_weights = []
        hidden_biases = []
        
        for layer in self.hidden_layers:
            hidden_weights.append(layer.weight)
            if layer.bias is not None:
                hidden_biases.append(layer.bias)
        
        if hidden_weights:
            # Depth-muP requires Adam LR to be scaled by 1/sqrt(L) for residual branches
            hid_w_scale = (1.0 / m) * depth_lr_scale
            groups.append({
                'params': hidden_weights, 
                'lr': global_lr * hid_w_scale, 
                'name': 'hidden_weights'
            })
            print(f"  - Hidden Weights:     LR scale = 1/m * 1/sqrt(L) = {hid_w_scale:.6f}")

        if hidden_biases:
            # Depth-muP requires Adam LR to be scaled by 1/sqrt(L)
            hid_b_scale = (1.0 / sqrt_m) * depth_lr_scale
            groups.append({
                'params': hidden_biases, 
                'lr': global_lr * hid_b_scale, 
                'name': 'hidden_biases'
            })
            print(f"  - Hidden Biases:      LR scale = 1/sqrt(m) * 1/sqrt(L) = {hid_b_scale:.6f}")

        # 3. Output Layer
        #   - Output weights: Adam LR = 1/sqrt(fan_in)
        #   - Output biases: Adam LR = 1/sqrt(fan_out)
        
        # 3-a. Output Weights
        # fan_in = m -> lr_scale = 1/sqrt(m)
        output_const = 1.0 # 50.0
        output_w_scale = output_const / sqrt_m
        groups.append({
            'params': [self.classifier.weight], 
            'lr': global_lr * output_w_scale, 
            'name': 'output_layer_weights'
        })
        print(f"  - Output Layer Weights: LR scale = 1/sqrt(m) = {output_w_scale:.6f}")

        # 3-b. Output Biases
        # fan_out = 1 -> lr_scale = 1/sqrt(1) = 1.0
        if self.classifier.bias is not None:
            output_b_scale = 1.0
            groups.append({
                'params': [self.classifier.bias], 
                'lr': global_lr * output_b_scale, 
                'name': 'output_layer_biases'
            })
            print(f"  - Output Layer Biases:  LR scale = 1.0")
        
        return groups

























# # sp/model.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# # ==========================================
# # [実験用設定] ボトルネック層の次元
# # 反例作成実験のために，最終隠れ層の出力次元をここで制御します．
# # config.yaml の hidden_dim (1024) は第1層の幅として維持され，
# # ここで指定した次元が第2層(最終隠れ層)の幅となります．
# # ==========================================
# EXPERIMENTAL_BOTTLENECK_DIM = 4
# # ==========================================


# """_muP (TP5論文 Table 9)_

# |  | Input weights & all biases | Output weights | Hidden weights |
# | --- | --- | --- | --- |
# | Init. Var. | 1/fan_out | 1/fan_in | 1/fan_in |
# | Multiplier | √fan_out | 1/√fan_in | 1 |
# | SGD LR | 1 | 1 | 1 |
# | Adam LR | 1/√fan_out | 1/√fan_in | 1/fan_in |

# """

# class ParametrizedLinear(nn.Module):
#     """
#     重みとバイアスを定義し，特定のmultiplierと初期化分散でparametrizationを行う線形層．
#     TP5論文の "Table 9: muP Formulation in the Style of [57]..." の形式に従う．
#     """
#     def __init__(self, in_features, out_features, bias=True, 
#                  weight_mult=1.0, bias_mult=1.0, 
#                  weight_init_std=1.0, bias_init_std=1.0):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight_mult = weight_mult
#         self.bias_mult = bias_mult
#         self.weight_init_std = weight_init_std
#         self.bias_init_std = bias_init_std
        
#         # 重みパラメータの定義 (Shape: out x in)
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
#         # バイアスパラメータの定義 (Shape: out)
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)
            
#         # パラメータの初期化を実行
#         self.reset_parameters()
            
#     def reset_parameters(self):
#         """
#         特定の標準偏差でパラメータを初期化する
#         """
#         # 重みの初期化: N(0, weight_init_std^2)
#         nn.init.normal_(self.weight, mean=0.0, std=self.weight_init_std)
        
#         # バイアスの初期化: N(0, bias_init_std^2)
#         if self.bias is not None:
#             nn.init.normal_(self.bias, mean=0.0, std=self.bias_init_std)

#     def forward(self, input):
#         # Linear計算: input @ weight.T
#         # F.linear(input, weight) は input @ weight.T を行う
#         out = F.linear(input, self.weight, None)
        
#         # 重みのスケーリング適用
#         if self.weight_mult != 1.0:
#             out = out * self.weight_mult
            
#         # バイアスの加算とスケーリング
#         if self.bias is not None:
#             out = out + (self.bias * self.bias_mult)
            
#         return out
        
#     def extra_repr(self):
#         return (f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, '
#                 f'weight_mult={self.weight_mult}, bias_mult={self.bias_mult}, '
#                 f'weight_init_std={self.weight_init_std:.2e}, bias_init_std={self.bias_init_std:.2e}')

# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, 
#                  num_residual_blocks=None, num_hidden_layers=None,
#                  activation_fn='relu', 
#                  use_skip_connections=False, initialization_method='muP', 
#                  use_bias=False, use_zero_bias_init=False,
#                  bottleneck_dim=EXPERIMENTAL_BOTTLENECK_DIM): # 実験用引数を追加
#         """
#         MLPモデルの定義
#         Args:
#             input_dim (int): 入力次元数
#             hidden_dim (int): 隠れ層の次元数 (m) - 第1層の幅として使用
#             num_residual_blocks (int, optional): 残差ブロックの数 L
#             num_hidden_layers (int, optional): 隠れ層の総数 H
#             activation_fn (str): 活性化関数名
#             use_skip_connections (bool): Skip Connectionを使用するかどうかのフラグ
#             initialization_method (str): パラメータ化の手法 ('muP', 'NTP')
#             use_bias (bool): バイアス項を使用するかどうかのフラグ
#             use_zero_bias_init (bool): バイアスを0で初期化するかどうかのフラグ
#             bottleneck_dim (int): 最終隠れ層の出力次元 (実験用ボトルネック)
#         """
#         super().__init__()
        
#         if initialization_method not in ['muP', 'NTP']:
#              raise ValueError(f"Unknown initialization_method: {initialization_method}. Only 'muP' and 'NTP' are supported.")

#         self.use_skip_connections = use_skip_connections
#         self.initialization_method = initialization_method
#         self.hidden_dim = float(hidden_dim) # m (Base width)
#         self.bottleneck_dim = float(bottleneck_dim) if bottleneck_dim is not None else self.hidden_dim
#         self.use_bias = use_bias
#         self.use_zero_bias_init = use_zero_bias_init

#         # 構成の決定 (ResNet/MLP)
#         if self.use_skip_connections:
#             # ResNet
#             if num_residual_blocks is None:
#                 if num_hidden_layers is not None:
#                     print(f"[Warning] use_skip_connections=True but num_residual_blocks is None. Using num_hidden_layers ({num_hidden_layers}) as L.")
#                     self.num_blocks = num_hidden_layers
#                 else:
#                     raise ValueError("num_residual_blocks must be specified when use_skip_connections=True.")
#             else:
#                 self.num_blocks = num_residual_blocks
            
#             additional_layers_count = self.num_blocks
#             self.model_type = "ResNet"
#             depth_scaling_L = self.num_blocks
            
#             # ResNetでボトルネックを使うと次元不整合が起きるため警告
#             if self.bottleneck_dim != self.hidden_dim:
#                 print(f"[WARNING] Bottleneck dim ({int(self.bottleneck_dim)}) != Hidden dim ({int(self.hidden_dim)}) in ResNet.")
#                 print("          Skip connections will likely fail due to dimension mismatch!")
            
#         else:
#             # MLP
#             if num_hidden_layers is None:
#                 raise ValueError("num_hidden_layers must be specified when use_skip_connections=False.")
            
#             if num_hidden_layers < 1:
#                 raise ValueError("num_hidden_layers must be at least 1.")
            
#             self.total_hidden_layers = num_hidden_layers
#             additional_layers_count = self.total_hidden_layers - 1
#             self.model_type = "MLP"
#             depth_scaling_L = 1

#         self.depth_scaling_L = depth_scaling_L

#         # スケーリング係数と初期化分散の計算 (Base width m に基づく)
#         m = self.hidden_dim
#         sqrt_m = np.sqrt(m)
        
#         print(f"\nInitializing {self.model_type} with method: '{initialization_method}'")
#         print(f"        - Base Hidden Dim (m): {int(m)}")
#         print(f"        - Bottleneck Dim (Final Hidden Output): {int(self.bottleneck_dim)}")
        
#         if self.model_type == "ResNet":
#             print(f"        - Depth L (num_residual_blocks): {self.num_blocks} (Total Hidden Layers: {1 + self.num_blocks})")
#         else:
#             print(f"        - Total Hidden Layers (H): {self.total_hidden_layers} (Structure: Raw Input + {1 + additional_layers_count} Layers)")

#         if use_bias:
#             init_type = "Zero" if use_zero_bias_init else "Random"
#             print(f"        - Bias: Enabled (Initialization: {init_type})")

#         # --- 1. 入力層 (Input Layer / Projection W_in) ---
#         # Input -> m
#         if initialization_method == 'muP':
#             in_w_mult = sqrt_m
#             in_b_mult = sqrt_m
#             in_w_init_std = 1.0 / sqrt_m
#             in_b_init_std = 1.0 / sqrt_m
#             in_bias_flag = use_bias
#         else: # NTP
#             in_w_mult = 1.0
#             in_b_mult = 1.0
#             in_w_init_std = 1.0
#             in_b_init_std = 1.0
#             in_bias_flag = use_bias

#         if self.use_zero_bias_init:
#             in_b_init_std = 0.0

#         print(f"        - Input Layer:  W mult={in_w_mult:.2f}, W init std={in_w_init_std:.2e}")

#         self.input_layer = ParametrizedLinear(
#             input_dim, int(m), 
#             bias=in_bias_flag, 
#             weight_mult=in_w_mult, 
#             bias_mult=in_b_mult,
#             weight_init_std=in_w_init_std,
#             bias_init_std=in_b_init_std
#         )

#         # --- 2. 隠れ層 (Hidden Layers) ---
#         self.hidden_layers = nn.ModuleList()

#         # Depth-muP Scaling
#         self.depth_mult = 1.0
#         if initialization_method == 'muP' and use_skip_connections and depth_scaling_L > 0:
#             self.depth_mult = 1.0 / np.sqrt(depth_scaling_L)
        
#         # Hidden Layerの基本設定 (m x m を想定)
#         if initialization_method == 'muP':
#             hid_w_mult = 1.0
#             hid_b_mult = sqrt_m
#             hid_w_init_std = 1.0 / sqrt_m
#             hid_b_init_std = 1.0 / sqrt_m
#             hid_bias_flag = use_bias
#         else: # NTP
#             hid_w_mult = 1.0 / sqrt_m
#             hid_b_mult = 1.0
#             hid_w_init_std = 1.0
#             hid_b_init_std = 1.0
#             hid_bias_flag = use_bias

#         if self.use_zero_bias_init:
#             hid_b_init_std = 0.0

#         # 層を追加 (ボトルネック対応)
#         current_dim = int(m)
        
#         for i in range(additional_layers_count):
#             # 最後の隠れ層の場合，出力次元を bottleneck_dim にする
#             is_last_hidden = (i == additional_layers_count - 1)
            
#             # ResNetの場合は次元を変えるとSkip Connectionが壊れるので変えない (警告済み)
#             if self.use_skip_connections:
#                 next_dim = int(m)
#             else:
#                 next_dim = int(self.bottleneck_dim) if is_last_hidden else int(m)
            
#             print(f"        - Hidden Layer {i+1}: {current_dim} -> {next_dim}")
            
#             self.hidden_layers.append(ParametrizedLinear(
#                 current_dim, next_dim,
#                 bias=hid_bias_flag,
#                 weight_mult=hid_w_mult,
#                 bias_mult=hid_b_mult,
#                 weight_init_std=hid_w_init_std,
#                 bias_init_std=hid_b_init_std
#             ))
#             current_dim = next_dim

#         # --- 3. 出力層 (Output Layer) ---
#         # Classifierの入力次元は current_dim (つまり bottleneck_dim) になっている
#         clf_in_dim = current_dim
        
#         # muP係数の再計算 (入力次元が変わるため)
#         # Output weights: W mult = 1/sqrt(fan_in), Init std = 1/sqrt(fan_in)
#         sqrt_fan_in = np.sqrt(clf_in_dim)
        
#         if initialization_method == 'muP':
#             out_w_mult = 1.0 / sqrt_fan_in
#             out_b_mult = 1.0
#             out_w_init_std = 1.0 / sqrt_fan_in
#             out_b_init_std = 1.0
#             out_bias_flag = use_bias
#         else: # NTP
#             out_w_mult = 1.0 / sqrt_fan_in
#             out_b_mult = 1.0
#             out_w_init_std = 1.0
#             out_b_init_std = 1.0
#             out_bias_flag = use_bias
            
#         if self.use_zero_bias_init:
#             out_b_init_std = 0.0

#         print(f"        - Output Layer: In={clf_in_dim} -> Out=1")
#         print(f"                        W mult={out_w_mult:.2e}, W init std={out_w_init_std:.2e}")
            
#         self.classifier = ParametrizedLinear(
#             clf_in_dim, 1,
#             bias=out_bias_flag,
#             weight_mult=out_w_mult,
#             bias_mult=out_b_mult,
#             weight_init_std=out_w_init_std,
#             bias_init_std=out_b_init_std
#         )

#         # 活性化関数
#         if activation_fn == 'relu':
#             self.activation = nn.ReLU()
#         elif activation_fn == 'gelu':
#             self.activation = nn.GELU()
#         elif activation_fn == 'tanh':
#             self.activation = nn.Tanh()
#         elif activation_fn == 'identity':
#             self.activation = nn.Identity()
#         elif activation_fn == 'silu':
#             self.activation = nn.SiLU()
#         elif activation_fn == 'softplus':
#             self.activation = nn.Softplus()
#         elif activation_fn == 'abs':
#             self.activation = torch.abs
#         else:
#             raise ValueError(f"Unknown activation function: {activation_fn}")


#     def forward(self, x):
#         outputs = {}
#         z = x.view(x.shape[0], -1)
        
#         # --- 0. Input Layer (Raw Input) ---
#         outputs['layer_0'] = z 

#         # --- 1. 最初の隠れ層 (Projection / First Hidden) ---
#         z = self.input_layer(z)
#         outputs['layer_1'] = z 

#         # --- 2. 追加の隠れ層 ---
#         for i, layer in enumerate(self.hidden_layers): 
#             identity = z 
            
#             if self.use_skip_connections:
#                 # [ResNet]
#                 branch = layer(z) 
#                 branch = self.activation(branch)
#                 if self.initialization_method == 'muP':
#                      branch = branch - branch.mean(dim=1, keepdim=True)
#                 if self.initialization_method == 'muP':
#                     branch = branch * self.depth_mult
#                 z = identity + branch
#             else:
#                 # [MLP]
#                 z_act = self.activation(z) 
#                 branch = layer(z_act)
#                 z = branch 
            
#             # i=0 (追加層1つ目) -> layer_2
#             outputs[f'layer_{i+2}'] = z

#         # --- 3. 出力層 (Output Layer) ---
#         # 最後に活性化関数を通してから分類器へ
#         z_final = self.activation(z)
#         output_scalar = self.classifier(z_final).squeeze(-1)
#         outputs['logit'] = output_scalar

#         return output_scalar, outputs

#     def get_optimizer_parameters(self, optimizer_name, global_lr):
#         """
#         オプティマイザに渡すパラメータグループを作成する．
#         muPかつAdamの場合のみ層ごとの学習率スケーリングを行う．
#         """
#         if self.initialization_method == 'NTP' or optimizer_name == 'SGD':
#             return [{'params': self.parameters(), 'lr': global_lr}]
        
#         # --- muP + Adam の場合 ---
#         groups = []
#         m = self.hidden_dim
#         sqrt_m = np.sqrt(m)
        
#         # Adamにおける深さ方向のLRスケーリング
#         depth_lr_scale = 1.0
#         if self.use_skip_connections and self.depth_scaling_L > 0:
#             depth_lr_scale = 1.0 / np.sqrt(self.depth_scaling_L)
        
#         print(f"Configuring Adam parameters for muP (Base m={int(m)})...")

#         # 1. Input Layer
#         input_lr_scale = 1.0 / sqrt_m
#         input_params = []
#         input_params.append(self.input_layer.weight)
#         if self.input_layer.bias is not None:
#             input_params.append(self.input_layer.bias)
        
#         groups.append({
#             'params': input_params, 
#             'lr': global_lr * input_lr_scale, 
#             'name': 'input_layer (W, b)'
#         })
        
#         # 2. Hidden Layers
#         # Hidden weights: LR scale = 1/m (入力次元がmの場合)
#         # ここでは簡略化のため，全てのHidden Layerの入力次元が m であると仮定してスケーリングする．
#         # (ボトルネック層への入力も m なので，重み行列のサイズは K x m だが，fan_in=m なので 1/m で正しい)
#         hidden_weights = []
#         hidden_biases = []
        
#         for layer in self.hidden_layers:
#             hidden_weights.append(layer.weight)
#             if layer.bias is not None:
#                 hidden_biases.append(layer.bias)
        
#         if hidden_weights:
#             hid_w_scale = (1.0 / m) * depth_lr_scale
#             groups.append({
#                 'params': hidden_weights, 
#                 'lr': global_lr * hid_w_scale, 
#                 'name': 'hidden_weights'
#             })

#         if hidden_biases:
#             hid_b_scale = (1.0 / sqrt_m) * depth_lr_scale
#             groups.append({
#                 'params': hidden_biases, 
#                 'lr': global_lr * hid_b_scale, 
#                 'name': 'hidden_biases'
#             })

#         # 3. Output Layer
#         # Output weights: Adam LR = 1/sqrt(fan_in)
#         # ここでの fan_in は bottleneck_dim
#         clf_in_dim = self.classifier.in_features
#         sqrt_clf_in = np.sqrt(clf_in_dim)
        
#         output_w_scale = 1.0 / sqrt_clf_in
#         groups.append({
#             'params': [self.classifier.weight], 
#             'lr': global_lr * output_w_scale, 
#             'name': 'output_layer_weights'
#         })
#         print(f"  - Output Layer Weights (fan_in={clf_in_dim}): LR scale = 1/sqrt({clf_in_dim}) = {output_w_scale:.6f}")

#         if self.classifier.bias is not None:
#             output_b_scale = 1.0
#             groups.append({
#                 'params': [self.classifier.bias], 
#                 'lr': global_lr * output_b_scale, 
#                 'name': 'output_layer_biases'
#             })
        
#         return groups
