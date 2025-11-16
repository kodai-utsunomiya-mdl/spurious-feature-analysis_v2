# sp/feature_extractor.py

import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
from torchvision import transforms
import numpy as np

# torchvision.models.feature_extraction をインポート
try:
    from torchvision.models.feature_extraction import create_feature_extractor
except ImportError:
    print("Warning: torchvision.models.feature_extraction not found. "
          "ViT intermediate layer extraction might not work.")
    create_feature_extractor = None

# --- DINOv2 / ViT の両方に対応する汎用ラッパー ---
class ViTFeatureExtractor(torch.nn.Module):
    def __init__(self, vit_model, model_name, target_block_index=-1, aggregation_mode='cls_token'):
        """
        ViT/DINOv2 モデルから指定されたブロックの特徴を抽出するラッパー
        Args:
            vit_model (nn.Module): ベースとなる ViT または DINOv2 モデル
            model_name (str): モデル名 ('ViT_B_16', 'DINOv2_ViT_S_14' など)
            target_block_index (int): 抽出対象のブロックインデックス (0-based). -1 は最後のブロック.
            aggregation_mode (str): 集約方法 ('cls_token', 'mean_pool_patch', 'mean_pool_all')
        """
        super().__init__()
        self.vit_model = vit_model
        self.model_name = model_name
        self.aggregation_mode = aggregation_mode

        # --- モデルタイプとブロック構造の特定 ---
        if 'DINOv2' in self.model_name:
            self.model_type = 'dinov2'
            self.blocks = self.vit_model.blocks
            self.num_blocks = len(self.blocks)
        elif 'ViT' in self.model_name:
            self.model_type = 'torchvision'
            self.blocks = self.vit_model.encoder.layers
            self.num_blocks = len(self.blocks)
        else:
            raise ValueError(f"Unknown ViT model name: {model_name}. Cannot determine structure.")

        # --- ターゲットブロックのインデックスを解決 ---
        if target_block_index < 0:
            self.target_block_index = self.num_blocks + target_block_index
        else:
            self.target_block_index = target_block_index

        if not (0 <= self.target_block_index < self.num_blocks):
            raise ValueError(f"Invalid target_block_index: {target_block_index}. "
                             f"Model has {self.num_blocks} blocks (0 to {self.num_blocks-1}).")

        self.is_last_block = (self.target_block_index == self.num_blocks - 1)

        print(f"ViTFeatureExtractor ({self.model_type}) configured:")
        print(f"  Target Block Index: {self.target_block_index} "
              f"({'Last Block' if self.is_last_block else 'Intermediate Block'})")
        print(f"  Aggregation Mode: {self.aggregation_mode}")

        # --- 抽出方法のセットアップ ---
        
        # DINOv2 の場合
        if self.model_type == 'dinov2':
            # 最後のブロック かつ 'cls_token' or 'mean_pool_patch' の場合
            if self.is_last_block and (self.aggregation_mode in ['cls_token', 'mean_pool_patch']):
                self.extraction_method = 'dinov2_forward_features'
                print(f"  Using DINOv2 'forward_features()' (final norm output)")
            
            # それ以外 (途中のブロック or mean_pool_all) の場合
            else:
                self.extraction_method = 'dinov2_get_intermediate'
                # 'get_intermediate_layers' は 1-indexed (n=1 は最後のブロック)
                self.n_blocks_to_return = self.num_blocks - self.target_block_index
                print(f"  Using DINOv2 'get_intermediate_layers(n={self.n_blocks_to_return})' (pre-norm output)")

        # torchvision ViT の場合
        elif self.model_type == 'torchvision':
            if create_feature_extractor is None:
                raise ImportError("torchvision.models.feature_extraction is required for ViT intermediate layers.")
            
            self.extraction_method = 'torchvision_feature_extractor'
            
            # ターゲットノード名を決定
            if self.is_last_block:
                # 最後のブロックの場合，encoder.ln の出力
                target_node_name = 'encoder.ln'
            else:
                # 途中のブロック (e.g., block 10)
                target_node_name = f'encoder.layers.encoder_layer_{self.target_block_index}.add_1'
            
            print(f"  Using torchvision.feature_extraction, targeting node: '{target_node_name}'")
            self.feature_extractor = create_feature_extractor(
                self.vit_model, return_nodes=[target_node_name]
            )
            self.target_node_name = target_node_name


    def forward(self, x):
        block_output = None

        # --- 1. 特徴テンソル (N, 1+L, D) を取得 ---
        if self.extraction_method == 'dinov2_forward_features':
            # DINOv2 の最終層 (正規化済み)
            outputs_dict = self.vit_model.forward_features(x)
            if self.aggregation_mode == 'cls_token':
                return outputs_dict['x_norm_clstoken'] # (N, D)
            elif self.aggregation_mode == 'mean_pool_patch':
                return outputs_dict['x_norm_patchtokens'].mean(dim=1) # (N, D)
            # (このパスでは mean_pool_all はサポートされない)
        
        elif self.extraction_method == 'dinov2_get_intermediate':
            # DINOv2 の中間層 (正規化なし)
            intermediate_list = self.vit_model.get_intermediate_layers(
                x, n=self.n_blocks_to_return, return_class_token=False
            )
            block_output = intermediate_list[0] # (N, 1+L, D)

        elif self.extraction_method == 'torchvision_feature_extractor':
            # torchvision ViT の指定層
            features_dict = self.feature_extractor(x)
            block_output = features_dict[self.target_node_name] # (N, 1+L, D)
        
        else:
            raise RuntimeError("Unknown extraction method.")

        # --- 2. Aggregation (block_output が (N, 1+L, D) の場合) ---
        if block_output is None:
            raise RuntimeError("Feature tensor was not extracted.")

        if self.aggregation_mode == 'cls_token':
            # [0] 番目が [CLS] トークン
            return block_output[:, 0, :]
        elif self.aggregation_mode == 'mean_pool_patch':
            # [1:] 番目がパッチトークン
            return block_output[:, 1:, :].mean(dim=1)
        elif self.aggregation_mode == 'mean_pool_all':
            # [CLS] もパッチも全部平均
            return block_output.mean(dim=1)
        else:
            raise ValueError(f"Unknown vit_aggregation_mode: {self.aggregation_mode}")


# --- 特徴抽出ヘルパー関数 ---
def extract_features(extractor, X_data, device, batch_size=64):
    """ 
    OOMを避けるためミニバッチで特徴抽出を実行
    X_data は (N, C, H, W) の [0, 1] 画像テンソルと仮定
    """
    print(f"Extracting features from {len(X_data)} samples using device {device}...")
    extractor.eval()
    extractor.to(device)
    features_list = []
    
    # ResNet, ViT, DINOv2 はいずれもImageNet正規化を期待するため，ここで適用
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = TensorDataset(X_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for i, (batch,) in enumerate(loader):
            batch = batch.to(device)
            # 正規化を適用
            batch_normalized = normalize(batch)
            features = extractor(batch_normalized)
            features_list.append(features.cpu())
            if i % 50 == 0 and i > 0:
                print(f"  ... processed {i * batch_size} / {len(X_data)} samples")
                
    print("Feature extraction complete.")
    return torch.cat(features_list, dim=0)

# キャッシュファイル名生成ヘルパー
def get_cache_filename(dataset_name, model_name, config, split):
    """設定に基づいて一意なキャッシュファイル名を生成"""
    name_parts = [dataset_name, model_name]
    
    # データセット固有のパラメータを追加
    if dataset_name == 'ColoredMNIST':
        if split == 'train':
            name_parts.append(f"corr{config.get('train_correlation', 0.0)}")
            name_parts.append(f"y{config.get('train_label_marginal', 0.0)}")
            name_parts.append(f"a{config.get('train_attribute_marginal', 0.0)}")
            name_parts.append(f"n{config.get('num_train_samples', 0)}")
        else: # test
            name_parts.append(f"corr{config.get('test_correlation', 0.0)}")
            name_parts.append(f"y{config.get('test_label_marginal', 0.0)}")
            name_parts.append(f"a{config.get('test_attribute_marginal', 0.0)}")
            name_parts.append(f"n{config.get('num_test_samples', 0)}")
    elif dataset_name == 'WaterBirds':
        if split == 'train':
            name_parts.append(f"n{config.get('num_train_samples', 0)}")
        else: # test
            name_parts.append(f"n{config.get('num_test_samples', 0)}")

    # 特徴抽出器のパラメータを追加
    if 'ResNet' in model_name:
        name_parts.append(config.get('feature_extractor_resnet_intermediate_layer', 'avgpool'))
        if config.get('feature_extractor_resnet_intermediate_layer', 'avgpool') != 'avgpool':
            name_parts.append(f"pool{config.get('feature_extractor_resnet_pooling_output_size', 1)}")
    elif 'ViT' in model_name:
        name_parts.append(f"block{config.get('feature_extractor_vit_target_block', -1)}")
        name_parts.append(config.get('feature_extractor_vit_aggregation_mode', 'cls_token'))
    
    name_parts.append(split) # 'train' or 'test'
    # ファイル名に使えない文字を置換
    filename = '_'.join(map(str, name_parts)).replace(' ', '_').replace('/', '_').replace('-', 'm')
    if len(filename) > 100:
         # シンプルなハッシュ（Python標準のhash）を使って短縮
         hash_val = hash(filename)
         filename = f"{filename[:80]}_{hash_val}"

    return f"{filename}.pt"


def setup_feature_extractor(config):
    """
    config に基づいて特徴抽出器モデルをセットアップし，
    モデルとMLPの入力次元を返す．
    """
    
    model_name = config.get('feature_extractor_model_name', 'ResNet18') 
    print(f"Setting up feature extractor: {model_name}...")
    
    feature_extractor = None
    input_dim_for_mlp = None 

    # --- モデル名に基づいてロード ---
    if 'ResNet' in model_name:
        # --- 1. ResNet系 (ResNet18, ResNet50) ---
        
        # (ResNet系 固有の設定)
        intermediate_layer_name = config.get('feature_extractor_resnet_intermediate_layer', 'avgpool')
        pool_output_size = config.get('feature_extractor_resnet_pooling_output_size', 1)
        
        if model_name == 'ResNet18':
            resnet_base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            base_output_dim = 512 # fc.in_features
            layer3_channels = 256
            layer4_channels = 512
        elif model_name == 'ResNet50':
            resnet_base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            base_output_dim = 2048 # fc.in_features
            layer3_channels = 1024
            layer4_channels = 2048
        else:
            raise ValueError(f"Unknown ResNet model: {model_name}")

        if intermediate_layer_name == 'avgpool':
            print(f"Using standard {model_name} output (after avgpool, {base_output_dim} dim)")
            resnet_base.fc = torch.nn.Identity()
            feature_extractor = torch.nn.Sequential(resnet_base)
            input_dim_for_mlp = base_output_dim

        else:
            print(f"Using intermediate feature map from: '{intermediate_layer_name}'")
            print(f"Using AdaptiveAvgPool with output size: ({pool_output_size}, {pool_output_size})")

            modules_list = [
                resnet_base.conv1, resnet_base.bn1, resnet_base.relu, resnet_base.maxpool,
                resnet_base.layer1, resnet_base.layer2,
            ]
            
            if intermediate_layer_name == 'layer3':
                modules_list.append(resnet_base.layer3)
                intermediate_channels = layer3_channels
            elif intermediate_layer_name == 'layer4':
                modules_list.append(resnet_base.layer3)
                modules_list.append(resnet_base.layer4)
                intermediate_channels = layer4_channels
            else:
                raise ValueError(f"Unsupported 'feature_extractor_resnet_intermediate_layer' for ResNet: {intermediate_layer_name}. "
                                 f"Must be 'avgpool', 'layer3', or 'layer4'.")

            backbone = torch.nn.Sequential(*modules_list)
            adaptive_pool = torch.nn.AdaptiveAvgPool2d((pool_output_size, pool_output_size))
            flatten = torch.nn.Flatten()
            pooled_dim = intermediate_channels * (pool_output_size ** 2)

            feature_extractor = torch.nn.Sequential(backbone, adaptive_pool, flatten)
            input_dim_for_mlp = pooled_dim
            
            print(f"Extractor: {model_name}-to-{intermediate_layer_name} ({intermediate_channels} channels) -> "
                  f"AdaptiveAvgPool({pool_output_size}x{pool_output_size}) -> Flatten -> "
                  f"(output dim: {input_dim_for_mlp})")

    # --- 2. ViT/DINOv2系 ---
    elif 'ViT' in model_name:
        print(f"Using {model_name}.")
        
        # --- ViT/DINOv2 固有の設定を読み込み ---
        target_block = config.get('feature_extractor_vit_target_block', -1)
        aggregation_mode = config.get('feature_extractor_vit_aggregation_mode', 'cls_token')
        
        print(f"  ViT/DINOv2 Settings: target_block={target_block}, aggregation_mode='{aggregation_mode}'")
        print("  (ResNet intermediate layer/pooling settings are ignored.)")

        base_model = None
        output_dim = 0
        
        if model_name == 'ViT_B_16':
            base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            output_dim = base_model.heads.head.in_features # 768
        
        elif 'DINOv2' in model_name:
            # DINOv2 ファミリーのモデル名と torch.hub 名，出力次元のマッピング
            dinov2_model_map = {
                'DINOv2_ViT_S_14': ('dinov2_vits14', 384),
                'DINOv2_ViT_B_14': ('dinov2_vitb14', 768),
                'DINOv2_ViT_L_14': ('dinov2_vitl14', 1024),
                'DINOv2_ViT_G_14': ('dinov2_vitg14', 1536),
            }
            
            if model_name not in dinov2_model_map:
                raise ValueError(f"Unknown DINOv2 model: {model_name}. "
                                 f"Supported DINOv2 models are: {list(dinov2_model_map.keys())}")
            
            hub_model_name, model_output_dim = dinov2_model_map[model_name]
            
            try:
                # torch.hub.set_dir('.') # 保存場所をカレントディレクトリに指定したい場合
                base_model = torch.hub.load('facebookresearch/dinov2', hub_model_name)
                output_dim = model_output_dim # マッピングから取得した次元
            except Exception as e:
                print(f"Failed to load DINOv2 model '{hub_model_name}' from torch.hub: {e}")
                print("Please ensure you have an internet connection and 'torch.hub' can access GitHub.")
                raise e
        else:
            # 'ViT' を含むが上記で処理されなかった場合
            raise ValueError(f"Unknown ViT/DINOv2 model: {model_name}")

        # --- 汎用ラッパーで包む ---
        feature_extractor = ViTFeatureExtractor(
            base_model, 
            model_name=model_name,
            target_block_index=target_block,
            aggregation_mode=aggregation_mode
        )
        
        input_dim_for_mlp = output_dim
        print(f"Extractor: {model_name} wrapped by ViTFeatureExtractor (output dim: {input_dim_for_mlp})")
    
    else:
        raise ValueError(f"Unknown 'feature_extractor_model_name': {model_name}")

    
    # 特徴抽出器のパラメータはすべて凍結
    for param in feature_extractor.parameters():
        param.requires_grad = False
        
    print(f"All parameters of {model_name} backbone are frozen.")
    
    return feature_extractor, input_dim_for_mlp
