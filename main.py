# sp/main.py

import os
import yaml
import time
import shutil
import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import wandb
import torchvision.models as models
from torchvision import transforms

# スクリプトをインポート
import data_loader
import utils
import model as model_module
import analysis
import plotting

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

# --- ヘルパー関数 ---
def get_loss_function(scores, y_batch, loss_type='mse'):
    """ 損失関数を計算 """
    if loss_type == 'logistic':
        return F.softplus(-y_batch * scores).mean()
    elif loss_type == 'mse':
        return F.mse_loss(scores, y_batch)
    else:
        raise ValueError(f"Unknown loss_function: {loss_type}")

def main(config_path='config.yaml'):
    # 1. 設定ファイルの読み込み
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # --- debias_method の読み込み ---
    debias_method = config.get('debias_method', 'None')
    loss_function_name = config['loss_function']
    
    # eval_batch_size を config から読み込む
    # 見つからない場合は None を設定 (utils.py 側でフルバッチとして扱われる)
    eval_batch_size = config.get('eval_batch_size', None)


    # wandbの初期化
    if config.get('wandb', {}).get('enable', False):
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            config=config,
            name=f"{config['experiment_name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        print("wandb is enabled and initialized.")

    # 2. 結果保存ディレクトリの作成
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_dir = os.path.join('results', f"{config['experiment_name']}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(result_dir, 'config_used.yaml'))
    print(f"Results will be saved to: {result_dir}")

    # デバイス設定
    device = config['device'] if torch.cuda.is_available() and config['device'] == 'cuda' else "cpu"
    print(f"Using device: {device}")

    # 3. データセットの準備
    print("\n--- 1. Preparing Dataset ---")
    
    # キャッシュディレクトリの準備
    CACHE_DIR = config.get('features_cache_dir', 'features_cache') # config.yaml に 'features_cache_dir' を追加可能
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Using feature cache directory: {CACHE_DIR}")

    # --- 特徴抽出器のセットアップ (configに応じて) ---
    feature_extractor = None
    # configから設定を読み込む
    use_feature_extractor = config.get('use_feature_extractor', False)
    # モデル名を取得
    model_name = config.get('feature_extractor_model_name', 'ResNet18') 
    
    # MLPの入力次元を保持する変数
    input_dim_for_mlp = None 

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

    # キャッシュパスの変数を初期化
    cache_path_train_X = None
    cache_path_train_y = None
    cache_path_train_a = None
    cache_path_test_X = None
    cache_path_test_y = None
    cache_path_test_a = None


    if use_feature_extractor:
        print(f"Setting up feature extractor: {model_name}...")
        
        # キャッシュパスの生成
        # (configに基づいてパスを生成)
        # X, y, a それぞれにキャッシュファイルを作成
        base_name_train = get_cache_filename(config['dataset_name'], model_name, config, 'train')
        base_name_test = get_cache_filename(config['dataset_name'], model_name, config, 'test')
        
        cache_path_train_X = os.path.join(CACHE_DIR, base_name_train.replace('.pt', '_X.pt'))
        cache_path_train_y = os.path.join(CACHE_DIR, base_name_train.replace('.pt', '_y.pt'))
        cache_path_train_a = os.path.join(CACHE_DIR, base_name_train.replace('.pt', '_a.pt'))
        
        cache_path_test_X = os.path.join(CACHE_DIR, base_name_test.replace('.pt', '_X.pt'))
        cache_path_test_y = os.path.join(CACHE_DIR, base_name_test.replace('.pt', '_y.pt'))
        cache_path_test_a = os.path.join(CACHE_DIR, base_name_test.replace('.pt', '_a.pt'))

        print(f"Train feature cache path (X): {cache_path_train_X}")
        print(f"Test feature cache path (X): {cache_path_test_X}")

        # --- モデル名に基づいてロード ---
        # (L346 - L472: input_dim_for_mlp を計算するために，このブロックはキャッシュの有無に関わらず実行)
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
    
    elif not use_feature_extractor:
        if config['dataset_name'] == 'WaterBirds':
            print("Using raw WaterBirds images (3x224x224). OOM might occur.")
        else:
            print("Using raw image pixels.")

    # 特徴抽出の実行 or キャッシュのロード
    
    # キャッシュの存在確認
    use_cache = (
        use_feature_extractor and
        cache_path_train_X is not None and
        os.path.exists(cache_path_train_X) and
        os.path.exists(cache_path_train_y) and
        os.path.exists(cache_path_train_a) and
        os.path.exists(cache_path_test_X) and
        os.path.exists(cache_path_test_y) and
        os.path.exists(cache_path_test_a)
    )

    if use_cache:
        # --- キャッシュが存在する場合 ---
        try:
            print(f"Loading features and labels from cache...")
            X_train = torch.load(cache_path_train_X, map_location=torch.device('cpu')) # CPUにロード
            y_train = torch.load(cache_path_train_y, map_location=torch.device('cpu'))
            a_train = torch.load(cache_path_train_a, map_location=torch.device('cpu'))
            X_test = torch.load(cache_path_test_X, map_location=torch.device('cpu')) # CPUにロード
            y_test = torch.load(cache_path_test_y, map_location=torch.device('cpu'))
            a_test = torch.load(cache_path_test_a, map_location=torch.device('cpu'))
            
            print("Successfully loaded features and labels from cache.")
            print(f"Feature dimensions from cache: Train={X_train.shape}, Test={X_test.shape}")
            
            # feature_extractor はもう不要なのでメモリ解放
            if feature_extractor is not None:
                del feature_extractor
                feature_extractor = None

        except Exception as e:
            print(f"Warning: Failed to load features from cache: {e}. Re-extracting...")
            use_cache = False # ロード失敗
            if feature_extractor is None:
                 raise RuntimeError("Feature extractor was not set up, but cache load failed.") # 安全装置
    
    if not use_cache:
        # --- キャッシュが存在しない (or ロード失敗) の場合 ---
        
        # 生データをロード
        if config['dataset_name'] == 'ColoredMNIST':
            image_size = 28

            train_y_bar = config.get('train_label_marginal', 0.0)
            train_a_bar = config.get('train_attribute_marginal', 0.0)
            test_y_bar = config.get('test_label_marginal', 0.0)
            test_a_bar = config.get('test_attribute_marginal', 0.0)

            X_train, y_train, a_train = data_loader.get_colored_mnist(
                num_samples=config['num_train_samples'],
                correlation=config['train_correlation'],
                label_marginal=train_y_bar,
                attribute_marginal=train_a_bar,
                train=True
            )
            X_test, y_test, a_test = data_loader.get_colored_mnist(
                num_samples=config['num_test_samples'],
                correlation=config['test_correlation'],
                label_marginal=test_y_bar,
                attribute_marginal=test_a_bar,
                train=False
            )

        elif config['dataset_name'] == 'WaterBirds':
            image_size = 224
            X_train, y_train, a_train, X_test, y_test, a_test = data_loader.get_waterbirds_dataset(
                num_train=config['num_train_samples'], num_test=config['num_test_samples'], image_size=image_size
            )
            
        else:
            raise ValueError(f"Unknown dataset: {config['dataset_name']}")

        # 特徴抽出 (必要な場合)
        if use_feature_extractor:
            if feature_extractor is None:
                 raise RuntimeError("Feature extractor was not set up, but cache was not found.") # 安全装置
            
            print("--- Starting Feature Extraction (Train) ---")
            X_train_features = extract_features(feature_extractor, X_train, device)
            print("--- Starting Feature Extraction (Test) ---")
            X_test_features = extract_features(feature_extractor, X_test, device)
            print(f"Feature dimensions after extraction: Train={X_train_features.shape}, Test={X_test_features.shape}")
            
            # X_train, X_test を特徴量で上書き
            X_train = X_train_features
            X_test = X_test_features
            
            # キャッシュに保存
            try:
                print(f"Saving features and labels to cache...")
                torch.save(X_train, cache_path_train_X)
                torch.save(y_train, cache_path_train_y)
                torch.save(a_train, cache_path_train_a)
                torch.save(X_test, cache_path_test_X)
                torch.save(y_test, cache_path_test_y)
                torch.save(a_test, cache_path_test_a)
                print("Successfully saved features and labels to cache.")
            except Exception as e_save:
                print(f"Warning: Failed to save features to cache: {e_save}")
        
        # (use_feature_extractor=False の場合は，生データのまま進む)


    if config['show_and_save_samples']:
        # --- データが画像の場合のみサンプル表示 ---
        if X_train.dim() == 4: # (B, C, H, W) 
            utils.show_dataset_samples(X_train, y_train, a_train, config['dataset_name'], result_dir)
        else:
            print("Skipping dataset sample visualization (data is pre-extracted features).")

    X_train = utils.l2_normalize_images(X_train)
    X_test = utils.l2_normalize_images(X_test)

    # --- グループのリストを定義 ---
    group_keys = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]

    # --- バイアス除去手法のための設定 ---
    static_weights = None
    dro_q_weights = None
    
    if debias_method == 'IW_uniform':
        print("\n--- Importance Weighting (Uniform Target) Enabled (Equivalent to v_inv) ---")
        print(f"  [Warning] 'train_batch_size' config is ignored. Using full-batch (per-group) gradient calculation.")
        print("  Removing both marginal bias (Term II) and spurious correlation (Term III).")

        # 理論に基づき，重みを一律 1/4 (0.25) に設定 (v_inv の勾配流)
        static_weights = {g: 0.25 for g in group_keys}

        print("  Using static weights for uniform target distribution (w_g = 0.25 for all):")
        for g, w in static_weights.items():
            print(f"  w_g{g} = {w:.6f}")
        
        train_loader = None
        
    elif debias_method == 'GroupDRO':
        print("\n--- Group DRO Enabled ---")
        print(f"  [Warning] 'train_batch_size' config is ignored. Using full-batch (per-group) gradient calculation.")
        
        # 動的重み q を一様分布で初期化
        dro_q_weights = torch.ones(len(group_keys), device=device) / len(group_keys)
        
        print(f"  Using dynamic weights 'q' initialized to: {dro_q_weights.cpu().numpy()}")
        print(f"  Group weight step size (eta_q): {config['dro_eta_q']}")
        
        train_loader = None

    elif debias_method == 'None':
        # 通常のERM学習
        print(f"\n--- ERM (Debias Method: None) Enabled ---")
        train_batch_size_erm = config.get('train_batch_size', 50000) 
        print(f"  Using train_batch_size: {train_batch_size_erm}")
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=train_batch_size_erm, shuffle=True)
    
    else:
        raise ValueError(f"Unknown debias_method: {debias_method}. Must be 'None', 'IW_uniform', or 'GroupDRO'.")

    utils.display_group_distribution(y_train, a_train, "Train Set", config['dataset_name'], result_dir)
    utils.display_group_distribution(y_test, a_test, "Test Set", config['dataset_name'], result_dir)

    # 4. モデルとオプティマイザの準備
    print("\n--- 2. Setting up Model and Optimizer ---")
    
    # --- input_dim の計算 (config に応じて) ---
    if use_feature_extractor:
        # input_dim_for_mlp の扱い
        # input_dim_for_mlp が設定されていない (キャッシュロードなどで) 場合，
        # ロードした X_train の次元から復元する
        if input_dim_for_mlp is None:
             if X_train.dim() == 2: # (B, D)
                 input_dim_for_mlp = X_train.shape[1]
                 print(f"Using FEATURE input_dim (inferred from cached data): {input_dim_for_mlp}")
             else:
                 raise ValueError(f"Cached feature data has unexpected dimensions: {X_train.shape}")
        
        if input_dim_for_mlp is None:
             raise ValueError("input_dim_for_mlp was not set correctly during feature extractor setup.")
        input_dim = input_dim_for_mlp
        print(f"Using FEATURE input_dim: {input_dim}")
        
    else:
        # 特徴抽出を使わない場合
        
        # X_train の形状から input_dim を計算
        # (B, C, H, W) -> C*H*W または (B, D) -> D
        if X_train.dim() == 4: # 画像データ
            input_dim = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
        elif X_train.dim() == 2: # すでに特徴量
             input_dim = X_train.shape[1]
        else:
            raise ValueError(f"Unexpected X_train dimensions: {X_train.shape}")
            
        print(f"Using RAW input_dim ({config['dataset_name']}): {input_dim}")
        
    model = model_module.MLP(
        input_dim=input_dim, hidden_dim=config['hidden_dim'],
        num_hidden_layers=config['num_hidden_layers'], activation_fn=config['activation_function'],
        use_skip_connections=config['use_skip_connections'],
        initialization_method=config['initialization_method']
    ).to(device)

    # apply_manual_parametrization はモデルの重みを直接初期化する
    model_module.apply_manual_parametrization(
        model, method=config['initialization_method'],
        hidden_dim=config['hidden_dim'],
        fix_final_layer=config.get('fix_final_layer', False)
    )

    # オプティマイザに渡すパラメータを設定
    # (fix_final_layer=True の場合, model.parameters() は
    #  requires_grad=True のパラメータのみを返すため自動的に処理される)
    optimizer_params_list = model.parameters()

    # オプティマイザの作成
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(optimizer_params_list, lr=config['learning_rate'])
    else:
        optimizer = optim.SGD(optimizer_params_list, lr=config['learning_rate'], momentum=config['momentum'])


    if config.get('wandb', {}).get('enable', False):
        wandb.watch(model, log='all', log_freq=100)

    all_target_layers = [f'layer_{i+1}' for i in range(config['num_hidden_layers'])]

    # 5. 学習・評価ループ
    print("\n--- 3. Starting Training & Evaluation Loop ---")
    print(f"Using eval_batch_size: {eval_batch_size if eval_batch_size is not None else 'Full Batch'}")
    
    history = {k: [] for k in ['train_avg_loss', 'test_avg_loss', 'train_worst_loss', 'test_worst_loss',
                               'train_avg_acc', 'test_avg_acc', 'train_worst_acc', 'test_worst_acc',
                               'train_group_losses', 'test_group_losses', 'train_group_accs', 'test_group_accs']}

    analysis_histories = {name: {} for name in [
        'jacobian_norm_train', 'jacobian_norm_test',
        'grad_basis_train', 'grad_basis_test',
        'gap_factors_train', 'gap_factors_test',
        'static_dynamic_decomp_train', 'static_dynamic_decomp_test'
    ]}
    
    for epoch in range(config['epochs']):
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

        # --- 評価 ---
        train_metrics = utils.evaluate_model(model, X_train, y_train, a_train, device, loss_function_name, eval_batch_size)
        test_metrics = utils.evaluate_model(model, X_test, y_test, a_test, device, loss_function_name, eval_batch_size)
        
        for key_base in history.keys():
            if key_base.startswith('train_'):
                history[key_base].append(train_metrics[key_base.replace('train_', '')])
            else:
                history[key_base].append(test_metrics[key_base.replace('test_', '')])

        print(f"Epoch {epoch+1:5d}/{config['epochs']} | Train [Loss: {train_metrics['avg_loss']:.4f}, Worst: {train_metrics['worst_loss']:.4f}, Acc: {train_metrics['avg_acc']:.4f}, Worst: {train_metrics['worst_acc']:.4f}] | Test [Loss: {test_metrics['avg_loss']:.4f}, Worst: {test_metrics['worst_loss']:.4f}, Acc: {test_metrics['avg_acc']:.4f}, Worst: {test_metrics['worst_acc']:.4f}]")

        if config.get('wandb', {}).get('enable', False):
            log_metrics = {
                'epoch': epoch + 1,
                'train_avg_loss': train_metrics['avg_loss'],
                'train_worst_loss': train_metrics['worst_loss'],
                'train_avg_acc': train_metrics['avg_acc'],
                'train_worst_acc': train_metrics['worst_acc'],
                'test_avg_loss': test_metrics['avg_loss'],
                'test_worst_loss': test_metrics['worst_loss'],
                'test_avg_acc': test_metrics['avg_acc'],
                'test_worst_acc': test_metrics['worst_acc'],
            }
            if debias_method == 'GroupDRO':
                for i, g in enumerate(group_keys):
                    log_metrics[f'group_q_weight/q_g{g}'] = dro_q_weights[i].item()

            for i in range(4):
                log_metrics[f'train_group_{i}_loss'] = train_metrics['group_losses'][i]
                log_metrics[f'train_group_{i}_acc'] = train_metrics['group_accs'][i]
                log_metrics[f'test_group_{i}_loss'] = test_metrics['group_losses'][i]
                log_metrics[f'test_group_{i}_acc'] = test_metrics['group_accs'][i]
            
            # y=-1 の損失差 (少数派 - 多数派)
            if not np.isnan(train_metrics['group_losses'][1]) and not np.isnan(train_metrics['group_losses'][0]):
                log_metrics['train_loss_gap_y_neg1'] = train_metrics['group_losses'][1] - train_metrics['group_losses'][0]
            if not np.isnan(test_metrics['group_losses'][1]) and not np.isnan(test_metrics['group_losses'][0]):
                log_metrics['test_loss_gap_y_neg1'] = test_metrics['group_losses'][1] - test_metrics['group_losses'][0]
            
            # y=+1 の損失差 (少数派 - 多数派)
            if not np.isnan(train_metrics['group_losses'][2]) and not np.isnan(train_metrics['group_losses'][3]):
                log_metrics['train_loss_gap_y_pos1'] = train_metrics['group_losses'][2] - train_metrics['group_losses'][3]
            if not np.isnan(test_metrics['group_losses'][2]) and not np.isnan(test_metrics['group_losses'][3]):
                log_metrics['test_loss_gap_y_pos1'] = test_metrics['group_losses'][2] - test_metrics['group_losses'][3]

            wandb.log(log_metrics)

        # --- チェックポイント分析 ---
        current_epoch = epoch + 1

        def should_run(analysis_name, epoch_list_name):
            if not config.get(analysis_name, False):
                return False
            epoch_list = config.get(epoch_list_name)
            if epoch_list is None: # キーが存在しないか，値がNone（毎エポック実行）
                return True
            return current_epoch in epoch_list # リストが指定されている場合

        run_grad_basis = should_run('analyze_gradient_basis', 'gradient_basis_analysis_epochs')
        run_gap_factors = should_run('analyze_gap_dynamics_factors', 'gap_dynamics_factors_analysis_epochs')
        run_jacobian_norm = should_run('analyze_jacobian_norm', 'jacobian_norm_analysis_epochs')
        run_static_dynamic = should_run('analyze_static_dynamic_decomposition', 'static_dynamic_decomposition_analysis_epochs')

        run_any_analysis = run_grad_basis or run_gap_factors or run_jacobian_norm or run_static_dynamic

        if run_any_analysis:
            print(f"\n{'='*25} CHECKPOINT ANALYSIS @ EPOCH {current_epoch} {'='*25}")

            # train_outputs, test_outputs は analysis.py で使われなくなった
            train_outputs, test_outputs = (None, None)

            temp_config = config.copy()
            temp_config['analyze_gradient_basis'] = run_grad_basis
            temp_config['analyze_gap_dynamics_factors'] = run_gap_factors
            temp_config['analyze_jacobian_norm'] = run_jacobian_norm
            temp_config['analyze_static_dynamic_decomposition'] = run_static_dynamic

            analysis.run_all_analyses(
                temp_config, current_epoch, all_target_layers, model, train_outputs, test_outputs,
                X_train, y_train, a_train, X_test, y_test, a_test, analysis_histories,
                history
            )

            if config.get('wandb', {}).get('enable', False):
                analysis_log_metrics = {}
                for history_key, history_dict in analysis_histories.items():
                    if current_epoch in history_dict:
                        epoch_data = history_dict[current_epoch]
                        if isinstance(epoch_data, dict):
                            for sub_key, value in epoch_data.items():
                                # jacobian, basis, gap_factors, static_dynamic_decomp はすべて
                                # スカラ値の辞書を返すため，単純にログ記録
                                analysis_log_metrics[f'analysis/{history_key}/{sub_key}'] = value
                if analysis_log_metrics:
                    analysis_log_metrics['epoch'] = current_epoch
                    wandb.log(analysis_log_metrics)

            print(f"{'='*25} END OF ANALYSIS @ EPOCH {current_epoch} {'='*27}\n")

    # 6. 最終結果の保存とプロット
    print("\n--- 4. Saving Final Results and Plotting ---")
    history_df = pd.DataFrame(history)
    history_df.index.name = 'epoch'
    history_df.to_csv(os.path.join(result_dir, 'training_history.csv'))

    plotting.plot_all_results(history_df, analysis_histories, all_target_layers, result_dir, config)

    if config.get('wandb', {}).get('enable', False):
        wandb.finish()

    print(f"\nExperiment finished. All results saved in: {result_dir}")

if __name__ == '__main__':
    main(config_path='config.yaml')
