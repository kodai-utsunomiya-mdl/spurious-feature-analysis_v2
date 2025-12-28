# # sp/data_loader.py

# import torch
# import os
# import numpy as np
# from torchvision.datasets import MNIST, CIFAR10
# from torchvision import transforms
# import zipfile
# import pandas as pd
# from PIL import Image
# import sys
# import torch.nn.functional as F

# # # 'wilds'ライブラリのインポート
# # try:
# #     import wilds
# # except ImportError:
# #     print("Warning: 'wilds' library not found. WaterBirds dataset will not be available.")
# #     print("Please install it using: pip install wilds")
# #     wilds = None

# def colorize_mnist(images_all, labels_0_9_all, num_samples, label_marginal, attribute_marginal, correlation):
#     """ 
#     MNISTデータセットに色付けを行い，指定された統計量を持つデータを作成
    
#     Args:
#         images_all (torch.Tensor): MNISTの全画像テンソル (N_all, H, W)
#         labels_0_9_all (torch.Tensor): MNISTの全ラベルテンソル (N_all,)
#         num_samples (int): 生成するサンプル数 (N)
#         label_marginal (float): 目標とするラベルの期待値 E[Y] (y_bar)
#         attribute_marginal (float): 目標とする属性の期待値 E[A] (a_bar)
#         correlation (float): 目標とする共分散 Cov(Y, A) (rho_train)
#     """
    
#     y_bar = label_marginal
#     a_bar = attribute_marginal
#     rho_train = correlation
    
#     # 1. 元のデータを Y = {-1, +1} に変換
#     labels_pm1_all = (labels_0_9_all >= 5).float() * 2.0 - 1.0
    
#     # 2. Y=+1 と Y=-1 の利用可能な全インデックスを取得
#     all_indices_y_pos = torch.where(labels_pm1_all == 1.0)[0]
#     all_indices_y_neg = torch.where(labels_pm1_all == -1.0)[0]

#     # 3. 4つのグループの確率(pi_g)と目標サンプル数(N_g)を計算
#     # E[YA] = Cov(Y,A) + E[Y]E[A]
#     E_ya = rho_train + y_bar * a_bar
    
#     # (Y, A) = (+1, +1), (+1, -1), (-1, +1), (-1, -1)
#     # 4*pi(y,a) = 1 + y*E[Y] + a*E[A] + (y*a)*E[YA]
#     pi_pp = (1 + y_bar + a_bar + E_ya) / 4.0
#     pi_pn = (1 + y_bar - a_bar - E_ya) / 4.0
#     pi_np = (1 - y_bar + a_bar - E_ya) / 4.0
#     pi_nn = (1 - y_bar - a_bar + E_ya) / 4.0
    
#     probabilities = [pi_pp, pi_pn, pi_np, pi_nn]
    
#     # 確率が [0, 1] の範囲外になる場合はエラー
#     if not all(p >= -1e-6 for p in probabilities): # 浮動小数点誤差を許容
#         raise ValueError(f"Invalid statistics (y_bar={y_bar}, a_bar={a_bar}, rho={rho_train}) result in negative probabilities: {probabilities}")
#     if abs(sum(probabilities) - 1.0) > 1e-6:
#         raise ValueError(f"Probabilities do not sum to 1: {probabilities}")

#     # 目標サンプル数を計算 (丸め処理)
#     N_pp = int(np.round(num_samples * pi_pp))
#     N_pn = int(np.round(num_samples * pi_pn))
#     N_np = int(np.round(num_samples * pi_np))
#     N_nn = int(np.round(num_samples * pi_nn))
    
#     # 丸め誤差を調整 (最大のグループに押し付ける)
#     counts = [N_pp, N_pn, N_np, N_nn]
#     N_total_calc = sum(counts)
#     if N_total_calc != num_samples:
#         diff = num_samples - N_total_calc
#         max_idx = np.argmax(counts)
#         counts[max_idx] += diff
    
#     N_pp, N_pn, N_np, N_nn = counts[0], counts[1], counts[2], counts[3]

#     # 4. Yごとの必要枚数を計算
#     N_pos_target = N_pp + N_pn
#     N_neg_target = N_np + N_nn

#     # 5. 利用可能なサンプル数で足りるかチェック
#     if N_pos_target > len(all_indices_y_pos):
#         raise ValueError(f"Cannot sample {N_pos_target} (Y=+1) samples. Only {len(all_indices_y_pos)} available in the source dataset.")
#     if N_neg_target > len(all_indices_y_neg):
#         raise ValueError(f"Cannot sample {N_neg_target} (Y=-1) samples. Only {len(all_indices_y_neg)} available in the source dataset.")

#     # 6. Y=+1, Y=-1 グループからランダムサンプリング
#     indices_y_pos_sample = np.random.choice(all_indices_y_pos, N_pos_target, replace=False)
#     indices_y_neg_sample = np.random.choice(all_indices_y_neg, N_neg_target, replace=False)

#     # 7. 属性(A)を割り当て
#     # Y=+1 グループ
#     indices_pp = indices_y_pos_sample[:N_pp] # A = +1 (Red)
#     indices_pn = indices_y_pos_sample[N_pp:] # A = -1 (Green)
#     # Y=-1 グループ
#     indices_np = indices_y_neg_sample[:N_np] # A = +1 (Red)
#     indices_nn = indices_y_neg_sample[N_np:] # A = -1 (Green)
    
#     # 8. 最終的なインデックスと属性(A)リストを作成
#     final_indices = np.concatenate([indices_pp, indices_pn, indices_np, indices_nn])
    
#     attributes_pm1 = torch.zeros(num_samples, dtype=torch.float32)
#     # (Y, A) = (+1, +1)
#     attributes_pm1[0:N_pp] = 1.0
#     # (Y, A) = (+1, -1)
#     attributes_pm1[N_pp:N_pos_target] = -1.0
#     # (Y, A) = (-1, +1)
#     attributes_pm1[N_pos_target:(N_pos_target + N_np)] = 1.0
#     # (Y, A) = (-1, -1)
#     attributes_pm1[(N_pos_target + N_np):num_samples] = -1.0

#     # 9. 最終的なデータセットを構築
#     images_subset = images_all[final_indices]
#     labels_pm1_subset = labels_pm1_all[final_indices]
    
#     # 10. 色付け処理
#     images_gray = images_subset.float() / 255.0
#     images_rgb = torch.stack([images_gray, images_gray, images_gray], dim=1)
#     n_samples_final = len(images_subset) # = num_samples

#     digit_mask = (images_gray > 0.01).unsqueeze(1)
#     color_factors = torch.ones(n_samples_final, 3, 1, 1, dtype=images_gray.dtype)

#     red_indices = (attributes_pm1 == 1.0)
#     green_indices = (attributes_pm1 == -1.0)

#     color_factors[red_indices, 1, :, :] = 0.25
#     color_factors[red_indices, 2, :, :] = 0.25
#     color_factors[green_indices, 0, :, :] = 0.25
#     color_factors[green_indices, 2, :, :] = 0.25

#     colored_images = images_rgb * color_factors
#     final_images_rgb = torch.where(digit_mask, colored_images, images_rgb)
    
#     # 11. データをシャッフルして返す
#     shuffle_perm = torch.randperm(n_samples_final)
    
#     return final_images_rgb[shuffle_perm], labels_pm1_subset[shuffle_perm], attributes_pm1[shuffle_perm]

# def get_colored_mnist(num_samples, correlation, label_marginal, attribute_marginal, train=True):
#     """ ColoredMNISTデータセットをロードして生成 """
#     set_name = 'train' if train else 'test'
#     print(f"Preparing Colored MNIST for {set_name} set...")
#     print(f"  Target Stats: N={num_samples}, E[Y]={label_marginal}, E[A]={attribute_marginal}, Cov(Y,A)={correlation}")
    
#     mnist_dataset = MNIST('./data', train=train, download=True)

#     # 元のMNISTデータセット全体を読み込む
#     images = mnist_dataset.data
#     targets = mnist_dataset.targets

#     return colorize_mnist(images, targets, num_samples, label_marginal, attribute_marginal, correlation)

# def get_colored_mnist_all(config):
#     """ 
#     CMNISTの全データセットを取得 (Train, Test) 
#     Validationは作成しない (use_dfr=TrueのときにTrainから分割する)
#     """
#     image_size = 28
#     train_y_bar = config.get('train_label_marginal', 0.0)
#     train_a_bar = config.get('train_attribute_marginal', 0.0)
#     test_y_bar = config.get('test_label_marginal', 0.0)
#     test_a_bar = config.get('test_attribute_marginal', 0.0)

#     X_train, y_train, a_train = get_colored_mnist(
#         num_samples=config['num_train_samples'],
#         correlation=config['train_correlation'],
#         label_marginal=train_y_bar,
#         attribute_marginal=train_a_bar,
#         train=True
#     )
    
#     X_test, y_test, a_test = get_colored_mnist(
#         num_samples=config['num_test_samples'],
#         correlation=config['test_correlation'],
#         label_marginal=test_y_bar,
#         attribute_marginal=test_a_bar,
#         train=False
#     )
    
#     # Validationセットは返さない
#     return X_train, y_train, a_train, X_test, y_test, a_test


# def get_waterbirds_dataset(num_train, num_test, image_size):
#     """ WaterBirdsデータセットをロード (Train, Test) """
#     # 公式のValidationセット (split=1) はロードしない
    
#     data_dir = 'data/waterbirds_v1.0'
#     archive_path = os.path.join(data_dir, 'archive.zip')
#     unzip_dir = os.path.join(data_dir, 'waterbird') # 展開後のディレクトリ名
#     metadata_path = os.path.join(unzip_dir, 'metadata.csv')

#     # 展開済みのデータが存在しない場合
#     if not os.path.exists(metadata_path):
#         # zipファイル自体も存在しない場合，ダウンロードを促す
#         if not os.path.exists(archive_path):
#             print("="*80)
#             print("Waterbirds dataset not found.")
#             print("Please download 'archive.zip' from the following URL:")
#             print("https://www.kaggle.com/datasets/bahardibaie/waterbird?resource=download")
#             print(f"And place it in the following directory: {data_dir}")
#             print("="*80)
#             sys.exit(1) # プログラムを停止
        
#         # zipファイルが存在する場合は展開する
#         print(f"Extracting {archive_path}...")
#         os.makedirs(data_dir, exist_ok=True)
#         with zipfile.ZipFile(archive_path, 'r') as zip_ref:
#             zip_ref.extractall(data_dir)
#         print("Extraction complete.")

#     # メタデータを読み込む
#     metadata_df = pd.read_csv(metadata_path)

#     # CMNIST と同様に [0, 1] スケーリング (ToTensor) のみを行う
#     transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    
#     def get_data_from_split(split_id, num_samples):
#         split_df = metadata_df[metadata_df['split'] == split_id]
        
#         num_to_sample = min(num_samples, len(split_df))
#         # 確実に同じデータを引けるようにシードを固定
#         sampled_df = split_df.sample(n=num_to_sample, random_state=42) 
        
#         images, y_labels, a_labels = [], [], []
        
#         # 画像読み込みループ
#         for _, row in sampled_df.iterrows():
#             img_path = os.path.join(unzip_dir, row['img_filename'])
#             image = Image.open(img_path).convert('RGB')
#             images.append(transform(image))
            
#             y_labels.append(row['y'])
#             a_labels.append(row['place'])
            
#         return torch.stack(images), torch.tensor(y_labels, dtype=torch.long), torch.tensor(a_labels, dtype=torch.long)

#     print("Loading Waterbirds Train set...")
#     X_train, y_train_01, a_train_01 = get_data_from_split(0, num_train)
    
#     # Validation (split=1) はスキップ

#     print("Loading Waterbirds Test set...")
#     X_test, y_test_01, a_test_01 = get_data_from_split(2, num_test)

#     # ラベルを-1, +1形式に変換
#     y_train_pm1 = y_train_01.float() * 2.0 - 1.0
#     a_train_pm1 = a_train_01.float() * 2.0 - 1.0 
    
#     y_test_pm1 = y_test_01.float() * 2.0 - 1.0
#     a_test_pm1 = a_test_01.float() * 2.0 - 1.0

#     return X_train, y_train_pm1, a_train_pm1, X_test, y_test_pm1, a_test_pm1

# def create_dominoes_dataset(mnist_images, mnist_targets, cifar_images, cifar_targets, 
#                           num_samples, label_marginal, attribute_marginal, correlation, 
#                           image_size=224, seed=None):
#     """
#     Dominoesデータセット (MNIST + CIFAR10) を生成
#     Top: MNIST (0/1) -> Spurious Attribute (0: -1, 1: +1)
#     Bottom: CIFAR10 (Car/Truck) -> Core Label (Car: -1, Truck: +1)
    
#     正規化: [0, 1] 範囲の float32 (他のデータセットと一致)
#     """
#     if seed is not None:
#         np.random.seed(seed)
#         torch.manual_seed(seed)

#     # 1. ターゲットと属性の定義
#     # MNIST: 0 -> A=-1, 1 -> A=+1
#     # CIFAR: Car(1) -> Y=-1, Truck(9) -> Y=+1
    
#     # 2. データのフィルタリング
#     mnist_mask_0 = (mnist_targets == 0)
#     mnist_mask_1 = (mnist_targets == 1)
#     # CIFARのターゲットはリスト形式の場合があるためTensor化
#     cifar_tensor_targets = torch.tensor(cifar_targets)
#     cifar_mask_car = (cifar_tensor_targets == 1)
#     cifar_mask_truck = (cifar_tensor_targets == 9)

#     mnist_indices_0 = torch.where(mnist_mask_0)[0].numpy()
#     mnist_indices_1 = torch.where(mnist_mask_1)[0].numpy()
#     cifar_indices_car = torch.where(cifar_mask_car)[0].numpy()
#     cifar_indices_truck = torch.where(cifar_mask_truck)[0].numpy()

#     # 3. グループごとのサンプル数計算 (colorize_mnistと同様のロジックを使用)
#     y_bar = label_marginal
#     a_bar = attribute_marginal
#     rho_train = correlation
    
#     # E[YA] = Cov(Y,A) + E[Y]E[A]
#     E_ya = rho_train + y_bar * a_bar
    
#     pi_pp = (1 + y_bar + a_bar + E_ya) / 4.0 # Y=+1 (Truck), A=+1 (1)
#     pi_pn = (1 + y_bar - a_bar - E_ya) / 4.0 # Y=+1 (Truck), A=-1 (0)
#     pi_np = (1 - y_bar + a_bar - E_ya) / 4.0 # Y=-1 (Car), A=+1 (1)
#     pi_nn = (1 - y_bar - a_bar + E_ya) / 4.0 # Y=-1 (Car), A=-1 (0)
    
#     probabilities = [pi_pp, pi_pn, pi_np, pi_nn]
#     if not all(p >= -1e-6 for p in probabilities):
#         raise ValueError(f"Invalid statistics result in negative probabilities: {probabilities}")

#     counts = [int(np.round(num_samples * p)) for p in probabilities]
#     # 丸め誤差調整
#     if sum(counts) != num_samples:
#         counts[np.argmax(counts)] += num_samples - sum(counts)
#     N_pp, N_pn, N_np, N_nn = counts

#     # 4. サンプリング (データ数が足りない場合は重複を許容)
#     def sample_indices(pool, n):
#         replace = len(pool) < n
#         return np.random.choice(pool, n, replace=replace)

#     # Pairs: (CIFAR, MNIST)
#     # Group PP: Truck (+1), 1 (+1)
#     idx_c_pp = sample_indices(cifar_indices_truck, N_pp)
#     idx_m_pp = sample_indices(mnist_indices_1, N_pp)
    
#     # Group PN: Truck (+1), 0 (-1)
#     idx_c_pn = sample_indices(cifar_indices_truck, N_pn)
#     idx_m_pn = sample_indices(mnist_indices_0, N_pn)
    
#     # Group NP: Car (-1), 1 (+1)
#     idx_c_np = sample_indices(cifar_indices_car, N_np)
#     idx_m_np = sample_indices(mnist_indices_1, N_np)

#     # Group NN: Car (-1), 0 (-1)
#     idx_c_nn = sample_indices(cifar_indices_car, N_nn)
#     idx_m_nn = sample_indices(mnist_indices_0, N_nn)

#     # 統合
#     cifar_indices_all = np.concatenate([idx_c_pp, idx_c_pn, idx_c_np, idx_c_nn])
#     mnist_indices_all = np.concatenate([idx_m_pp, idx_m_pn, idx_m_np, idx_m_nn])
    
#     # ラベルと属性の作成 (-1, +1 形式)
#     # Y: -1 (Car), +1 (Truck)
#     # A: -1 (0), +1 (1)
#     y_labels = torch.cat([
#         torch.ones(N_pp + N_pn),     # Truck (+1)
#         torch.ones(N_np + N_nn) * -1 # Car (-1)
#     ])
#     a_labels = torch.cat([
#         torch.ones(N_pp),          # A=+1
#         torch.ones(N_pn) * -1,     # A=-1
#         torch.ones(N_np),          # A=+1
#         torch.ones(N_nn) * -1      # A=-1
#     ])

#     # 5. 画像生成と結合
#     # 正規化: [0, 1] に統一 (ColoredMNIST, WaterBirdsと同じ)
    
#     # 目標サイズ
#     target_h, target_w = image_size, image_size # 通常224
    
#     # --- レイアウト設定 ---
#     # MNISTを小さく，CIFARを大きくする
#     # MNISTの高さ (48px)
#     mnist_fixed_h = 48 
#     # CIFARの高さ (残り全部 = 176px)
#     cifar_fixed_h = target_h - mnist_fixed_h 
    
#     # --- MNIST処理 ---
#     # (N, 28, 28) -> Resize(48, 48) -> RGB -> Padding to (48, 224)
#     mnist_raw = mnist_images[mnist_indices_all].float() / 255.0
#     mnist_raw = mnist_raw.unsqueeze(1) # (N, 1, 28, 28)
    
#     # 1. 縦横48x48にリサイズ (アスペクト比維持のため正方形に)
#     resize_mnist = transforms.Resize((mnist_fixed_h, mnist_fixed_h), antialias=True)
#     mnist_resized = resize_mnist(mnist_raw) # (N, 1, 48, 48)
    
#     # 2. RGB化
#     mnist_rgb = torch.cat([mnist_resized]*3, dim=1) # (N, 3, 48, 48)
    
#     # 3. 左右をパディングして幅224にする (中央配置)
#     pad_left = (target_w - mnist_fixed_h) // 2
#     pad_right = target_w - mnist_fixed_h - pad_left
#     # F.pad引数: (left, right, top, bottom)
#     mnist_final = F.pad(mnist_rgb, (pad_left, pad_right, 0, 0), value=0) # (N, 3, 48, 224)

#     # --- CIFAR処理 ---
#     # (N, 32, 32, 3) numpy -> Tensor(N, 3, 32, 32) -> Resize(176, 224)
#     cifar_raw_np = cifar_images[cifar_indices_all]
#     cifar_tensor = torch.from_numpy(cifar_raw_np).float().permute(0, 3, 1, 2) / 255.0 # (N, 3, 32, 32)

#     # 画面下部に引き伸ばす (解像度重視)
#     resize_cifar = transforms.Resize((cifar_fixed_h, target_w), antialias=True)
#     cifar_final = resize_cifar(cifar_tensor) # (N, 3, 176, 224)

#     # --- 結合 ---
#     # 縦方向 (dim=2) に結合 -> (N, 3, 224, 224)
#     # 上: MNIST, 下: CIFAR
#     final_images = torch.cat([mnist_final, cifar_final], dim=2)
    
#     # シャッフル
#     perm = torch.randperm(num_samples)
    
#     return final_images[perm], y_labels[perm], a_labels[perm]

# def get_dominoes_all(config):
#     """ Dominoesデータセット (Train/Test) を取得 """
#     print("Preparing Dominoes dataset...")
#     # ResNet/ViTなどの特徴抽出器に入力するため224x224に統一
#     image_size = 224 
    
#     # データロード
#     mnist_train = MNIST('./data', train=True, download=True)
#     mnist_test = MNIST('./data', train=False, download=True)
#     cifar_train = CIFAR10('./data', train=True, download=True)
#     cifar_test = CIFAR10('./data', train=False, download=True)

#     # Train
#     print("Generating Dominoes Train set...")
#     X_train, y_train, a_train = create_dominoes_dataset(
#         mnist_train.data, mnist_train.targets, 
#         cifar_train.data, cifar_train.targets,
#         config['num_train_samples'], 
#         config.get('train_label_marginal', 0.0),
#         config.get('train_attribute_marginal', 0.0),
#         config['train_correlation'],
#         image_size, seed=42
#     )

#     # Val (Validationは作成しない)

#     # Test (Testソースから生成)
#     print("Generating Dominoes Test set...")
#     X_test, y_test, a_test = create_dominoes_dataset(
#         mnist_test.data, mnist_test.targets, 
#         cifar_test.data, cifar_test.targets,
#         config['num_test_samples'], 
#         config.get('test_label_marginal', 0.0),
#         config.get('test_attribute_marginal', 0.0),
#         config['test_correlation'],
#         image_size, seed=2023
#     )
    
#     return X_train, y_train, a_train, X_test, y_test, a_test












# sp/data_loader.py

import torch
import os
import numpy as np
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import zipfile
import pandas as pd
from PIL import Image
import sys
import torch.nn.functional as F

# # 'wilds'ライブラリのインポート
# try:
#     import wilds
# except ImportError:
#     print("Warning: 'wilds' library not found. WaterBirds dataset will not be available.")
#     print("Please install it using: pip install wilds")
#     wilds = None

def colorize_mnist(images_all, labels_0_9_all, num_samples, label_marginal, attribute_marginal, correlation, exclude_indices=None):
    """ 
    MNISTデータセットに色付けを行い，指定された統計量を持つデータを作成
    
    Args:
        images_all (torch.Tensor): MNISTの全画像テンソル (N_all, H, W)
        labels_0_9_all (torch.Tensor): MNISTの全ラベルテンソル (N_all,)
        num_samples (int): 生成するサンプル数 (N)
        label_marginal (float): 目標とするラベルの期待値 E[Y] (y_bar)
        attribute_marginal (float): 目標とする属性の期待値 E[A] (a_bar)
        correlation (float): 目標とする共分散 Cov(Y, A) (rho_train)
        exclude_indices (list or array, optional): 使用を避ける元画像のインデックス
    """
    
    y_bar = label_marginal
    a_bar = attribute_marginal
    rho_train = correlation
    
    # 1. 元のデータを Y = {-1, +1} に変換
    labels_pm1_all = (labels_0_9_all >= 5).float() * 2.0 - 1.0
    
    # 2. Y=+1 と Y=-1 の利用可能な全インデックスを取得
    all_indices_y_pos = torch.where(labels_pm1_all == 1.0)[0].numpy()
    all_indices_y_neg = torch.where(labels_pm1_all == -1.0)[0].numpy()

    # 除外リストがある場合，利用可能プールから削除
    if exclude_indices is not None:
        exclude_set = set(exclude_indices)
        all_indices_y_pos = np.array([idx for idx in all_indices_y_pos if idx not in exclude_set])
        all_indices_y_neg = np.array([idx for idx in all_indices_y_neg if idx not in exclude_set])

    # 3. 4つのグループの確率(pi_g)と目標サンプル数(N_g)を計算
    # E[YA] = Cov(Y,A) + E[Y]E[A]
    E_ya = rho_train + y_bar * a_bar
    
    # (Y, A) = (+1, +1), (+1, -1), (-1, +1), (-1, -1)
    # 4*pi(y,a) = 1 + y*E[Y] + a*E[A] + (y*a)*E[YA]
    pi_pp = (1 + y_bar + a_bar + E_ya) / 4.0
    pi_pn = (1 + y_bar - a_bar - E_ya) / 4.0
    pi_np = (1 - y_bar + a_bar - E_ya) / 4.0
    pi_nn = (1 - y_bar - a_bar + E_ya) / 4.0
    
    probabilities = [pi_pp, pi_pn, pi_np, pi_nn]
    
    # 確率が [0, 1] の範囲外になる場合はエラー
    if not all(p >= -1e-6 for p in probabilities): # 浮動小数点誤差を許容
        raise ValueError(f"Invalid statistics (y_bar={y_bar}, a_bar={a_bar}, rho={rho_train}) result in negative probabilities: {probabilities}")
    if abs(sum(probabilities) - 1.0) > 1e-6:
        raise ValueError(f"Probabilities do not sum to 1: {probabilities}")

    # 目標サンプル数を計算 (丸め処理)
    N_pp = int(np.round(num_samples * pi_pp))
    N_pn = int(np.round(num_samples * pi_pn))
    N_np = int(np.round(num_samples * pi_np))
    N_nn = int(np.round(num_samples * pi_nn))
    
    # 丸め誤差を調整 (最大のグループに押し付ける)
    counts = [N_pp, N_pn, N_np, N_nn]
    N_total_calc = sum(counts)
    if N_total_calc != num_samples:
        diff = num_samples - N_total_calc
        max_idx = np.argmax(counts)
        counts[max_idx] += diff
    
    N_pp, N_pn, N_np, N_nn = counts[0], counts[1], counts[2], counts[3]

    # --- 利用可能枚数のチェックとキャッピング (DFR Validation用) ---
    # Y=+1 グループが必要とする合計 (A=+1 と A=-1 の合計)
    N_pos_needed = N_pp + N_pn
    # Y=-1 グループが必要とする合計
    N_neg_needed = N_np + N_nn

    # 不足している場合の処理 (サンプル数を縮小する)
    ratio_pos = len(all_indices_y_pos) / N_pos_needed if N_pos_needed > 0 else 1.0
    ratio_neg = len(all_indices_y_neg) / N_neg_needed if N_neg_needed > 0 else 1.0
    min_ratio = min(ratio_pos, ratio_neg)

    if min_ratio < 1.0:
        print(f"  [Warning] Not enough samples available in the remainder pool.")
        print(f"    Requested: Y=+1 -> {N_pos_needed}, Y=-1 -> {N_neg_needed}")
        print(f"    Available: Y=+1 -> {len(all_indices_y_pos)}, Y=-1 -> {len(all_indices_y_neg)}")
        print(f"    Scaling down total samples by factor {min_ratio:.4f}")
        
        N_pp = int(N_pp * min_ratio)
        N_pn = int(N_pn * min_ratio)
        N_np = int(N_np * min_ratio)
        N_nn = int(N_nn * min_ratio)
        
        # 再計算
        N_pos_needed = N_pp + N_pn
        N_neg_needed = N_np + N_nn
        num_samples = N_pp + N_pn + N_np + N_nn
        print(f"    New Total: {num_samples} (N_pp={N_pp}, N_pn={N_pn}, N_np={N_np}, N_nn={N_nn})")

    # 5. 利用可能なサンプル数で足りるかチェック (上記でスケーリング済みだが念のため)
    if N_pos_needed > len(all_indices_y_pos):
        raise ValueError(f"Cannot sample {N_pos_needed} (Y=+1) samples. Only {len(all_indices_y_pos)} available in the source dataset.")
    if N_neg_needed > len(all_indices_y_neg):
        raise ValueError(f"Cannot sample {N_neg_needed} (Y=-1) samples. Only {len(all_indices_y_neg)} available in the source dataset.")

    # 6. Y=+1, Y=-1 グループからランダムサンプリング
    indices_y_pos_sample = np.random.choice(all_indices_y_pos, N_pos_needed, replace=False)
    indices_y_neg_sample = np.random.choice(all_indices_y_neg, N_neg_needed, replace=False)

    # 7. 属性(A)を割り当て
    # Y=+1 グループ
    indices_pp = indices_y_pos_sample[:N_pp] # A = +1 (Red)
    indices_pn = indices_y_pos_sample[N_pp:] # A = -1 (Green)
    # Y=-1 グループ
    indices_np = indices_y_neg_sample[:N_np] # A = +1 (Red)
    indices_nn = indices_y_neg_sample[N_np:] # A = -1 (Green)
    
    # 8. 最終的なインデックスと属性(A)リストを作成
    final_indices = np.concatenate([indices_pp, indices_pn, indices_np, indices_nn])
    
    attributes_pm1 = torch.zeros(len(final_indices), dtype=torch.float32)
    # (Y, A) = (+1, +1)
    start = 0
    attributes_pm1[start : start+N_pp] = 1.0; start += N_pp
    # (Y, A) = (+1, -1)
    attributes_pm1[start : start+N_pn] = -1.0; start += N_pn
    # (Y, A) = (-1, +1)
    attributes_pm1[start : start+N_np] = 1.0; start += N_np
    # (Y, A) = (-1, -1)
    attributes_pm1[start : start+N_nn] = -1.0; start += N_nn

    # 9. 最終的なデータセットを構築
    images_subset = images_all[final_indices]
    labels_pm1_subset = labels_pm1_all[final_indices]
    
    # 10. 色付け処理
    images_gray = images_subset.float() / 255.0
    images_rgb = torch.stack([images_gray, images_gray, images_gray], dim=1)
    n_samples_final = len(images_subset) # = num_samples

    digit_mask = (images_gray > 0.01).unsqueeze(1)
    color_factors = torch.ones(n_samples_final, 3, 1, 1, dtype=images_gray.dtype)

    red_indices = (attributes_pm1 == 1.0)
    green_indices = (attributes_pm1 == -1.0)

    color_factors[red_indices, 1, :, :] = 0.25
    color_factors[red_indices, 2, :, :] = 0.25
    color_factors[green_indices, 0, :, :] = 0.25
    color_factors[green_indices, 2, :, :] = 0.25

    colored_images = images_rgb * color_factors
    final_images_rgb = torch.where(digit_mask, colored_images, images_rgb)
    
    # 11. データをシャッフルして返す
    shuffle_perm = torch.randperm(n_samples_final)
    
    # 使用したインデックスも返す（重複防止用）
    return final_images_rgb[shuffle_perm], labels_pm1_subset[shuffle_perm], attributes_pm1[shuffle_perm], final_indices

def get_colored_mnist(num_samples, correlation, label_marginal, attribute_marginal, train=True, exclude_indices=None):
    """ ColoredMNISTデータセットをロードして生成 """
    set_name = 'train' if train else 'test'
    if exclude_indices is None:
        print(f"Preparing Colored MNIST for {set_name} set...")
        print(f"  Target Stats: N={num_samples}, E[Y]={label_marginal}, E[A]={attribute_marginal}, Cov(Y,A)={correlation}")
    else:
        print(f"Preparing Colored MNIST for DFR Validation (from remainder)...")
        print(f"  Target Stats: N={num_samples} (Balanced), Excluded Indices: {len(exclude_indices)}")
    
    mnist_dataset = MNIST('./data', train=train, download=True)

    # 元のMNISTデータセット全体を読み込む
    images = mnist_dataset.data
    targets = mnist_dataset.targets

    return colorize_mnist(images, targets, num_samples, label_marginal, attribute_marginal, correlation, exclude_indices)

def get_colored_mnist_all(config):
    """ 
    CMNISTの全データセットを取得 (Train, DFR Val, Test) 
    use_dfr=True の場合，Trainの残りからValidationを作成する
    """
    image_size = 28
    train_y_bar = config.get('train_label_marginal', 0.0)
    train_a_bar = config.get('train_attribute_marginal', 0.0)
    test_y_bar = config.get('test_label_marginal', 0.0)
    test_a_bar = config.get('test_attribute_marginal', 0.0)

    # 1. Train Set (Biased)
    X_train, y_train, a_train, train_indices_used = get_colored_mnist(
        num_samples=config['num_train_samples'],
        correlation=config['train_correlation'],
        label_marginal=train_y_bar,
        attribute_marginal=train_a_bar,
        train=True
    )
    
    # 2. DFR Validation Set (Balanced from Remainder)
    X_val, y_val, a_val = None, None, None
    
    if config.get('use_dfr', False):
        samples_per_group = config.get('dfr_val_samples_per_group', 100)
        total_val_samples = samples_per_group * 4
        
        # Trainで使用したインデックスを除外して生成
        # DFR用は相関なし (correlation=0), マージナル0 (均衡) で作成
        X_val, y_val, a_val, _ = get_colored_mnist(
            num_samples=total_val_samples,
            correlation=0.0,
            label_marginal=0.0,
            attribute_marginal=0.0,
            train=True, # Trainソースからサンプリング
            exclude_indices=train_indices_used
        )
        print(f"Generated DFR Validation Set from remaining Train data: {len(X_val)} samples.")

    # 3. Test Set
    X_test, y_test, a_test, _ = get_colored_mnist(
        num_samples=config['num_test_samples'],
        correlation=config['test_correlation'],
        label_marginal=test_y_bar,
        attribute_marginal=test_a_bar,
        train=False
    )
    
    return X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test


def get_waterbirds_dataset(num_train, num_test, image_size):
    """ WaterBirdsデータセットをロード (Train, Val, Test) """
    # 公式のValidationセット (split=1) もロードする
    
    data_dir = 'data/waterbirds_v1.0'
    archive_path = os.path.join(data_dir, 'archive.zip')
    unzip_dir = os.path.join(data_dir, 'waterbird') # 展開後のディレクトリ名
    metadata_path = os.path.join(unzip_dir, 'metadata.csv')

    # 展開済みのデータが存在しない場合
    if not os.path.exists(metadata_path):
        # zipファイル自体も存在しない場合，ダウンロードを促す
        if not os.path.exists(archive_path):
            print("="*80)
            print("Waterbirds dataset not found.")
            print("Please download 'archive.zip' from the following URL:")
            print("https://www.kaggle.com/datasets/bahardibaie/waterbird?resource=download")
            print(f"And place it in the following directory: {data_dir}")
            print("="*80)
            sys.exit(1) # プログラムを停止
        
        # zipファイルが存在する場合は展開する
        print(f"Extracting {archive_path}...")
        os.makedirs(data_dir, exist_ok=True)
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")

    # メタデータを読み込む
    metadata_df = pd.read_csv(metadata_path)

    # CMNIST と同様に [0, 1] スケーリング (ToTensor) のみを行う
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    
    def get_data_from_split(split_id, num_samples):
        # split_id: 0=Train, 1=Val, 2=Test
        split_df = metadata_df[metadata_df['split'] == split_id]
        
        # num_samplesが指定されていればその数だけ，なければ全部 (Noneの場合)
        if num_samples is not None:
            num_to_sample = min(num_samples, len(split_df))
            sampled_df = split_df.sample(n=num_to_sample, random_state=42) 
        else:
            sampled_df = split_df # 全データ
        
        images, y_labels, a_labels = [], [], []
        
        # 画像読み込みループ
        for _, row in sampled_df.iterrows():
            img_path = os.path.join(unzip_dir, row['img_filename'])
            image = Image.open(img_path).convert('RGB')
            images.append(transform(image))
            
            y_labels.append(row['y'])
            a_labels.append(row['place'])
        
        if len(images) == 0:
            return None, None, None
            
        return torch.stack(images), torch.tensor(y_labels, dtype=torch.long), torch.tensor(a_labels, dtype=torch.long)

    print("Loading Waterbirds Train set...")
    X_train, y_train_01, a_train_01 = get_data_from_split(0, num_train)
    
    print("Loading Waterbirds Validation set (Official)...")
    # Validationは全件読み込む (後でバランス調整する)
    X_val_01, y_val_01, a_val_01 = get_data_from_split(1, num_samples=None)

    print("Loading Waterbirds Test set...")
    X_test, y_test_01, a_test_01 = get_data_from_split(2, num_test)

    # ラベルを-1, +1形式に変換
    y_train_pm1 = y_train_01.float() * 2.0 - 1.0
    a_train_pm1 = a_train_01.float() * 2.0 - 1.0 
    
    if y_val_01 is not None:
        y_val_pm1 = y_val_01.float() * 2.0 - 1.0
        a_val_pm1 = a_val_01.float() * 2.0 - 1.0
    else:
        y_val_pm1, a_val_pm1 = None, None
    
    y_test_pm1 = y_test_01.float() * 2.0 - 1.0
    a_test_pm1 = a_test_01.float() * 2.0 - 1.0

    return X_train, y_train_pm1, a_train_pm1, X_val_01, y_val_pm1, a_val_pm1, X_test, y_test_pm1, a_test_pm1

def create_dominoes_dataset(mnist_images, mnist_targets, cifar_images, cifar_targets, 
                          num_samples, label_marginal, attribute_marginal, correlation, 
                          image_size=224, seed=None, exclude_indices_mnist=None, exclude_indices_cifar=None):
    """
    Dominoesデータセット (MNIST + CIFAR10) を生成
    Top: MNIST (0/1) -> Spurious Attribute (0: -1, 1: +1)
    Bottom: CIFAR10 (Car/Truck) -> Core Label (Car: -1, Truck: +1)
    
    正規化: [0, 1] 範囲の float32 (他のデータセットと一致)
    exclude_indices_*: 使用を避けるインデックス
    Return: X, y, a, used_indices_mnist, used_indices_cifar
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # 1. ターゲットと属性の定義
    # MNIST: 0 -> A=-1, 1 -> A=+1
    # CIFAR: Car(1) -> Y=-1, Truck(9) -> Y=+1
    
    # 2. データのフィルタリング
    mnist_mask_0 = (mnist_targets == 0)
    mnist_mask_1 = (mnist_targets == 1)
    # CIFARのターゲットはリスト形式の場合があるためTensor化
    cifar_tensor_targets = torch.tensor(cifar_targets)
    cifar_mask_car = (cifar_tensor_targets == 1)
    cifar_mask_truck = (cifar_tensor_targets == 9)

    mnist_indices_0 = torch.where(mnist_mask_0)[0].numpy()
    mnist_indices_1 = torch.where(mnist_mask_1)[0].numpy()
    cifar_indices_car = torch.where(cifar_mask_car)[0].numpy()
    cifar_indices_truck = torch.where(cifar_mask_truck)[0].numpy()

    # 除外処理
    if exclude_indices_mnist is not None:
        ex_m = set(exclude_indices_mnist)
        mnist_indices_0 = np.array([i for i in mnist_indices_0 if i not in ex_m])
        mnist_indices_1 = np.array([i for i in mnist_indices_1 if i not in ex_m])
    
    if exclude_indices_cifar is not None:
        ex_c = set(exclude_indices_cifar)
        cifar_indices_car = np.array([i for i in cifar_indices_car if i not in ex_c])
        cifar_indices_truck = np.array([i for i in cifar_indices_truck if i not in ex_c])

    # 3. グループごとのサンプル数計算 (colorize_mnistと同様のロジックを使用)
    y_bar = label_marginal
    a_bar = attribute_marginal
    rho_train = correlation
    
    # E[YA] = Cov(Y,A) + E[Y]E[A]
    E_ya = rho_train + y_bar * a_bar
    
    pi_pp = (1 + y_bar + a_bar + E_ya) / 4.0 # Y=+1 (Truck), A=+1 (1)
    pi_pn = (1 + y_bar - a_bar - E_ya) / 4.0 # Y=+1 (Truck), A=-1 (0)
    pi_np = (1 - y_bar + a_bar - E_ya) / 4.0 # Y=-1 (Car), A=+1 (1)
    pi_nn = (1 - y_bar - a_bar + E_ya) / 4.0 # Y=-1 (Car), A=-1 (0)
    
    probabilities = [pi_pp, pi_pn, pi_np, pi_nn]
    if not all(p >= -1e-6 for p in probabilities):
        raise ValueError(f"Invalid statistics result in negative probabilities: {probabilities}")

    counts = [int(np.round(num_samples * p)) for p in probabilities]
    # 丸め誤差調整
    if sum(counts) != num_samples:
        counts[np.argmax(counts)] += num_samples - sum(counts)
    N_pp, N_pn, N_np, N_nn = counts

    # --- 利用可能数のチェックとキャッピング ---
    # Group PP: Truck(C), 1(M) -> 需要 N_pp
    # Group PN: Truck(C), 0(M) -> 需要 N_pn
    # Group NP: Car(C),   1(M) -> 需要 N_np
    # Group NN: Car(C),   0(M) -> 需要 N_nn
    
    # 必要数
    req_truck = N_pp + N_pn
    req_car = N_np + N_nn
    req_m1 = N_pp + N_np
    req_m0 = N_pn + N_nn
    
    # 利用可能数
    avail_truck = len(cifar_indices_truck)
    avail_car = len(cifar_indices_car)
    avail_m1 = len(mnist_indices_1)
    avail_m0 = len(mnist_indices_0)
    
    ratio_truck = avail_truck / req_truck if req_truck > 0 else 1.0
    ratio_car = avail_car / req_car if req_car > 0 else 1.0
    ratio_m1 = avail_m1 / req_m1 if req_m1 > 0 else 1.0
    ratio_m0 = avail_m0 / req_m0 if req_m0 > 0 else 1.0
    
    min_ratio = min(ratio_truck, ratio_car, ratio_m1, ratio_m0)
    
    if min_ratio < 1.0:
        print(f"  [Warning] Not enough samples in Dominoes source.")
        print(f"    Scaling down total samples by {min_ratio:.4f}")
        N_pp = int(N_pp * min_ratio)
        N_pn = int(N_pn * min_ratio)
        N_np = int(N_np * min_ratio)
        N_nn = int(N_nn * min_ratio)
    
    # 4. サンプリング (データ数が足りない場合は重複を許容)
    def sample_indices(pool, n):
        # 重複なしサンプリングを基本とする
        replace = len(pool) < n
        return np.random.choice(pool, n, replace=replace)

    # Pairs: (CIFAR, MNIST)
    # Group PP: Truck (+1), 1 (+1)
    idx_c_pp = sample_indices(cifar_indices_truck, N_pp)
    idx_m_pp = sample_indices(mnist_indices_1, N_pp)
    
    # Group PN: Truck (+1), 0 (-1)
    idx_c_pn = sample_indices(cifar_indices_truck, N_pn)
    idx_m_pn = sample_indices(mnist_indices_0, N_pn)
    
    # Group NP: Car (-1), 1 (+1)
    idx_c_np = sample_indices(cifar_indices_car, N_np)
    idx_m_np = sample_indices(mnist_indices_1, N_np)

    # Group NN: Car (-1), 0 (-1)
    idx_c_nn = sample_indices(cifar_indices_car, N_nn)
    idx_m_nn = sample_indices(mnist_indices_0, N_nn)

    # 統合
    cifar_indices_sample = np.concatenate([idx_c_pp, idx_c_pn, idx_c_np, idx_c_nn])
    mnist_indices_sample = np.concatenate([idx_m_pp, idx_m_pn, idx_m_np, idx_m_nn])
    
    # ラベルと属性の作成 (-1, +1 形式)
    # Y: -1 (Car), +1 (Truck)
    # A: -1 (0), +1 (1)
    y_labels = torch.cat([
        torch.ones(N_pp + N_pn),     # Truck (+1)
        torch.ones(N_np + N_nn) * -1 # Car (-1)
    ])
    a_labels = torch.cat([
        torch.ones(N_pp),          # A=+1
        torch.ones(N_pn) * -1,     # A=-1
        torch.ones(N_np),          # A=+1
        torch.ones(N_nn) * -1      # A=-1
    ])

    # 5. 画像生成と結合
    # 正規化: [0, 1] に統一 (ColoredMNIST, WaterBirdsと同じ)
    
    # 目標サイズ
    target_h, target_w = image_size, image_size # 通常224
    
    # --- レイアウト設定 ---
    # MNISTを小さく，CIFARを大きくする
    # MNISTの高さ (48px)
    mnist_fixed_h = 48 
    # CIFARの高さ (残り全部 = 176px)
    cifar_fixed_h = target_h - mnist_fixed_h 
    
    # --- MNIST処理 ---
    # (N, 28, 28) -> Resize(48, 48) -> RGB -> Padding to (48, 224)
    mnist_raw = mnist_images[mnist_indices_sample].float() / 255.0
    mnist_raw = mnist_raw.unsqueeze(1) # (N, 1, 28, 28)
    
    # 1. 縦横48x48にリサイズ (アスペクト比維持のため正方形に)
    resize_mnist = transforms.Resize((mnist_fixed_h, mnist_fixed_h), antialias=True)
    mnist_resized = resize_mnist(mnist_raw) # (N, 1, 48, 48)
    
    # 2. RGB化
    mnist_rgb = torch.cat([mnist_resized]*3, dim=1) # (N, 3, 48, 48)
    
    # 3. 左右をパディングして幅224にする (中央配置)
    pad_left = (target_w - mnist_fixed_h) // 2
    pad_right = target_w - mnist_fixed_h - pad_left
    # F.pad引数: (left, right, top, bottom)
    mnist_final = F.pad(mnist_rgb, (pad_left, pad_right, 0, 0), value=0) # (N, 3, 48, 224)

    # --- CIFAR処理 ---
    # (N, 32, 32, 3) numpy -> Tensor(N, 3, 32, 32) -> Resize(176, 224)
    cifar_raw_np = cifar_images[cifar_indices_sample]
    cifar_tensor = torch.from_numpy(cifar_raw_np).float().permute(0, 3, 1, 2) / 255.0 # (N, 3, 32, 32)

    # 画面下部に引き伸ばす (解像度重視)
    resize_cifar = transforms.Resize((cifar_fixed_h, target_w), antialias=True)
    cifar_final = resize_cifar(cifar_tensor) # (N, 3, 176, 224)

    # --- 結合 ---
    # 縦方向 (dim=2) に結合 -> (N, 3, 224, 224)
    # 上: MNIST, 下: CIFAR
    final_images = torch.cat([mnist_final, cifar_final], dim=2)
    
    # シャッフル
    perm = torch.randperm(len(y_labels))
    
    return final_images[perm], y_labels[perm], a_labels[perm], mnist_indices_sample, cifar_indices_sample

def get_dominoes_all(config):
    """ Dominoesデータセット (Train/Test) を取得 """
    print("Preparing Dominoes dataset...")
    # ResNet/ViTなどの特徴抽出器に入力するため224x224に統一
    image_size = 224 
    
    # データロード
    mnist_train = MNIST('./data', train=True, download=True)
    mnist_test = MNIST('./data', train=False, download=True)
    cifar_train = CIFAR10('./data', train=True, download=True)
    cifar_test = CIFAR10('./data', train=False, download=True)

    # Train
    print("Generating Dominoes Train set...")
    X_train, y_train, a_train, used_m_train, used_c_train = create_dominoes_dataset(
        mnist_train.data, mnist_train.targets, 
        cifar_train.data, cifar_train.targets,
        config['num_train_samples'], 
        config.get('train_label_marginal', 0.0),
        config.get('train_attribute_marginal', 0.0),
        config['train_correlation'],
        image_size, seed=42
    )

    # Val (DFR用，残りからバランス作成)
    X_val, y_val, a_val = None, None, None
    if config.get('use_dfr', False):
        print("Generating Dominoes DFR Validation set from remainder...")
        samples_per_group = config.get('dfr_val_samples_per_group', 100)
        total_val_samples = samples_per_group * 4
        
        X_val, y_val, a_val, _, _ = create_dominoes_dataset(
            mnist_train.data, mnist_train.targets, 
            cifar_train.data, cifar_train.targets,
            total_val_samples, 
            0.0, 0.0, 0.0, # Balanced
            image_size, seed=999,
            exclude_indices_mnist=used_m_train,
            exclude_indices_cifar=used_c_train
        )

    # Test (Testソースから生成)
    print("Generating Dominoes Test set...")
    X_test, y_test, a_test, _, _ = create_dominoes_dataset(
        mnist_test.data, mnist_test.targets, 
        cifar_test.data, cifar_test.targets,
        config['num_test_samples'], 
        config.get('test_label_marginal', 0.0),
        config.get('test_attribute_marginal', 0.0),
        config['test_correlation'],
        image_size, seed=2023
    )
    
    return X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test

def balance_dataset_by_group_size(X, y, a, target_samples_per_group):
    """
    指定されたグループごとのサンプル数になるようにデータセットをダウンサンプリングする
    指定数より少ないグループがある場合は，最小のグループサイズに合わせて全てのグループを揃え，警告を出す
    """
    if X is None: return None, None, None
    
    group_keys = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    indices_by_group = {}
    min_available = float('inf')
    
    for g in group_keys:
        y_val, a_val = g
        mask = (y == y_val) & (a == a_val)
        indices = torch.where(mask)[0]
        indices_by_group[g] = indices
        if len(indices) < min_available:
            min_available = len(indices)
            
    final_per_group = target_samples_per_group
    
    if min_available < target_samples_per_group:
        print(f"  [Warning] Requested {target_samples_per_group} samples per group, but smallest group has only {min_available}.")
        print(f"  Using {min_available} samples per group (Balanced) instead.")
        final_per_group = min_available
        
    selected_indices_list = []
    for g in group_keys:
        indices = indices_by_group[g]
        # ランダムシャッフルして先頭から取得
        perm = torch.randperm(len(indices))
        selected = indices[perm[:final_per_group]]
        selected_indices_list.append(selected)
        
    all_selected = torch.cat(selected_indices_list)
    # 最後に全体をシャッフル
    final_perm = torch.randperm(len(all_selected))
    final_indices = all_selected[final_perm]
    
    return X[final_indices], y[final_indices], a[final_indices]
