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

def colorize_mnist(images_all, labels_0_9_all, num_samples, label_marginal, attribute_marginal, correlation):
    """ 
    MNISTデータセットに色付けを行い，指定された統計量を持つデータを作成
    
    Args:
        images_all (torch.Tensor): MNISTの全画像テンソル (N_all, H, W)
        labels_0_9_all (torch.Tensor): MNISTの全ラベルテンソル (N_all,)
        num_samples (int): 生成するサンプル数 (N)
        label_marginal (float): 目標とするラベルの期待値 E[Y] (y_bar)
        attribute_marginal (float): 目標とする属性の期待値 E[A] (a_bar)
        correlation (float): 目標とする共分散 Cov(Y, A) (rho_train)
    """
    
    y_bar = label_marginal
    a_bar = attribute_marginal
    rho_train = correlation
    
    # 1. 元のデータを Y = {-1, +1} に変換
    labels_pm1_all = (labels_0_9_all >= 5).float() * 2.0 - 1.0
    
    # 2. Y=+1 と Y=-1 の利用可能な全インデックスを取得
    all_indices_y_pos = torch.where(labels_pm1_all == 1.0)[0]
    all_indices_y_neg = torch.where(labels_pm1_all == -1.0)[0]

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

    # 4. Yごとの必要枚数を計算
    N_pos_target = N_pp + N_pn
    N_neg_target = N_np + N_nn

    # 5. 利用可能なサンプル数で足りるかチェック
    if N_pos_target > len(all_indices_y_pos):
        raise ValueError(f"Cannot sample {N_pos_target} (Y=+1) samples. Only {len(all_indices_y_pos)} available in the source dataset.")
    if N_neg_target > len(all_indices_y_neg):
        raise ValueError(f"Cannot sample {N_neg_target} (Y=-1) samples. Only {len(all_indices_y_neg)} available in the source dataset.")

    # 6. Y=+1, Y=-1 グループからランダムサンプリング
    indices_y_pos_sample = np.random.choice(all_indices_y_pos, N_pos_target, replace=False)
    indices_y_neg_sample = np.random.choice(all_indices_y_neg, N_neg_target, replace=False)

    # 7. 属性(A)を割り当て
    # Y=+1 グループ
    indices_pp = indices_y_pos_sample[:N_pp] # A = +1 (Red)
    indices_pn = indices_y_pos_sample[N_pp:] # A = -1 (Green)
    # Y=-1 グループ
    indices_np = indices_y_neg_sample[:N_np] # A = +1 (Red)
    indices_nn = indices_y_neg_sample[N_np:] # A = -1 (Green)
    
    # 8. 最終的なインデックスと属性(A)リストを作成
    final_indices = np.concatenate([indices_pp, indices_pn, indices_np, indices_nn])
    
    attributes_pm1 = torch.zeros(num_samples, dtype=torch.float32)
    # (Y, A) = (+1, +1)
    attributes_pm1[0:N_pp] = 1.0
    # (Y, A) = (+1, -1)
    attributes_pm1[N_pp:N_pos_target] = -1.0
    # (Y, A) = (-1, +1)
    attributes_pm1[N_pos_target:(N_pos_target + N_np)] = 1.0
    # (Y, A) = (-1, -1)
    attributes_pm1[(N_pos_target + N_np):num_samples] = -1.0

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
    
    return final_images_rgb[shuffle_perm], labels_pm1_subset[shuffle_perm], attributes_pm1[shuffle_perm]

def get_colored_mnist(num_samples, correlation, label_marginal, attribute_marginal, train=True):
    """ ColoredMNISTデータセットをロードして生成 """
    set_name = 'train' if train else 'test'
    print(f"Preparing Colored MNIST for {set_name} set...")
    print(f"  Target Stats: N={num_samples}, E[Y]={label_marginal}, E[A]={attribute_marginal}, Cov(Y,A)={correlation}")
    
    mnist_dataset = MNIST('./data', train=train, download=True)

    # 元のMNISTデータセット全体を読み込む
    images = mnist_dataset.data
    targets = mnist_dataset.targets

    return colorize_mnist(images, targets, num_samples, label_marginal, attribute_marginal, correlation)

def get_colored_mnist_all(config):
    """ 
    CMNISTの全データセットを取得 (Train, Test) 
    Validationは作成しない (use_dfr=TrueのときにTrainから分割する)
    """
    image_size = 28
    train_y_bar = config.get('train_label_marginal', 0.0)
    train_a_bar = config.get('train_attribute_marginal', 0.0)
    test_y_bar = config.get('test_label_marginal', 0.0)
    test_a_bar = config.get('test_attribute_marginal', 0.0)

    X_train, y_train, a_train = get_colored_mnist(
        num_samples=config['num_train_samples'],
        correlation=config['train_correlation'],
        label_marginal=train_y_bar,
        attribute_marginal=train_a_bar,
        train=True
    )
    
    X_test, y_test, a_test = get_colored_mnist(
        num_samples=config['num_test_samples'],
        correlation=config['test_correlation'],
        label_marginal=test_y_bar,
        attribute_marginal=test_a_bar,
        train=False
    )
    
    # Validationセットは返さない
    return X_train, y_train, a_train, X_test, y_test, a_test


def get_waterbirds_dataset(num_train, num_test, image_size):
    """ WaterBirdsデータセットをロード (Train, Test) """
    # 公式のValidationセット (split=1) はロードしない
    
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
        split_df = metadata_df[metadata_df['split'] == split_id]
        
        num_to_sample = min(num_samples, len(split_df))
        # 確実に同じデータを引けるようにシードを固定
        sampled_df = split_df.sample(n=num_to_sample, random_state=42) 
        
        images, y_labels, a_labels = [], [], []
        
        # 画像読み込みループ
        for _, row in sampled_df.iterrows():
            img_path = os.path.join(unzip_dir, row['img_filename'])
            image = Image.open(img_path).convert('RGB')
            images.append(transform(image))
            
            y_labels.append(row['y'])
            a_labels.append(row['place'])
            
        return torch.stack(images), torch.tensor(y_labels, dtype=torch.long), torch.tensor(a_labels, dtype=torch.long)

    print("Loading Waterbirds Train set...")
    X_train, y_train_01, a_train_01 = get_data_from_split(0, num_train)
    
    # Validation (split=1) はスキップ

    print("Loading Waterbirds Test set...")
    X_test, y_test_01, a_test_01 = get_data_from_split(2, num_test)

    # ラベルを-1, +1形式に変換
    y_train_pm1 = y_train_01.float() * 2.0 - 1.0
    a_train_pm1 = a_train_01.float() * 2.0 - 1.0 
    
    y_test_pm1 = y_test_01.float() * 2.0 - 1.0
    a_test_pm1 = a_test_01.float() * 2.0 - 1.0

    return X_train, y_train_pm1, a_train_pm1, X_test, y_test_pm1, a_test_pm1

def create_dominoes_dataset(mnist_images, mnist_targets, cifar_images, cifar_targets, 
                          num_samples, label_marginal, attribute_marginal, correlation, 
                          image_size=224, seed=None):
    """
    Dominoesデータセット (MNIST + CIFAR10) を生成
    Top: MNIST (0/1) -> Spurious Attribute (0: -1, 1: +1)
    Bottom: CIFAR10 (Car/Truck) -> Core Label (Car: -1, Truck: +1)
    
    正規化: [0, 1] 範囲の float32 (他のデータセットと一致)
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

    # 4. サンプリング (データ数が足りない場合は重複を許容)
    def sample_indices(pool, n):
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
    cifar_indices_all = np.concatenate([idx_c_pp, idx_c_pn, idx_c_np, idx_c_nn])
    mnist_indices_all = np.concatenate([idx_m_pp, idx_m_pn, idx_m_np, idx_m_nn])
    
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
    mnist_raw = mnist_images[mnist_indices_all].float() / 255.0
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
    cifar_raw_np = cifar_images[cifar_indices_all]
    cifar_tensor = torch.from_numpy(cifar_raw_np).float().permute(0, 3, 1, 2) / 255.0 # (N, 3, 32, 32)

    # 画面下部に引き伸ばす (解像度重視)
    resize_cifar = transforms.Resize((cifar_fixed_h, target_w), antialias=True)
    cifar_final = resize_cifar(cifar_tensor) # (N, 3, 176, 224)

    # --- 結合 ---
    # 縦方向 (dim=2) に結合 -> (N, 3, 224, 224)
    # 上: MNIST, 下: CIFAR
    final_images = torch.cat([mnist_final, cifar_final], dim=2)
    
    # シャッフル
    perm = torch.randperm(num_samples)
    
    return final_images[perm], y_labels[perm], a_labels[perm]

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
    X_train, y_train, a_train = create_dominoes_dataset(
        mnist_train.data, mnist_train.targets, 
        cifar_train.data, cifar_train.targets,
        config['num_train_samples'], 
        config.get('train_label_marginal', 0.0),
        config.get('train_attribute_marginal', 0.0),
        config['train_correlation'],
        image_size, seed=42
    )

    # Val (Validationは作成しない)

    # Test (Testソースから生成)
    print("Generating Dominoes Test set...")
    X_test, y_test, a_test = create_dominoes_dataset(
        mnist_test.data, mnist_test.targets, 
        cifar_test.data, cifar_test.targets,
        config['num_test_samples'], 
        config.get('test_label_marginal', 0.0),
        config.get('test_attribute_marginal', 0.0),
        config['test_correlation'],
        image_size, seed=2023
    )
    
    return X_train, y_train, a_train, X_test, y_test, a_test
