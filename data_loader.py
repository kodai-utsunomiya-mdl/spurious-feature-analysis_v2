# sp/data_loader.py

import torch
import os
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, Subset
import zipfile
import pandas as pd
from PIL import Image
import sys # sysモジュールをインポート

# 'wilds'ライブラリのインポート
try:
    import wilds
except ImportError:
    print("Warning: 'wilds' library not found. WaterBirds dataset will not be available.")
    print("Please install it using: pip install wilds")
    wilds = None

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


def get_waterbirds_dataset(num_train, num_test, image_size):
    """ WaterBirdsデータセットをKaggleから手動ダウンロードしたファイルを使ってロード """
    
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

    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    
    def get_data_from_split(split_id, num_samples):
        split_df = metadata_df[metadata_df['split'] == split_id]
        
        num_to_sample = min(num_samples, len(split_df))
        sampled_df = split_df.sample(n=num_to_sample, random_state=42) 
        
        images, y_labels, a_labels = [], [], []
        
        for _, row in sampled_df.iterrows():
            img_path = os.path.join(unzip_dir, row['img_filename'])
            image = Image.open(img_path).convert('RGB')
            images.append(transform(image))
            
            y_labels.append(row['y'])
            a_labels.append(row['place'])
            
        return torch.stack(images), torch.tensor(y_labels, dtype=torch.long), torch.tensor(a_labels, dtype=torch.long)

    # split_id: 0 for train, 1 for val, 2 for test
    X_train, y_train_01, a_train_01 = get_data_from_split(0, num_train)
    X_test, y_test_01, a_test_01 = get_data_from_split(2, num_test)

    # ラベルを-1, +1形式に変換
    y_train_pm1 = y_train_01.float() * 2.0 - 1.0
    a_train_pm1 = a_train_01.float() * 2.0 - 1.0 
    y_test_pm1 = y_test_01.float() * 2.0 - 1.0
    a_test_pm1 = a_test_01.float() * 2.0 - 1.0

    return X_train, y_train_pm1, a_train_pm1, X_test, y_test_pm1, a_test_pm1
