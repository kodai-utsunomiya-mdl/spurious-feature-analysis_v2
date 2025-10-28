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

def colorize_mnist(images, labels, correlation):
    """ MNISTデータセットに色付けを行い，ラベルと色の相関を持つデータを作成 """
    labels_pm1 = (labels >= 5).float() * 2.0 - 1.0
    images_gray = images.float() / 255.0
    n_samples = len(labels_pm1)

    images_rgb = torch.stack([images_gray, images_gray, images_gray], dim=1)

    prob_color_matches_label = (1.0 + correlation) / 2.0
    attributes_pm1 = torch.zeros_like(labels_pm1)

    for i in range(n_samples):
        y_i = labels_pm1[i]
        if torch.rand(1) < prob_color_matches_label:
            a_i = y_i
        else:
            a_i = -y_i
        attributes_pm1[i] = a_i

    digit_mask = (images_gray > 0.01).unsqueeze(1)
    color_factors = torch.ones(n_samples, 3, 1, 1, dtype=images_gray.dtype)

    red_indices = (attributes_pm1 == 1.0)
    green_indices = (attributes_pm1 == -1.0)

    color_factors[red_indices, 1, :, :] = 0.25
    color_factors[red_indices, 2, :, :] = 0.25
    color_factors[green_indices, 0, :, :] = 0.25
    color_factors[green_indices, 2, :, :] = 0.25

    colored_images = images_rgb * color_factors
    final_images_rgb = torch.where(digit_mask, colored_images, images_rgb)

    return final_images_rgb, labels_pm1, attributes_pm1

def get_colored_mnist(num_samples, correlation, train=True):
    """ ColoredMNISTデータセットをロードして生成 """
    print(f"Preparing Colored MNIST for {'train' if train else 'test'} set...")
    mnist_dataset = MNIST('./data', train=train, download=True)

    images = mnist_dataset.data[:num_samples]
    targets = mnist_dataset.targets[:num_samples]

    return colorize_mnist(images, targets, correlation)

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
