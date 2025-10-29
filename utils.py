# sp/utils.py

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset, DataLoader

def l2_normalize_images(images_tensor):
    """ バッチ内の各画像をL2ノルムが1になるように正規化 """
    original_shape = images_tensor.shape
    # 画像をフラット化
    images_flat = images_tensor.view(original_shape[0], -1)
    # L2ノルムを計算して正規化
    norms = torch.linalg.norm(images_flat, ord=2, dim=1, keepdim=True)
    images_flat_normalized = images_flat / (norms + 1e-8) # ゼロ除算を防止
    # 元の形状に戻す
    images_normalized = images_flat_normalized.view(original_shape)
    return images_normalized

def show_dataset_samples(X_data, y_data, a_data, dataset_name, save_dir, num_samples=10):
    """ データセットのサンプル画像を表示し，ファイルに保存する """
    print(f"\nDisplaying and saving {num_samples} sample images from the {dataset_name} train set...")
    fig = plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        image, label, attr = X_data[i], y_data[i].item(), a_data[i].item()
        image_to_show = image.permute(1, 2, 0).cpu().numpy()

        if dataset_name == 'ColoredMNIST':
            label_text, attr_text = ("Digit>=5" if label == 1.0 else "Digit<5"), ("Red" if attr == 1.0 else "Green")
        elif dataset_name == 'WaterBirds':
            label_text, attr_text = ("Waterbird" if label == 1.0 else "Landbird"), ("BG:Water" if attr == 1.0 else "BG:Land")
        else:
            label_text, attr_text = f"y={label}", f"a={attr}"

        plt.subplot(2, num_samples // 2, i + 1)
        plt.imshow(image_to_show)
        plt.title(f"Label: {label_text}\nAttr: {attr_text}", fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "dataset_samples.png")
    plt.savefig(save_path)
    print(f"Sample images saved to {save_path}")
    plt.show() # 画面にも表示
    plt.close(fig)


def display_group_distribution(y_data, a_data, set_name, dataset_type, result_dir=None):
    """ データセットのグループごとのサンプル数を表示し，ファイルに保存 """
    header = f"\n--- {set_name} Group Distribution ({dataset_type}) ---"
    
    if dataset_type == 'WaterBirds':
        labels = {
            'Waterbird on Water (y=+1, a=+1)': (1, 1), 'Waterbird on Land (y=+1, a=-1)': (1, -1),
            'Landbird on Water (y=-1, a=+1)': (-1, 1), 'Landbird on Land (y=-1, a=-1)': (-1, -1),
        }
    else: # ColoredMNIST or Fallback
        labels = {
            'Digit>=5, Red (y=+1, a=+1)': (1, 1), 'Digit>=5, Green (y=+1, a=-1)': (1, -1),
            'Digit<5, Red (y=-1, a=+1)': (-1, 1), 'Digit<5, Green (y=-1, a=-1)': (-1, -1),
        }

    group_counts = {name: ((y_data == y_val) & (a_data == a_val)).sum().item()
                    for name, (y_val, a_val) in labels.items()}

    # 表示とファイル書き込み用のテキストを作成
    output_lines = [header]
    for name, count in group_counts.items():
        line = f"{name:<35}: {count:>5} samples"
        output_lines.append(line)
    
    total_line = f"{'Total':<35}: {len(y_data):>5} samples\n"
    output_lines.append(total_line)
    
    output_text = "\n".join(output_lines)
    
    # コンソールに表示
    print(output_text)

    # ファイルに追記
    if result_dir:
        filepath = os.path.join(result_dir, 'data_distribution.txt')
        with open(filepath, 'a', encoding='utf-8') as f:
            # 複数回呼び出された場合のために，区切り線を追加
            f.write(output_text + "\n")

def evaluate_model(model, X_data, y_data, a_data, device, loss_function, eval_batch_size):
    """ モデルの性能を評価し，各種メトリクスを返す """
    model.eval()

    dataset = TensorDataset(X_data, y_data)
    batch_size_to_use = eval_batch_size if eval_batch_size is not None and eval_batch_size < len(X_data) else len(X_data)
    if batch_size_to_use <= 0:
        batch_size_to_use = len(X_data)
        
    loader = DataLoader(dataset, batch_size=batch_size_to_use, shuffle=False)
    
    all_scores = []
    all_y_batch = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            scores_batch, _ = model(X_batch)
            all_scores.append(scores_batch.cpu())
            all_y_batch.append(y_batch.cpu())

    scores = torch.cat(all_scores, dim=0)
    y_data_eval = torch.cat(all_y_batch, dim=0)

    y_01 = ((y_data_eval + 1) / 2).long()
    a_01 = ((a_data + 1) / 2).long()
    group_indices = 2 * y_01 + a_01

    if loss_function == 'logistic':
        losses = F.softplus(-y_data_eval * scores)
    elif loss_function == 'mse':
        losses = (scores - y_data_eval).pow(2)
    else:
        raise ValueError(f"Unknown loss_function: {loss_function}")

    avg_loss = losses.mean().item()
    preds = torch.sign(scores)
    corrects = (preds == y_data_eval).float()
    avg_acc = corrects.mean().item()

    group_losses = torch.full((4,), float('nan'))
    group_accs = torch.full((4,), float('nan'))

    for i in range(4):
        mask = (group_indices == i)
        if mask.sum() > 0:
            group_losses[i] = losses[mask].mean().item()
            group_accs[i] = corrects[mask].mean().item()

    valid_losses = group_losses[~torch.isnan(group_losses)]
    valid_accs = group_accs[~torch.isnan(group_accs)]
    
    worst_loss = valid_losses.max().item() if len(valid_losses) > 0 else 0.0
    worst_acc = valid_accs.min().item() if len(valid_accs) > 0 else 0.0

    return {
        'avg_loss': avg_loss, 'worst_loss': worst_loss,
        'avg_acc': avg_acc, 'worst_acc': worst_acc,
        'group_losses': group_losses.tolist(), 'group_accs': group_accs.tolist(),
    }
