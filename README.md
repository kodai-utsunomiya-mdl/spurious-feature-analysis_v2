# スプリアス相関下の勾配流と性能差ダイナミクスの分析

## 1. 概要

スプリアス相関が存在する状況下での深層ニューラルネットワークの学習ダイナミクスを分析するための研究コード．

主な目的は，訓練データに存在するバイアスが，学習の過程でどのようにしてグループ間（多数派・少数派）の性能差を生み出し，拡大・残存するかを理論的・実証的に解明すること．

## 2. 主な機能

* **データセット**: `ColoredMNIST` と `WaterBirds` に対応．
* **特徴抽出**:
    * 生のピクセルデータ，または事前学習済みモデルから抽出した特徴量の上でMLPを学習させることが可能．
    * 特徴抽出器として `ResNet18/50`，`ViT_B_16`，および `DINOv2`（`S`, `B`, `L`, `G`）の各バリアントをサポート．
    * `ResNet` や `ViT`/`DINOv2` の任意の中間ブロックから特徴を抽出可能．
* **学習手法**:
    * 標準的なERM ( `debias_method: "None"` )
    * Importance Weighting（均一な目標分布，`v_inv`勾配流に対応） ( `debias_method: "IW_uniform"` )
    * Group Distributionally Robust Optimization (GroupDRO) ( `debias_method: "GroupDRO"` )
* **詳細な分析**:
    * 学習中の任意のチェックポイントで `analysis.py` の詳細な分析（勾配基底，ヤコビアンノルム，静的・動的な分解など）を実行．
* **ロギング**:
    * 実験結果は `wandb`（Weights & Biases）にロギングされる．

## 3. セットアップ

### a. 依存関係のインストール
```bash
uv pip install torch torchvision numpy pandas matplotlib scikit-learn pyyaml pot wandb
````

### b. データセットの準備

#### ColoredMNIST

`data_loader.py` により `./data` ディレクトリに自動的にダウンロードされる．

#### WaterBirds

WaterBirdsデータセットは，`wilds` ライブラリではなく，Kaggleからの手動ダウンロードが必要．

1.  [KaggleのWaterbirdデータセットページ](https://www.kaggle.com/datasets/bahardibaie/waterbird?resource=download)にアクセス．
2.  `archive.zip` をダウンロード．
3.  プロジェクトルートに `data/waterbirds_v1.0/` ディレクトリを作成し，そこに `archive.zip` を配置．
      * 最終的なパス: `data/waterbirds_v1.0/archive.zip`
4.  初回実行時に，`data_loader.py` が自動的にzipファイルを展開する．

### c. DINOv2モデル

`DINOv2` を特徴抽出器として使用する場合，初回実行時に `torch.hub` を介してモデルの重みがダウンロードされる．インターネット接続が必要．

## 4\. 実験の実行

すべての実験設定は `config.yaml` ファイルで管理する．

1.  **設定の編集**: `config.yaml` を開き，使用するデータセット，モデル，学習手法，分析フラグなどを設定．
2.  **メインスクリプトの実行**:
    ```bash
    python main.py
    ```
3.  **結果の確認**:
      * コンソールに進捗が出力される．
      * 詳細なメトリクスは `wandb` のダッシュボードで確認できる．
      * 最終的なプロットは、実行後に生成される `results/<experiment_name>_<timestamp>/` ディレクトリ内に保存される．

## 5\. `config.yaml` の設定

`config.yaml` ファイルには，実験を制御するための主要なパラメータが含まれている．

### 基本設定

  * `experiment_name`: 実験名．`results` ディレクトリや `wandb` で使用される．
  * `dataset_name`: `ColoredMNIST` または `WaterBirds` を指定．
  * `loss_function`: `logistic` または `mse`．
  * `device`: `cuda` または `cpu`．

### 特徴抽出器 (`feature_extractor.py` 関連)

  * `use_feature_extractor`: `true` に設定すると，`feature_extractor_model_name` で指定されたモデルで特徴抽出を行う．`false` の場合，生のピクセルデータを入力とする．
  * `feature_extractor_model_name`: `ResNet18`, `DINOv2_ViT_S_14` などを指定．【理論の観点から推奨: `DINOv2_ViT_G_14`】
  * **ViT/DINOv2系**:
      * `feature_extractor_vit_target_block`: 抽出するブロックのインデックス（-1は最終ブロック）．
      * `feature_extractor_vit_aggregation_mode`: `cls_token`, `mean_pool_patch`, `mean_pool_all` から選択．
  * **ResNet系**:
      * `feature_extractor_resnet_intermediate_layer`: `avgpool`, `layer3`, `layer4` から選択．

### モデル (`model.py` 関連)

  * `num_hidden_layers`, `hidden_dim`: 特徴量の上で学習するMLPの構造．
  * `initialization_method`: `muP` または `NTP` を指定．

### 学習 (`trainer.py` 関連)

  * `epochs`: 総エポック数．
  * `debias_method`: `None` (ERM), `IW_uniform` (v\_inv勾配流), `GroupDRO` から選択．
  * `dro_eta_q`: GroupDROのグループ重み `q` の更新ステップサイズ．

### 解析 (`analysis.py` 関連)

  * `analyze_jacobian_norm`, `analyze_gradient_basis`, `analyze_gap_dynamics_factors`, `analyze_static_dynamic_decomposition`: これらのフラグを `true` にすると，対応する詳細な分析が実行される．
  * `..._analysis_epochs`: `null`（またはキーを省略）の場合，毎エポック分析を実行する．`[10, 100, 1000]` のようにリストを指定すると，該当エポックでのみ分析を実行する．
  * その他．
