# Analysis of Gradient Flow and Group Performance Gaps under Spurious Correlations

## Overview

This repository provides research code for analyzing the learning dynamics of deep neural networks in the presence of spurious correlations.

The main objective is to theoretically and empirically understand how biases in training data create, amplify, or maintain performance gaps between groups (e.g., majority and minority groups) during the learning process.

## Main Features

* **Datasets**: Supports `ColoredMNIST` and `WaterBirds`.
* **Feature Extraction**:
    * Enables training an MLP on either raw pixel data or features extracted from pre-trained models.
    * Supports various feature extractors including `ResNet18/50`, `ViT_B_16`, and `DINOv2` variants (`S`, `B`,`L`, `G`).
    * Allows feature extraction from any intermediate block of `ResNet` or `ViT`/`DINOv2`.
* **Training Methods**:
    * Standard ERM (`debias_method: "None"`)
    * Importance Weighting (for uniform target distribution, corresponding to `v_inv` gradient flow) (`debias_method: "IW_uniform"`)
    * Group Distributionally Robust Optimization (GroupDRO) (`debias_method: "GroupDRO"`)
* **Deep Feature Reweighting (DFR)**:
    * Supports evaluating the quality of learned features by re-training the last layer (Logistic Regression) on a balanced validation set.
    * Can target any intermediate layer for DFR application (`dfr_target_layer`).
    * Supports flexible validation strategies (using the original validation set or splitting from the training set).
* **In-Depth Analysis**:
    * Executes detailed analyses from `analysis.py` (e.g., gradient basis, Jacobian norms, static/dynamic decomposition) at any checkpoint during training.
* **Logging**:
    * Logs experiment results to `wandb` (Weights & Biases).

## Setup

### a. Install Dependencies

    uv pip install torch torchvision numpy pandas matplotlib pyyaml wandb scikit-learn

### b. Dataset Preparation

#### ColoredMNIST

Automatically downloaded to the `./data` directory by `data_loader.py`.

#### WaterBirds

> [!IMPORTANT]
> The WaterBirds dataset requires manual download from Kaggle (it is not available via the `wilds` library).
>
> 1.  Go to the [Kaggle Waterbird dataset page](https.www.kaggle.com/datasets/bahardibaie/waterbird?resource=download).
> 2.  Download `archive.zip`.
> 3.  Create a `data/waterbirds_v1.0/` directory in the project root and place `archive.zip` there.
>     * Final path should be: `data/waterbirds_v1.0/archive.zip`
> 4.  On the first run, `data_loader.py` will automatically extract the zip file.

### c. DINOv2 Models

> [!NOTE]
> When using `DINOv2` as a feature extractor, the model weights will be downloaded via `torch.hub` on the first run. An internet connection is required.

## Running Experiments

All experiment settings are managed in the `config.yaml` file.

1.  **Edit Configuration**: Open `config.yaml` and set the desired dataset, model, training method, analysis flags, etc.
2.  **Run the Main Script**:

        python main.py

3.  **Check Results**:
    * Progress will be printed to the console.
    * Detailed metrics can be viewed on the `wandb` dashboard.
    * Final plots will be saved in the `results/<experiment_name>_<timestamp>/` directory generated after the run.

## `config.yaml` Settings

The `config.yaml` file contains the primary parameters for controlling experiments.

<details>
<summary>Click to see all configuration options</summary>

### Basic Settings

* `experiment_name`: Name of the experiment. Used for `results` directory and `wandb`.
* `dataset_name`: Specify `ColoredMNIST` or `WaterBirds`.
* `loss_function`: `logistic` or `mse`.
* `device`: `cuda` or `cpu`.

### Feature Extractor (`feature_extractor.py`)

* `use_feature_extractor`: If `true`, extracts features using the model specified in `feature_extractor_model_name`. If `false`, uses raw pixel data as input.
* `feature_extractor_model_name`: Specify `ResNet18`, `DINOv2_ViT_S_14`, etc.
    > [!TIP]
    > Recommendation for theoretical analysis: `DINOv2_ViT_G_14`
* **ViT/DINOv2**:
    * `feature_extractor_vit_target_block`: Index of the block to extract from (-1 for the last block).
    * `feature_extractor_vit_aggregation_mode`: Select from `cls_token`, `mean_pool_patch`, `mean_pool_all`.
* **ResNet**:
    * `feature_extractor_resnet_intermediate_layer`: Select from `avgpool`, `layer3`, `layer4`.

### Model (`model.py`)

* `num_hidden_layers`, `hidden_dim`: Structure of the MLP trained on the features.
* `initialization_method`: Specify `muP` or `NTP`.

### Training (`trainer.py`)

* `epochs`: Total number of epochs.
* `debias_method`: Select from `None` (ERM), `IW_uniform` (v\_inv gradient flow), `GroupDRO`.
* `dro_eta_q`: Step size (learning rate) for updating GroupDRO group weights `q`.

### Deep Feature Reweighting (DFR)

* `use_dfr`: If `true`, DFR is executed after the main training loop.
* `dfr_target_layer`: Layer to extract features from for DFR (e.g., `"last_hidden"`, `"layer_1"`).
* `dfr_val_split_strategy`:
    * `"original"`: Uses the dataset's official validation set (recommended for WaterBirds).
    * `"split_from_train"`: Splits the training set to create a validation set (recommended for ColoredMNIST).
* `dfr_val_ratio`: Ratio of training data to use as validation when `split_from_train` is selected.
* `dfr_reg`, `dfr_c_options`: Regularization settings for the logistic regression in DFR.

### Analysis (`analysis.py`)

* `analyze_jacobian_norm`, `analyze_gradient_basis`, `analyze_gap_dynamics_factors`, `analyze_static_dynamic_decomposition`: If `true`, the corresponding detailed analysis will be executed.
* `..._analysis_epochs`: If `null` (or key omitted), runs analysis every epoch. If a list is specified (e.g., `[10, 100, 1000]`), runs analysis only at those epochs.
* Others.

</details>
