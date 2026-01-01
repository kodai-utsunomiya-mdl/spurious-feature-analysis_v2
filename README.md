# Analysis of Gradient Flow and Group Performance Gaps under Spurious Correlations

## Main Features

* **Datasets**: Supports `ColoredMNIST`, `WaterBirds`, and `Dominoes`.
* **Feature Extraction**:
    * Enables training an MLP on either raw pixel data or features extracted from pre-trained models.
    * Supports various feature extractors including `ResNet18/50`, `ViT_B_16`, and `DINOv2` variants (`S`, `B`, `L`, `G`).
    * Allows feature extraction from any intermediate block of `ResNet` or `ViT`/`DINOv2`.
* **Training Methods**:
    * Standard ERM (`debias_method: "None"`)
    * Importance Weighting (for uniform target distribution, corresponding to `v_inv` gradient flow) (`debias_method: "IW_uniform"`)
    * Group Distributionally Robust Optimization (GroupDRO) (`debias_method: "GroupDRO"`)
* **Deep Feature Reweighting (DFR)**:
    * Supports evaluating the quality of learned features by re-training the last layer (Logistic Regression) on a balanced validation set.
    * Automatically generates a balanced validation set based on the dataset type:
        * **ColoredMNIST / Dominoes**: Generated from the remaining source data candidates not used for the training set.
        * **WaterBirds**: Sub-sampled from the dataset's official validation set to ensure class balance.
* **In-Depth Analysis**:
    * Executes detailed analyses from `analysis.py` at checkpoints:
        * Gradient basis & Jacobian norms
        * Static/Dynamic component decomposition
        * UMAP representation & Singular Value Decomposition (SVD)
        * Model output expectation/variance
* **Logging**:
    * Logs experiment results to `wandb` (Weights & Biases).

## Setup

### a. Install Dependencies

    uv pip install "numba>=0.59.0" torch torchvision numpy pandas matplotlib pyyaml wandb scikit-learn umap-learn cuml-cu12 --extra-index-url https://pypi.nvidia.com

### b. Dataset Preparation

#### ColoredMNIST & Dominoes

Automatically downloaded to the `./data` directory via `torchvision` in `data_loader.py`.
* **ColoredMNIST**: Generated from MNIST.
* **Dominoes**: Generated from MNIST and CIFAR10.

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
* `dataset_name`: Specify `ColoredMNIST`, `WaterBirds`, or `Dominoes`.
* `loss_function`: `logistic` or `mse`.
* `device`: `cuda` or `cpu`.
* `use_grayscale`: If `true`, converts images to 1-channel grayscale (valid when not using feature extractor).

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

* `initialization_method`: Specify `muP` (Maximal Update Parametrization) or `NTP` (Standard).
* `activation_function`: `relu`, `gelu`, `tanh`, `identity`, `silu`, `softplus`, or `abs`.
* **MLP Structure**:
    * `num_hidden_layers`: Total hidden layers (used when `use_skip_connections: false`).
    * `hidden_dim`: Width of hidden layers.
* **ResNet Structure**:
    * `use_skip_connections`: Set to `true` to use Residual connections.
    * `num_residual_blocks`: Number of residual blocks (L) (used when `use_skip_connections: true`).

### Training (`trainer.py`)

* `epochs`: Total number of epochs.
* `optimizer`: `Adam` or `SGD`.
* `learning_rate`: Base learning rate.
* `momentum`: Momentum factor (for SGD).
* `debias_method`: Select from `None` (ERM), `IW_uniform` (v_inv gradient flow), `GroupDRO`.
* `dro_eta_q`: Step size for updating GroupDRO group weights.
* `fix_final_layer`: If `true`, freezes the weights of the final classification layer.

### Deep Feature Reweighting (DFR)

* `use_dfr`: If `true`, DFR is executed after the main training loop.
* `dfr_val_samples_per_group`: Number of samples per group (g, y) to use for the balanced DFR validation set (default: 100).
* `dfr_reg`: Regularization strength (inverse of C) for DFR logistic regression.

### Analysis (`analysis.py`)

* **Analysis Flags**:
    * `analyze_jacobian_norm`
    * `analyze_gradient_basis`
    * `analyze_gap_dynamics_factors`
    * `analyze_static_dynamic_decomposition`
    * `analyze_model_output_expectation`
    * `analyze_umap_representation`
    * `analyze_singular_values`
* `..._analysis_epochs`: If `null` (or key omitted), runs analysis every epoch. If a list is specified (e.g., `[0, 10, 100]`), runs analysis only at those epochs.

</details>
