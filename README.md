# Reinforcement Learning for Interactive Polyp Segmentation in Colonoscopy Images

This repository contains the implementation of a reinforcement learning (RL) approach for interactive polyp segmentation in colonoscopy images. The project aims to simulate the interactive, human-like segmentation process using RL agents, contrasting with traditional fully automatic segmentation methods.

## Project Overview

Colorectal cancer is one of the leading causes of cancer-related deaths worldwide. Early detection of colorectal polyps during colonoscopy is crucial for preventing cancer development. This project explores a novel approach to polyp segmentation by employing reinforcement learning to create an interactive segmentation system that can:

1. Navigate to regions of interest
2. Expand or contract segmentation boundaries
3. Learn from feedback to improve segmentation quality

The RL approach is compared with a traditional U-Net model to evaluate its effectiveness.

## Dataset

The project uses the CVC-ClinicDB dataset, which includes:
- 612 colonoscopy images from 31 colonoscopy videos
- 330 images with pixel-level polyp annotations
- Images in TIFF format with corresponding binary masks

## Project Structure

```
.
├── code/
│   ├── data_utils.py         # Data loading and preprocessing utilities
│   ├── unet_model.py         # U-Net model implementation (baseline)
│   ├── rl_environment.py     # RL environment for interactive segmentation
│   ├── rl_agent.py           # PPO agent implementation
│   ├── train_unet.py         # Script for training the U-Net model
│   └── train_rl.py           # Script for training the RL agent
├── data/
│   └── raw/                  # Raw dataset files
├── models/                   # Saved models
├── results/                  # Evaluation results and visualizations
├── notebooks/                # Jupyter notebooks for analysis
├── main.py                   # Main script to run the project
└── README.md                 # Project documentation
```

## Requirements

- Python 3.8+
- PyTorch 1.7+
- OpenCV
- scikit-image
- matplotlib
- NumPy
- tqdm
- Stable-Baselines3
- Gym

## Installation

1. Create a conda environment:
   ```bash
   conda create -n wwb-567 python=3.8 -y
   conda activate wwb-567
   ```

2. Install the required packages:
   ```bash
   pip install torch torchvision numpy opencv-python scikit-image matplotlib seaborn gym stable-baselines3 monai
   ```

## Usage

### 1. Data Preparation

Place the CVC-ClinicDB dataset in the `data/raw/` directory. The dataset should have the following structure:
```
data/raw/
├── Original/       # Original colonoscopy images
└── Ground Truth/   # Binary mask annotations
```

### 2. Training and Evaluation

Run the main script with different modes:

```bash
# Train both U-Net and RL models and evaluate them
python main.py --mode all --data_dir data/raw --unet_epochs 30 --rl_episodes 1000

# Train only the U-Net model
python main.py --mode train_unet --unet_epochs 50

# Train only the RL agent
python main.py --mode train_rl --rl_episodes 2000

# Evaluate pre-trained models
python main.py --mode evaluate
```

## Approach Details

### 1. U-Net Baseline
- Standard U-Net architecture for medical image segmentation
- Trained using Dice loss
- Fully automatic segmentation

### 2. RL Interactive Segmentation
- Environment: Polyp image + current segmentation mask + pointer location
- Actions: Move pointer (up, down, left, right), expand region, shrink region, confirm segmentation
- Reward: Improvement in Dice coefficient between current mask and ground truth
- Agent: PPO (Proximal Policy Optimization) agent with CNN feature extractor
- Simulates human-like segmentation process with step-by-step refinement

## Results

The evaluation compares the segmentation performance of both approaches using:
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)
- Number of steps required by the RL agent
- Visualization of segmentation progression

The results demonstrate the potential of RL for interactive medical image segmentation, highlighting the trade-offs between fully automatic and interactive approaches.

## License

This project is licensed under the MIT License.

## Acknowledgments

- CVC-ClinicDB dataset providers
- PyTorch and Stable-Baselines3 communities
- Medical imaging experts who provided guidance 