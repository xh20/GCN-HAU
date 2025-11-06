# GCN-HAU
A comprehensive framework for human activity understanding through graph convolutional networks, featuring multiple architectures from published research.

## 📖 Overview

This repository contains implementations of several graph-based neural networks for human activity understanding, including:

- **TFGCN**: Understanding human activity with uncertainty measure for novelty in graph convolutional networks -- published in IJRR 2025
- **MMGCN**: Multi-Modal Graph Convolutional Network with Sinusoidal Encoding for Robust Human Action Segmentation -- published in IROS 2025  
- **PGCN**: Understanding spatio-temporal relations in human-object interaction using pyramid graph convolutional network -- published in IROS 2022

## 🏗️ Model Architecture

### Core Components
- `TF`: Temporal-Fusion decoder (from TFGCN)
- `SN-Res`: Spectral Normalization Residual blocks (from TFGCN)
- `TPP`: Temporal Pyramid Pooling (from PGCN)
- `FCN`: Fully Convolutional Network decoder
- `MM`: Multi Modal encoders (from MMGCN)
- `Weighted Mixup`: Data augmentation technique (from MMGCN)

### Available Model Variants

1. **AGCN-based**: `agcn`, `aagcn`
2. **STGCN-based**: `stgcn_fcn`, `stgcn_tpp`, `stgcn_tf`
3. **CTR-based**: `crt_fcn`, `crt_tpp`, `ctrgcn_tf`
4. **HAGCN-based**: `hagcn_tf`, `hagcn_tpp`, `hagcn_fcn`
5. **PGCN-based**: `pgcn_tf`, `pgcn`, `pgcn_fcn`
6. **Original implementations**: `tfgcn`, `mmgcn`

## 📁 Project Structure

GCN-HAU/  
├── models/  
│ ├── init.py  
│ ├── pgcn.py # Pyramid GCN  
│ ├── tfgcn.py # Temporal-Fusion GCN  
│ └── mmgcn.py # Multi-Modal GCN  
│ └── ...  
├── gaussian_model/  save the mean and std for SN models  
├── config/  
│ ├── bimacs/  
│ │ ├── pgcn/  
│ │ │ ├── train.yaml  
│ │ │ └── test.yaml  
│ │ ├── tfgcn/..   
│ │ └── mmgcn/..  
│ └── ikea/  
│ │ ├── pgcn/..  
│ │ ├── tfgcn/..  
│ │ └── mmgcn/..   
├── utils/  
├── graph/  
├── metrics/  
├── train.py  
├── test.py  
├── test_bimacs_SN.py # For OOD detection using the TFGCN with SN 
└── requirements.txt  

## ⚙️ Configuration

The project uses YAML configuration files for easy experimentation.

### Dataset Support

- **Bimacts**: Bimanual Actions Dataset (processed data can be found here: https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=66a14746)
- **IKEA**: IKEA Assembly Dataset

### Model Configuration

Each model has separate `train.yaml` and `test.yaml` files containing:

- Model architecture parameters
- Training hyperparameters
- Testing: 

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/xh20/GCN-HAU.git
cd GCN-HAU
pip install -r requirements.txt
```

### Training
The processed bimacs subject 1 train and test data are shared on the webdisk. For training and testing the MMGCN model, you need to download all images from the original bimacs dataset. If you need it, please contact me hao.xing@tum.de, and also cite the original paper from KIT.  
```
# Train PGCN on BIMACS dataset
python train.py --conf ./config/bimacs/pgcn/train.yaml

# Train TFGCN on IKEA dataset  
python train.py --conf ./config/ikea/tfgcn/train.yaml
```

### Weights
You can find the pretrained weights from the Weights/ folder:  
https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=66a14746

### Testing 
Please unzip the compressed test folder for testing. For the MMGCN model, please download all images from the original bimacs dataset.
```
# Test PGCN on BIMACS dataset
python test.py --conf config/bimacs/pgcn/test.yaml

# Test OOD detection with UQ-TFGCN, the model is trained on Bimacs, and test on noisy Bimacs and IKEA
python test_bimacs_SN.py --conf ./config/bimacs/tfgcn/test.yaml
```
### Results
Performance metrics and pretrained models for each architecture are available in the results/ directory. Please refer to the original papers for detailed performance comparisons.
Some results can be also found in the folder Results/: https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=66a14746

## 📝 Citation

If you use this code in your research, please cite the respective papers:
bibtex
```
@article{xing2025understanding,
  title={Understanding human activity with uncertainty measure for novelty in graph convolutional networks},
  author={Xing, Hao and Burschka, Darius},
  journal={The International Journal of Robotics Research},
  volume={44},
  number={6},
  pages={989--1005},
  year={2025},
  publisher={SAGE Publications Sage UK: London, England}
}

@article{xing2025multi,
  title={Multi-Modal Graph Convolutional Network with Sinusoidal Encoding for Robust Human Action Segmentation},
  author={Xing, Hao and Boey, Kai Zhe and Wu, Yuankai and Burschka, Darius and Cheng, Gordon},
  journal={arXiv preprint arXiv:2507.00752},
  year={2025}
}

@inproceedings{xing2022understanding,
  title={Understanding spatio-temporal relations in human-object interaction using pyramid graph convolutional network},
  author={Xing, Hao and Burschka, Darius},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={5195--5201},
  year={2022},
  organization={IEEE}
}
```
## 📄 License  
This project is released under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments
Thanks to all the contributors of the original papers  
Built upon open-source graph convolutional network frameworks  
Dataset providers: Bimanual Actions and IKEA ASM datasets  
