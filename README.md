# [ROBOT24] MultImgSegAgrEvaDLmTTSaCVS: Multispectral Image Segmentation in Agriculture: Evaluating Deep Learning Models with Train-Test Split and Cross-Validation Strategies

This repository contains the code implementation for the paper titled "**Multispectral Image Segmentation in Agriculture: Evaluating Deep Learning Models with Train-Test Split and Cross-Validation Strategies**" which was submitted to the ROBOT 24 Conference. The goal of this project is to evaluate the impact of different deep learning models and evaluation protocols on multispectral datasets collected by a UAV over vineyards in Portugal.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Datasets](#datasets)
5. [Results](#results)
6. [References](#references)
7. [License](#license)

## Overview
This project compares segmentation models like SegNet and DeepLabV3 applied to multispectral imagery (RGB, NDVI, GNDVI) for vineyard segmentation. It evaluates train-test split and different cross-validation techniques (standard k-fold and group k-fold), focusing on their performance in real-world agricultural tasks.

The study includes:
- Evaluation of dataset split strategies (train-test, random k-fold, group k-fold).
- Comparison of single modality vs early fusion representations.
- Analysis of the models' generalization performance.

<p align="center">
  <img src="training/overall_paper.png" alt="">
</p>

## Installation
To run the project, install the required libraries using the command:


```
pip install -r requirements.txt 
````

and run the project:

```
python training/main.py 
````

## Usage
- `training/main.py`: Trains and tests the model on the vineyard dataset.
- `training/global_perfm.txt`: performance for each model, modality and fold
- `training/util.py`: Contains constants and helper functions for data preprocessing 

## Dataset

This project makes use of the dataset "DL Vineyard Segmentation Study." Please cite the original source if you use this dataset in your research or project:

Cybonic, "DL Vineyard Segmentation Study," v1.0, GitHub, 2024. Available at: https://github.com/Cybonic/DL_vineyard_segmentation_study


## Results

The paper presents quantitative results on the segmentation performance of each model using metrics such as:

- Intersection over Union (IoU)

The results indicate that the early fusion representation achieves the highest performance across the various splitting protocols, compared to the single-input representations. The results also show that the train-test and random k-fold splitting approaches report similar results. However, when employing group k-fold the performance drops consistently across both models and the modalities. This indicates that the models lack strong generalization capabilities to new data and, on the other hand, that the train-test and random k-fold splitting protocols are appropriate to evaluate model within the same distribution but
are less adequate for out-of-distribution assessment.

| Model | Modality | Cross Val: 3 Folds | Cross Val: 4 Folds | Cross Val: 5 Folds | Cross Val: 6 Folds | Split: 70%-30% | Split: 75%-25% | Split: 80%-20% | Cross Val: Group |
|----------|---------|-------------------|-------------------|-------------------|-------------------|----------------|----------------|----------------|------------------|
| SegNet | RGB | 68.33 | 72.71 | 72.64 | 72.54 | 70.47 | **75.35** | 74.11 | 36.83 |
| | NDVI | 68.91 | 69.04 | 69.54 | 69.00 | 67.24 | 62.22 | 68.02 | 38.10 |
| | GNDVI | 67.58 | 67.90 | 67.50 | 68.21 | 68.39 | 67.77 | 68.81 | 37.15 |
| | E. FUS. | **73.33** | **73.58** | **74.03** | **73.69** | **72.86** | 73.31 | **74.17** | **46.43** |
|===|===|===|===|===|===|===|===|===|===|
| DeepLabV3 | RGB | 72.37 | **73.25** | 73.23 | 73.45 | 71.37 | 69.78 | 70.92 | 27.78 |
| | NDVI | 68.78 | 69.22 | 69.53 | 69.64 | 66.86 | 67.22 | 67.39 | 23.69 |
| | GNDVI | 67.27 | 67.63 | 68.01 | 68.32 | 66.78 | 67.74 | 66.90 | 12.66 |
| | E. FUS. | 72.72 | 72.88 | 73.54 | 73.65 | **72.76** | **72.23** | 71.17 | **32.23** |

Each column corresponds to an image and its mask obtained from the models and modalities described in the figure
<p align="center">
  <img src="training/segm_imgs.png" alt="">
</p>

## References

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{Cardoso2024,
  title={Multispectral Image Segmentation in Agriculture: Evaluating Deep Learning Models with Train-Test Split and Cross-Validation Strategies},
  author={Wilgo Cardoso and Tiago Barros and Gil Gonçalves and Cristiano Premebida and Urbano J. Nunes},
  booktitle={Proceedings of the ROBOT 24 Conference},
  year={2024},
}
```
For questions or collaborations, contact us at wilgo.moreira@isr.uc.pt.

## License

This project is licensed under the **Creative Commons Attribution 4.0 International License**. You are free to:

- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:

- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

A copy of the full license is available at [Creative Commons License](https://creativecommons.org/licenses/by/4.0/).