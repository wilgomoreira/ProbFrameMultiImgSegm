# [ICIR24] ProbFrameMultiImgSegm: A Probabilistic Framework Applied to Multispectral Image Segmentation

This repository contains the code implementation for the paper titled "**Probabilistic Framework Applied to Multispectral Image Segmentation**" which was submitted to the ICIR 24 Conference. This project focuses on evaluating the impact of different deep learning models and evaluation protocols by applying probability densities to the logit values of these models for probabilistic predictions. Kernel density estimation (KDE) is employed to provide non-parametric modeling of the underlying logit-value distributions. The study lays the groundwork for future research, where combining KDE with advanced techniques could improve segmentation performance. The evaluation is conducted on multispectral datasets collected by a UAV equipped with a multispectral camera over vineyards in Portugal.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Datasets](#datasets)
5. [Results](#results)
6. [References](#references)
7. [License](#license)

## Overview

Image segmentation is critical in agricultural applications, such as differentiating crops from weeds, monitoring plant health, and optimizing resource usage. Binary segmentation tasks in agriculture generally require estimating the likelihood of each pixel belonging to one of two classes (e.g., crop vs. non-crop). Traditional approaches often rely on the sigmoid function for likelihood estimation due to its simplicity and efficiency. However, the sigmoid function may not fully capture the complexity of agricultural image data.

This work investigates the use of kernel density estimation (KDE) as a non-parametric alternative for estimating probability densities in binary image segmentation. KDE introduces greater flexibility by making fewer assumptions about the data distribution, potentially improving segmentation accuracy in challenging agricultural environments. While focusing on KDE and sigmoid-based approaches, this study lays the groundwork for future research, with the expectation that combining KDE with advanced techniques could further enhance segmentation performance, offering more robust solutions for real-world agricultural tasks.~

The equations used:

1. The estimated probability density function:

$$
\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left( \frac{x - x_i}{h} \right)
$$

where $\hat{f}(x)$ is the estimated probability density function at point $x$, $n$ is the number of data points (in this case, the sample of logits), $h$ is the bandwidth, $K$ is the kernel function (Gaussian kernel in this case), and $x_i$ are the sampled logits from the training data.

2. The kernel function, typically Gaussian:

$$
K(u) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{u^2}{2} \right)
$$

3. The formula for the bandwidth $h$ :

$$
h = \left( \frac{4\hat{\sigma}^5}{3n} \right)^{\frac{1}{5}}
$$


## Installation
To run the project, install the required libraries using the command:


```
pip install -r requirements.txt 
````

and run the project:

```
python post_training/main.py 
````

## Usage
- `post_training/main.py`: Trains and tests the model on the vineyard dataset.
- `post_training/print/res_perfomance_soft.xlsx`: performance of average precision or ECE for each model, modality and fold
- `post_training/util.py`: Contains constants and helper functions for data preprocessing 

## Dataset

The [logitsAgro_dataset](https://github.com/wilgomoreira/logitsAgro_dataset) is a collection of logits specifically designed for training and testing deep learning models in the field of precision agriculture. It includes key parameters and configurations crucial for evaluating and deploying deep learning models in this domain.

What are included:

- **Model logits**: The data used in this study consists of logits obtained from the deep model (SegNet and DeepLabV3) during training and testing mode. These logits represent the unnormalized predictions (raw scores) before applying any activation function, such as sigmoid.
- **Original images**: The images originate from aerial multispectral imagery collected from three vineyards in central Portugal: Quinta de Baixo (QTA), ESAC, and Valdoeiro (VAL). Captured at a resolution of 240x240 pixels, these images were obtained using an unmanned aerial vehicle (UAV) equipped with an X7 RGB camera and a MicaSense Altum multispectral sensor. The dataset comprises RGB and near-infrared (NIR) bands, facilitating the calculation of vegetation indices such as NDVI and GNDVI. Ground-truth annotations for vine plants are included, enabling a thorough evaluation of the models. For more details, please refer to the dataset titled "DL Vineyard Segmentation Study," available at the following link: Cybonic, "DL Vineyard Segmentation Study," v1.0, GitHub, 2024. Available at: https://github.com/Cybonic/DL_vineyard_segmentation_study.



## Results

The paper presents quantitative results on the segmentation performance of each model using metrics such as:

- Average precision score

The results obtained using Kernel Density Estimation (KDE), while not consistently outperforming the traditional sigmoid method, show a close performance, particularly in certain cases. As illustrated in table below, KDE closely aligns with the Average Precision (AP) scores across various models, modalities, and datasets. We also evaluated the Expected Calibration Error (ECE) to assess confidence calibration. Notably, KDE produced better ECE results, which bodes well for the model's reliability.

Some significant findings include SEGNET RGB QTA (Sigmoid = 74.00, KDE = 74.99), SEGNET NDVI QTA (Sigmoid = 74.72, KDE = 75.51), SEGNET EARLY FUSION QTA (Sigmoid = 74.42, KDE = 75.78), and DEEPLAB EARLY FUSION VAL (Sigmoid = 90.59, KDE = 90.70). Furthermore, KDE demonstrated superior ECE performance in additional configurations, highlighting its capacity to improve model confidence. These comparisons indicate that probabilistic interpretations not only yield competitive AP scores but also enhance model confidence through improved calibration.

| Deep model | Modality | QTA |  | ESAC |  | VAL |  |
|------------|----------|-----|-----|------|-----|-----|-----|
|  |  | Sigmoid | KDE | Sigmoid | KDE | Sigmoid | KDE |
| SEGNET | RGB | 86.50 | 86.44 | 93.12 | 93.05 | 97.13 | 96.16 |
| SEGNET | NDVI | 83.85 | 82.64 | 87.87 | 87.84 | 96.72 | 96.14 |
| SEGNET | GNDVI | 83.34 | 83.33 | 88.6 | 88.58 | 95.96 | 95.49 |
| SEGNET | EARLY-FUSION | 86.50 | 86.47 | 94.78 | 94.75 | 97.27 | 95.58 |
|-------------|-------------|-------|-------|--------|-------|-------|-------|
| DEEPLABV3 | RGB | 83.07 | 82.92 | 95.56 | 95.56 | 94.49 | 93.69 |
| DEEPLABV3 | NDVI | 82.51 | 82.50 | 89.76 | 89.71 | 93.27 | 93.20 |
| DEEPLABV3 | GNDVI | 81.43 | 81.43 | 91.33 | 91.32 | 91.52 | 91.39 |
| DEEPLABV3 | EARLY-FUSION | 84.95 | 84.53 | 94.77 | 94.71 | 93.46 | 93.06 |

## References

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{Cardoso2024,
  title={Probabilistic Framework Applied to Multispectral Image Segmentation},
  author={Wilgo Cardoso and Tiago Barros and Gil Gonçalves and Cristiano Premebida},
  booktitle={Proceedings of the ICIR 24 Conference},
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