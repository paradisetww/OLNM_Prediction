# Occult Lymph Node Metastasis Prediction

This repository contains the whole process of our experiments.

## 1. Environment installation

`pip install -r requirements.txt`

## 2. Preprocessing

Put the database in the `./datasets/image_mask_old` directory, while place the data information in the `./datasets/training_validation.json`, `./datasets/test_1.json` and `./datasets/test_2.json` files.

Run the following command:

`python preprocessing.py`

We will obtain the preprocessed dataset in the `./datasets/image_mask_new` directory.

The overall structure of the dataset is as follows:

- datasets/
    - image_mask_old/
        - id_1/
            - image.nii.gz
            - mask.nii.gz
        - id_2/
            - image.nii.gz
            - mask.nii.gz
        - id_3/
            - image.nii.gz
            - mask.nii.gz
        - id_4/
            - image.nii.gz
            - mask.nii.gz
        - ...
    - image_mask_new/
        - id_1/
            - image.nii.gz
            - mask.nii.gz
        - id_2/
            - image.nii.gz
            - mask.nii.gz
        - id_3/
            - image.nii.gz
            - mask.nii.gz
        - id_4/
            - image.nii.gz
            - mask.nii.gz
        - ...
    - training_validation.json
        ```
        {
            "train": [
                {
                    "image": "id_1/image.nii.gz",
                    "mask": "id_1/mask.nii.gz",
                    "label": label_1,
                    "radiomics": [
                        feature_1_1,
                        feature_1_2,
                        feature_1_3,
                        feature_1_4,
                        feature_1_5,
                        feature_1_6,
                        feature_1_7,
                        feature_1_8,
                        feature_1_9,
                        feature_1_10
                    ]
                },
                ...
            ],
            "validation": [
                {
                    "image": "id_2/image.nii.gz",
                    "mask": "id_2/mask.nii.gz",
                    "label": label_2,
                    "radiomics": [
                        feature_2_1,
                        feature_2_2,
                        feature_2_3,
                        feature_2_4,
                        feature_2_5,
                        feature_2_6,
                        feature_2_7,
                        feature_2_8,
                        feature_2_9,
                        feature_2_10
                    ]
                },
                ...
            ]
        }
        ```
    - test_1.json
        ```
        {
            "test": [
                {
                    "image": "id_3/image.nii.gz",
                    "mask": "id_3/mask.nii.gz",
                    "label": label_3,
                    "radiomics": [
                        feature_3_1,
                        feature_3_2,
                        feature_3_3,
                        feature_3_4,
                        feature_3_5,
                        feature_3_6,
                        feature_3_7,
                        feature_3_8,
                        feature_3_9,
                        feature_3_10
                    ]
                },
                ...
            ]
        }
        ```
    - test_2.json
        ```
        {
            "test": [
                {
                    "image": "id_4/image.nii.gz",
                    "mask": "id_4/mask.nii.gz",
                    "label": label_4,
                    "radiomics": [
                        feature_4_1,
                        feature_4_2,
                        feature_4_3,
                        feature_4_4,
                        feature_4_5,
                        feature_4_6,
                        feature_4_7,
                        feature_4_8,
                        feature_4_9,
                        feature_4_10
                    ]
                },
                ...
            ]
        }
        ```

## 3. Pre-trained Models

We download the pre-trained models of [ResNet-18](https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view?usp=sharing), [ResNet-34](https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view?usp=sharing), [ResNet-50](https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view?usp=sharing) , and [Swin Transformer](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt) on large-scale medical datasets, and place `resnet_18_23dataset.pth`, `resnet_34_23dataset.pth`, `resnet_50_23dataset.pth`, and `model_swinvit.pt` files in the ` ./models` directory.

## 4. Train, Validation and Test

The model is built using PyTorch. All details have been assembled in the `./train` and `./test` directories. Please replace the `"data_dir"` parameter in the following `.sh` files with the absolute path, and then execute the following commands to train, validate, and test the models:

`sh train_test_resnet18_radiomics_cat_only.sh`

`sh train_test_resnet18_pretrain.sh`

`sh train_test_resnet18_pretrain_radiomics_cat_only.sh`

`sh train_test_resnet18_pretrain_radiomics_cat.sh`

`sh train_test_resnet18_pretrain_radiomics_add.sh`

`sh train_test_resnet18_pretrain_radiomics_lr_add.sh`

`sh train_test_resnet34_radiomics_cat_only.sh`

`sh train_test_resnet34_pretrain_radiomics_cat_only.sh`

`sh train_test_resnet50_radiomics_cat_only.sh`

`sh train_test_resnet50_pretrain_radiomics_cat_only.sh`

`sh train_test_densenet_radiomics_cat_only.sh`

`sh train_test_swintr_radiomics_cat_only.sh`

`sh train_test_swintr_pretrain_radiomics_cat_only.sh`

The results will be saved in the `./output` directory.

## 5. Citation

If you find our work to be useful for your research, please consider citing.

```
@article{tian2024predicting,
  title={Predicting occult lymph node metastasis in solid-predominantly invasive lung adenocarcinoma across multiple centers using radiomics-deep learning fusion model},
  author={Tian, Weiwei and Yan, Qinqin and Huang, Xinyu and Feng, Rui and Shan, Fei and Geng, Daoying and Zhang, Zhiyong},
  journal={Cancer Imaging},
  volume={24},
  number={1},
  pages={8},
  year={2024},
  publisher={Springer}
}
```