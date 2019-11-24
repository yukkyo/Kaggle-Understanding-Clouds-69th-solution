# Kaggle-Understanding-Clouds-69th-solution

https://www.kaggle.com/c/understanding_cloud_organization/overview

### 0. Environment

- Python 3.6.5
- CUDA 10.1.243

### 1. Preparation

#### 1.1 install Python packages

```bash
$ pip install -r docker/requirements.txt
```

#### 1.2 Download dataset

Use kaggle API command.

```text
input
├── sample_submission.csv
├── test_images
├── train.csv
└── train_images
```


#### 1.3 Split kfolds

```bash
$ cd src
$ python data_process/s01_make_kfold_csv.py --kfold 5
```


### 2. How to train

```bash
$ cd src
$ python train.py --config configs/model063.yaml --kfold 1
$ python train.py --config configs/model063.yaml --kfold 2
$ python train.py --config configs/model063.yaml --kfold 3
$ python train.py --config configs/model063.yaml --kfold 4
$ python train.py --config configs/model063.yaml --kfold 5
$ python train.py --config configs/model064.yaml --kfold 1
$ python train.py --config configs/model064.yaml --kfold 2
$ python train.py --config configs/model064.yaml --kfold 3
$ python train.py --config configs/model064.yaml --kfold 4
$ python train.py --config configs/model064.yaml --kfold 5
```

### 3. How to predict test

```bash
$ cd src
$ python submission.py --kfolds 12345 --config configs/model063.yaml --use-best --save-predicts
$ python submission.py --kfolds 12345 --config configs/model064.yaml --use-best --save-predicts
$ python utils/make_mean_sub.py
$ ls ../output/final
sub_model063064_kfolds12345_top062_minarea16250-13750-10000-5000_bottom042_usebest_6364avg_thres.csv
```
