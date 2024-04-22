# Introduction

In this project, we'll use the recent novel histopathology foundation models, PLIP, CONCH, and UNI, in conjunction with TransMIL. Our aim is to perform slide-level classification on the Prostate Cancer Grade Assessment (PANDA) dataset, and compare these foundation models with each other. These models are innovative and powerful tools for histopathology image analysis. TransMIL is a cutting-edge method for Multi Instance Learning (MIL), particularly for histopathology slides.

# How to Run

## Step 1: Extract Patches with CLAM

```bash
# First, clone the customized CLAM repository 
git clone https://github.com/alibalapour/CLAM.git

# Go, the CLAM directory
cd CLAM

# Specify the arguments and run the below command
python create_patches_fp.py --source DIR_TO_SLIDES --save_dir 'output' --patch_size 256 --preset bwh_biopsy.csv --seg --patch --stitch
```

## Step 2: Extract Features with CLAM

### ResNet50

```bash
# We are still in the CLAM directory

python extract_features_fp.py \
	--data_h5_dir DIR_TO_H5_COORDS \
	--data_slide_dir DIR_TO_SLIDES \
	--csv_path CSV_FILE_NAME \
	--feat_dir FEATURES_DIRECTORY \
	--batch_size 512 \
	--slide_ext .tiff \
	--no_auto_skip
```

### PLIP

```bash
python extract_features_fp_PLIP.py \
	--data_h5_dir DIR_TO_H5_COORDS \
	--data_slide_dir DIR_TO_SLIDES \
	--csv_path CSV_FILE_NAME \
	--feat_dir FEATURES_DIRECTORY \
	--batch_size 512 \
	--slide_ext .tiff \
	--no_auto_skip \
	--target_patch_size 224
```

### CONCH

```bash
python extract_features_fp.py \
	--data_h5_dir DIR_TO_H5_COORDS \
	--data_slide_dir DIR_TO_SLIDES \
	--csv_path CSV_FILE_NAME \
	--feat_dir FEATURES_DIRECTORY \
	--batch_size 256 \
	--slide_ext .tiff \
	--no_auto_skip \
	--target_patch_size 224 \
	--model_name 'conch_v1'
```

### UNI

```bash
python extract_features_fp.py \
	--data_h5_dir DIR_TO_H5_COORDS \
	--data_slide_dir DIR_TO_SLIDES \
	--csv_path CSV_FILE_NAME \
	--feat_dir FEATURES_DIRECTORY \
	--batch_size 256 \
	--slide_ext .tiff \
	--no_auto_skip \
	--target_patch_size 224 \
	--model_name 'uni_v1'
```

## Step 3: Train TransMIL

```bash
cd ..

# First, clone the customized TransMIL repository 
git clone https://github.com/alibalapour/TransMIL.git

# Go, the CLAM directory
cd TransMIL

# Install requirementes
pip install torchtext==0.6
pip install pytorch-lightning==1.2.3
pip install pytorch_toolbelt
pip install -r requirements.txt
pip uninstall omegaconf --yes
```

```bash
# Train on first fold of PANDA dataset with ResNet features

python train.py \
    --stage='train' \
    --config='PANDA/TransMIL.yaml' \
    --gpus=0 \
    --fold=0 \
    --data_dir='[PATH_TO_RESNET_FEATURES]' \
    --label_dir='dataset_csv/PANDA/fold_0.csv' \
    --log_path='logs/fold_0/'
```

```bash
# Test on test set of PANDA dataset with ResNet features

python train.py \
    --stage='test' \
    --config='PANDA/TransMIL.yaml' \
    --gpus=0 \
    --fold=0 \
    --data_dir='[PATH_TO_RESNET_FEATURES]' \
    --label_dir='dataset_csv/PANDA/fold_0.csv' \
    --log_path='logs/fold_0/'
```

To train and test another encoders, just replace the ‘—data_dir’ argument in the above commands.

You can access to a notebook for training and testing on the first fold of PANDA dataset via using this [link](https://www.kaggle.com/code/alibalapour/foundation-transmil).
