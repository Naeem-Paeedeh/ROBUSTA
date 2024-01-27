# Datasets

## Mini-ImageNet and CUB-200-2011

Our implementation is based on the [S3C](https://github.com/JAYATEJAK/S3C), which follow the [CEC](https://github.com/icoz69/CEC-CVPR2021). You can download the datasets by following the instructions in [the CEC repository](https://github.com/icoz69/CEC-CVPR2021).

## CIFAR-100

CIFAR-100 will be downloaded automatically at the first time.

## ImageNet

Following the TOPIC, S3C, and FLOWER implementations, we pretrain our model on ImageNet for the CUB-200-2011 exeperiments.

Please download the tar files for the train (138GB) and validation (6.3GB) sets from the [ImageNet website](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).

We modified the scripts from [this script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4#file-extract_ilsvrc-sh) as follows:

```bash
mv ILSVRC2012_img_train.tar <ImageNet directory>
cd <ImageNet directory>
mkdir train
tar -xvf ILSVRC2012_img_train.tar -C train
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
```

For validation set run the following commands:
```bash
mv ILSVRC2012_img_val.tar <ImageNet directory>
cd <ImageNet directory>
mkdir val
tar -xvf ILSVRC2012_img_val.tar -C val
```

After that, copy the [imagenet_val_prep.sh](dataloader/imagenet_val_prep.sh) to the val directory and run it.

## Session files

If the require session file exists in the [data/index_list](/data/index_list) subdirectory, it will be loaded. However, if the program cannot load any session file, it will create a new session file that contains the list of the selected samples for each random seed number.
