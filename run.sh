#!/bin/sh

# TODO: Set the following variables for each experiments.
# TODO: Please see the related sub-directories in the "Experiments" directory for correct seed numbers.

PHASE=1                     # 1, 2, 3
NUM_BASE_CLASSES=20         # 20 and 60 for Mini-ImageNet and CIFAR. 50 and 100 for CUB-200
NUM_SHOTS=1                 # 1, 5 for phase 3
DATASET="Mini-ImageNet"     # Mini-ImageNet, CIFAR-100, CUB-200-2011
SEED=11

# ImageNet pre-training for the CUB experiments
# PHASE=0
# NUM_BASE_CLASSES=1000
# NUM_SHOTS=5
# DATASET="ImageNet"
# SEED=41

if [ $PHASE -eq 3 ]; then
    ROOT="${DATASET}/${NUM_BASE_CLASSES}_base_classes/${NUM_SHOTS}-shot"
else
    ROOT="${DATASET}/${NUM_BASE_CLASSES}_base_classes"
fi

TOML_FILE_NAME="phase=${PHASE},seed=${SEED}"
TOML_FILE_PATH="Experiments/${ROOT}/${TOML_FILE_NAME}.toml"
python main.py --settings_file ${TOML_FILE_PATH}
