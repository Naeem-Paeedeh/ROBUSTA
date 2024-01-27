# ROBUSTA

This repository contains the implementation of the ROBUSTA method in the "Few-Shot Class Incremental Learning via Robust Transformer Approach" paper. The following explains how to prepare the datasets and run the code.

## Preparing the datasets

Please refer to the provided documentation in **[Datasets](Datasets.md)** to prepare the datasets.

### Create a conda environment

The conda environment can be installed by running the following bash command:

```bash
conda create --name ROBUSTA python=3.11.5
```

Next, the environment can be activated with:

```bash
conda activate ROBUSTA
```

After that, PyTorch can be installed in that environment with the following command:

```bash
pip3 install torch torchvision torchaudio
```

Finally, you can install all other requirements by executing the following command:

```bash
pip install -r requirements-pip.txt
```

### How to run the experiments

Please set the required variables in the [run.sh](run.sh) script and execute it.
All settings for all datasets, the number of base classes, and the number of shot combinations are in the **[Experiments](/Experiments)** directory.

For the ablation studies, you can use the [ablation_studies.sh](ablation_studies.sh) script. Just uncomment the required line to run the selected experiment.

All setting are created as TOML files. A TOML is given to the main.py as the value for the "--settings_file". When the program creates an output, it also adds the date and time to the log and snapshot files to make each experiment unique.

If the full path of a saved state of the model from the previous step is given as the input_file, it will be loaded as the beginning. However, when the program cannot load a file, it will notify and ask whether you want to continue with a randomly initialized model. If the input_file is set to empty, the program will proceed without asking any question.