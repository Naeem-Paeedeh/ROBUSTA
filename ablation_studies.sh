#!/bin/sh

# TODO: Uncomment and run the python scripts for each experiment
# TODO: Set the following five variables for each experiments.
# TODO: Please see the related sub-directory in the "Experiments" directory for correct seed numbers.
#--------------------------------------------------
# Experiments:

# 1- Skipping the self-supervised learning phase with the same learning rates for backbone and classifier that we used in the main experiment.
# python main.py --settings_file "Experiments/ablation studies/without_SSL/phase=2,seed=1.toml"
# python main.py --settings_file "Experiments/ablation studies/without_SSL/phase=3,seed=1.toml"
# python main.py --settings_file "Experiments/ablation studies/without_SSL/phase=3,seed=2.toml"
# python main.py --settings_file "Experiments/ablation studies/without_SSL/phase=3,seed=3.toml"
# python main.py --settings_file "Experiments/ablation studies/without_SSL/phase=3,seed=4.toml"
# python main.py --settings_file "Experiments/ablation studies/without_SSL/phase=3,seed=5.toml"
# 1- Skipping the self-supervised learning phase with the lr=1e-3 for AdamW, which is its default value.
# python main.py --settings_file "Experiments/ablation studies/without_SSL/phase=2,seed=1-lr=1e-3.toml"
# python main.py --settings_file "Experiments/ablation studies/without_SSL/phase=2,seed=1-Adam.toml"

# 2- without_stochastic_classifier
# python main.py --settings_file "Experiments/ablation studies/without_stochastic_classifier/phase=2,seed=1.toml"
# python main.py --settings_file "Experiments/ablation studies/without_stochastic_classifier/phase=3,seed=1.toml"
# python main.py --settings_file "Experiments/ablation studies/without_stochastic_classifier/phase=3,seed=2.toml"
# python main.py --settings_file "Experiments/ablation studies/without_stochastic_classifier/phase=3,seed=3.toml"
# python main.py --settings_file "Experiments/ablation studies/without_stochastic_classifier/phase=3,seed=4.toml"
# python main.py --settings_file "Experiments/ablation studies/without_stochastic_classifier/phase=3,seed=5.toml"

# 3- without_prediction_net
# python main.py --settings_file "Experiments/ablation studies/without_prediction_net/phase=3,seed=1.toml"
# python main.py --settings_file "Experiments/ablation studies/without_prediction_net/phase=3,seed=2.toml"
# python main.py --settings_file "Experiments/ablation studies/without_prediction_net/phase=3,seed=3.toml"
# python main.py --settings_file "Experiments/ablation studies/without_prediction_net/phase=3,seed=4.toml"
# python main.py --settings_file "Experiments/ablation studies/without_prediction_net/phase=3,seed=5.toml"

# 4- without_prefixes
# Please note that the program shows the accuracies for Oracle which is not important as it does not require task-ids for the prediction.
python main.py --settings_file "Experiments/ablation studies/without_prefixes/phase=3,seed=1.toml"
# python main.py --settings_file "Experiments/ablation studies/without_prefixes/phase=3,seed=2.toml"
# python main.py --settings_file "Experiments/ablation studies/without_prefixes/phase=3,seed=3.toml"
# python main.py --settings_file "Experiments/ablation studies/without_prefixes/phase=3,seed=4.toml"
# python main.py --settings_file "Experiments/ablation studies/without_prefixes/phase=3,seed=5.toml"
