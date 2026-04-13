# GRIN+: Towards Fast Yet Effective Machine Unlearning for Imbalanced Medical Data

This repository is the official implementation of [GRIN+: Towards Fast Yet Effective Machine Unlearning for Imbalanced Medical Data]()

## Abstract

As deep learning models become fundamental to modern healthcare, the "Right to be Forgotten" mandated by privacy regulations like GDPR and HIPAA necessitates effective machine unlearning (MU) to remove sensitive patient data from trained models. However, existing MU techniques often struggle with a fundamental "privacy-efficiency-utility" (PEU) trilemma, particularly in medical scenarios where data is frequently characterized by severe class imbalance and long-tailed distributions. In such cases, standard unlearning methods can fail to protect key clinical knowledge or mistakenly delete features essential for diagnosing rare conditions due to the gradient dominance of majority classes. To address these challenges, we propose GRIN+, a novel machine unlearning framework designed for fast and precise data erasure in imbalanced medical scenarios. GRIN+ decouples unlearning-specific knowledge from generalized representations at the parameter level by analyzing the gradient contributions of both "forget" and "retain" sets. It introduces a class-adaptive influence scoring mechanism to rectify gradient dominance and employs a direction-constrained update strategy to prevent the unintended erosion of vital clinical knowledge. Comprehensive benchmarking across multiple medical datasets, including skin cancer (ISIC), brain tumor (MRI), and breast ultrasound (BUSI), demonstrates that GRIN+ achieves an optimal balance of the PEU trilemma. Experimental results show that GRIN+ maintains high diagnostic accuracy and robust privacy while significantly enhancing runtime efficiency compared to existing baselines. We have open-sourced the GRIN+ code to support further research.

## Requirements
We recommend using conda to install the requirements.

```setup
conda env create -f environment/medu.yaml
conda activate medu
pip install -e .
```

We gathered the different steps into the `pipeline/` directory.

## 1. Preparing the splits
To prepare the dataset splits run the following command:

```splits
python pipeline/step_1_generate_fixed_splits.py dataset=isic,mri,busi --multirun
```
## 2. Generate the initial models
To generate the untrained models, use the following commands:

For the ResNet18 models:
```resnet18_initial
 python pipeline/step_2_generate_model_initialization.py model=resnet18 num_classes=3,4,9 model_seed="range(10)" --multirun
```
## 3. Link the initial models to their associated datasets folders
To avoid creating initial models for each dataset we link the different initial models to the datasets:
```link
bash pipeline/step_3_create_and_link_model_initializations_dir.sh 
```
## 4. Generate Original and Naive Model instructions

```link
python pipeline/step_4_generate_original_and_naive_model_specs.py --specs_yaml pipeline/datasets_original_and_naive_hyper.yaml --output_path commands/train_original_and_naive_instructions.txt
```
This generates `./commands/train_original_and_naive_instructions.txt`

## 5. Run the Original and Naive training phase
Each line of `./commands/train_original_and_naive.txt` is a command that can be invoked as is.

For instance, one can run `python pipeline/step_5_unlearn.py unlearner=original unlearner.cfg.num_epochs=50 unlearner.cfg.batch_size=256 unlearner.cfg.optimizer.learning_rate=0.1 model=resnet18 model_seed=0 dataset=isic`
Executing all these lines will train the original and naive models for all 3 datasets.
These models then serve as starting point (original) and reference (retrained) for the next steps.

## 6. Linking the original and naive models
Then we link the original and retrained models so that they can be used in the next steps.

```
python pipeline/step_6_link_original_and_naive.py
```

## 7. Hyperparameter search
Once we have the original and retrained models, we can proceed to the hyperparameter search.
The original models serve as starting point to the unlearning method.
While the retrained models are evaluate the performance of the models.
```
python pipeline/step_7_generate_optuna.py
```
This generates `commands/all_optuna.txt`

## 8. Run the different searches
Each line of `./commands/all_optuna.txt` is a command that can be invoked as is.
Similarly to step 5, each line of the file can be called separately.

## 9. Extract the best hyperparameter per unlearning method
Once the search are complete, one can run the following:

```
pipeline/step_8_generate_all_best_hp.py
```
This generates `commands/all_best_hp.txt`, which follows a similar format to step 5 and 7.

## 10. Unlearn using the best hyperparameter
Calling the different lines of `commands/all_best_hp.txt`, run the unlearning methods with the best set of hyperparameter found.
Once these models are unlearned they are ready for evaluation.



<!-- ##  Citation
If you found our work useful please consider citing it:

```bibtex

``` -->

## Acknowledgments
We would like to express our gratitude to all references in our paper that open-sourced their codebase, methodology, and dataset, which served as the foundation for our work.