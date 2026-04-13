DATASETS = ["isic", "busi", "mri"]
MODELS = ["resnet18"]
MODEL_SEEDS = [0, 1, 2]
ALL_MODEL_SEEDS = range(10)
REFERENCES = ["original", "naive"]
UNLEARNERS = [
    "kgltop2",    #MSG
    "kgltop5",    #CT
    "kgltop6",    #KDE
    "finetune",
    "successive_random_labels",
    "salun",
    "grin",
    "grinv2",
    "BiO",
    "fcu",
    "forgetMI",
    "grinplus",
]
ALL_UNLEARNERS = UNLEARNERS + REFERENCES
OBJECTIVES = ["objective10"]
