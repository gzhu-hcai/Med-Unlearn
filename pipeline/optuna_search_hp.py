import argparse
import copy
import glob
from pathlib import Path
import re

import omegaconf

import sys
import os
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_root)
from pipeline.optuna_utils import get_loaders

import gc
import time

import optuna
import torch
from hydra.utils import instantiate
from typing import Optional

from medu.configurations import (
    dataset_store,
    get_img_size_for_dataset,
    model_store,
    unlearner_store,
)
from medu.models import get_model_from_cfg
from medu.hpsearch.objectives import unlearner_optuna
from medu.hpsearch.suggestor import HyperParameterSuggestor
from medu.evaluation.run_time_efficiency import compute_run_time_efficiency
from medu.settings import DEFAULT_BATCH_SIZE, DEFAULT_OPTUNA_N_TRIALS
from pipeline.step_5_unlearn import UnlearnerApp

OBJECTIVE_TO_FUNC = {
    "objective10": unlearner_optuna,
}


def log_and_print(message, log_file):
    """Simultaneously print to the console and write to the log file."""
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def get_time_from_meta(dataset_name, unlearner_name, model_name, model_seed, trial):
    """
    Match files from a specified path and extract time values.
    Parameters:
        dataset_name (str): Dataset name (e.g., 'isic')
        unlearner_name (str): Forgetting learner name (e.g., "finetune")
        model_name (str): Model name (e.g., 'resnet18')
        model_seed (int): Model seed (e.g., 1)
    
    Returns:
        float: Time value (e.g., 256.8305039405823)
    
    Exceptions:
        FileNotFoundError: No matching meta file found
        ValueError: No time value found in the file
    """
    
    base_path_1 = os.path.join("artifacts", dataset_name, "unlearn", "unlearner_naive")
    base_path_2 = os.path.join("artifacts", dataset_name, "artifacts", dataset_name, "unlearn", unlearner_name)
    
    pattern = f"*{model_name}_{model_seed}_meta.txt"
    
    # Use glob to find matching files (current directory only, no recursion).
    matches_1 = glob.glob(os.path.join(base_path_1, pattern))
    
    if not matches_1:
        raise FileNotFoundError(
            f"No meta file found for dataset='{dataset_name}', model='{model_name}', seed={model_seed}"
        )
    
    # Read the first matching file
    with open(matches_1[0], 'r') as f:
        content_1 = f.read()
    
    # Dynamically select a regular expression based on dataset_name
    if dataset_name in ["isic", "mri", "busi"]:
        match_1 = re.search(r"time_0\s*:\s*([\d.]+)", content_1)
    else:
        match_1 = re.search(r"time\s*:\s*([\d.]+)", content_1)
    if not match_1:
        raise ValueError(f"Time value not found in {matches_1[0]}")

    
    matches_2 = glob.glob(os.path.join(base_path_2, pattern))
    if not matches_2:
        raise FileNotFoundError(
            f"No meta file found for dataset='{dataset_name}', unlearner='{unlearner_name}', model='{model_name}', seed={model_seed}"
        )
    with open(matches_2[0], 'r') as f:
        content_2 = f.read()
    match_2 = re.search(rf"time_{trial}\s*:\s*([\d.]+)", content_2)
    if not match_2:
        raise ValueError(f"Time value not found in {matches_2[0]}")
    
    return float(match_1.group(1)), float(match_2.group(1))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--unlearner", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_seed", type=int, default=1)
    parser.add_argument("--random_state", type=int, default=123)
    parser.add_argument("--subdir", type=str, default="optuna")
    parser.add_argument("--split_ndx", type=int, default=0)
    parser.add_argument("--forget_ndx", type=int, default=0)
    parser.add_argument(
        "--objective",
        type=str,
        choices=OBJECTIVE_TO_FUNC.keys(),
    )
    parser.add_argument("--num_trials", type=int, default=5)
    return parser


def get_configuration(dataset_name, unlearner_name, model_name):
    dataset_conf = dataset_store["dataset", dataset_name]
    model_conf = model_store["model", model_name]
    unlearner_conf = unlearner_store["unlearner", unlearner_name]
    return dataset_conf, unlearner_conf, model_conf


def instantiate_objects(dataset_name: str, model_name: str, unlearner_name: str):
    dataset_conf, unlearner_conf, model_conf = get_configuration(
        dataset_name=dataset_name, model_name=model_name, unlearner_name=unlearner_name
    )

    dataset_instance = instantiate(dataset_conf) if dataset_conf else None
    model_instance = instantiate(model_conf) if model_conf else None
    unlearner_instance = instantiate(unlearner_conf) if unlearner_conf else None

    return dataset_instance, unlearner_instance, model_instance


def to_optimize(
    trial,
    args,
    save_dir,
    objective_function,
    split_ndx: Optional[int],
    forget_ndx: Optional[int],
):
    print("split_ndx", split_ndx, "forget_ndx", forget_ndx)
    save_name = None
    assert (split_ndx is not None and forget_ndx is not None) or (
        split_ndx is None and forget_ndx is None
    )

    dataset_name = args.dataset
    unlearner_name = args.unlearner
    model_name = args.model
    print("Starting", model_name)
    dataset_cfg, unlearner, model_cfg = instantiate_objects(
        dataset_name=dataset_name,
        model_name=model_name,
        unlearner_name=unlearner_name,
    )
    model_seed = args.model_seed
    random_state = args.random_state
    img_size = get_img_size_for_dataset(dataset_name)
    root = Path(".")
    naive_cfg = copy.deepcopy(unlearner.cfg)
    original_cfg = copy.deepcopy(unlearner.cfg)
    
    naive_cfg.model_initializations_dir = "unlearn/unlearner_naive"
    original_cfg.model_initializations_dir = "unlearn/unlearner_original"
    weights_root = root / "artifacts" / dataset_name
    name_save_path = root / "artifacts" / dataset_name / "unlearn"

    naive_model = get_model_from_cfg(
        root=weights_root,
        model_cfg=model_cfg,
        unlearner_cfg=naive_cfg,
        num_classes=dataset_cfg.num_classes,
        model_seed=model_seed,
        img_size=img_size,
        split_ndx=split_ndx,
        forget_ndx=forget_ndx,
    )
    model = get_model_from_cfg(
        root=weights_root,
        model_cfg=model_cfg,
        unlearner_cfg=original_cfg,
        num_classes=dataset_cfg.num_classes,
        model_seed=model_seed,
        img_size=img_size,
        split_ndx=split_ndx,
        forget_ndx=forget_ndx,
    )

    hyper_parameters = unlearner.HYPER_PARAMETERS
    suggestor = HyperParameterSuggestor(dataset_name)
    suggestor.suggest_in_place(unlearner.cfg, hyper_parameters, trial)
    
    print(f"Unlearner Configuration {unlearner}")

    app = UnlearnerApp(
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        unlearner=unlearner,
        model_seed=model_seed,
        random_state=random_state,
    )
    ###
    img_size = get_img_size_for_dataset(dataset_name)
    save_name = None
    train_loader, retain_loader, forget_loader, val_loader, test_loader = get_loaders(
        root=root,
        dataset_cfg=dataset_cfg,
        unlearner_cfg=unlearner.cfg,
        split_ndx=split_ndx,
        forget_ndx=forget_ndx,
        random_state=random_state,
    )
    original_model, unlearned_model = app.run_from_model_and_loaders(
        root=root,
        model=model,
        loaders=[
            train_loader,
            retain_loader,
            forget_loader,
            val_loader,
            test_loader,
        ],
        save=True,
        save_path=name_save_path,
        save_name=save_name,
        unlearner_name=unlearner_name,
        trial_number=trial.number,  # Transmit trial number
    )
    from medu.settings import default_loaders
    unlearner.cfg.loaders = omegaconf.DictConfig(default_loaders())
    train_loader, retain_loader, forget_loader, val_loader, test_loader = get_loaders(
        root=root,
        dataset_cfg=dataset_cfg,
        unlearner_cfg=unlearner.cfg,
        split_ndx=split_ndx,
        forget_ndx=forget_ndx,
        random_state=random_state,
    )

    torch.cuda.empty_cache()
    gc.collect()
    save_path = save_dir / f"{trial.number}.pt"
    # print(f"Saved model at path {save_path}")
    log_file = save_dir / f"log_{trial.number}.txt"
    log_and_print(f"Saved model at path {save_path}", log_file)
    time_1, time_2 = get_time_from_meta(dataset_name, unlearner_name, model_name, model_seed, trial.number)
    run_time_eff = compute_run_time_efficiency(time_2, time_1)
    log_and_print(f"Run Time Efficiency (naive/unlearned): {run_time_eff:.4f}", log_file)
    
    torch.save(unlearned_model.state_dict(), save_path)
    return objective_function(
        original_model,
        naive_model,
        unlearned_model,
        dataset_name,
        DEFAULT_BATCH_SIZE,
        random_state,
        retain_loader,
        forget_loader,
        val_loader,
        test_loader,
        log_file=log_file,
    )


def format_study_name(
    dataset_name: str,
    unlearner_name: str,
    model_name: str,
    model_seed: int,
    objective: str,
    split_ndx: Optional[int] = None,
    forget_ndx: Optional[int] = None,
) -> str:
    name = f"opt-{dataset_name}-{unlearner_name}"
    name += f"-{model_name}-{model_seed}-{objective}"
    return name


def format_optuna_save_dir(
    dataset_name: str, study_name: str, subdir: str = "optuna"
) -> Path:
    save_dir = Path("artifacts") / f"{dataset_name}" / subdir / f"{study_name}"
    return save_dir


def get_study_and_save_dir(
    dataset,
    model_seed,
    study_name,
    objective,
    subdir="optuna",
    objective_to_func=OBJECTIVE_TO_FUNC,
):
    save_dir = format_optuna_save_dir(dataset, study_name, subdir=subdir)
    save_dir.mkdir(parents=True, exist_ok=True)
    db_path = save_dir / "study.db"
    storage_url = "sqlite:///" + str(db_path)

    if objective in [
        "objective1",
        "objective2",
        "objective3",
        "objective6",
        "objective7",
        "objective8",
    ]:
        sampler = optuna.samplers.NSGAIIISampler(seed=model_seed)
        directions = ["minimize", "minimize", "minimize"]
    elif objective in ["objective4"]:
        sampler = optuna.samplers.TPESampler(seed=model_seed)
        directions = ["minimize"]
    elif objective in ["objective5", "objective9", "objective10"]:
        sampler = optuna.samplers.NSGAIIISampler(seed=model_seed)
        directions = ["minimize", "minimize", "minimize", "minimize"]
    else:
        raise NotImplementedError(f"Objective '{objective}' not implemented.")

    study = optuna.create_study(
        study_name=study_name,
        directions=directions,
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True,
    )

    return study, save_dir, objective_to_func[objective]


class OptunaSearchApp:
    def __init__(
        self,
        subdir: str = "optuna",
        split_ndx: int = 0,
        forget_ndx: int = 0,
        num_trials: int = DEFAULT_OPTUNA_N_TRIALS,
    ):
        self.subdir = subdir
        self.num_trials = num_trials
        self.split_ndx = split_ndx
        self.forget_ndx = forget_ndx

    def run(
        self, dataset: str, unlearner: str, model: str, model_seed: int, objective: str
    ):
        num_trials = self.num_trials
        subdir = self.subdir
        study_name = format_study_name(
            dataset,
            unlearner,
            model,
            model_seed,
            objective=objective,
            split_ndx=self.split_ndx,
            forget_ndx=self.forget_ndx,
        )
        study, save_dir, objective_func = get_study_and_save_dir(
            dataset=dataset,
            model_seed=model_seed,
            study_name=study_name,
            objective=objective,
            subdir=subdir,
        )
        print(f"Starting study '{study_name}' for {num_trials} trials in '{save_dir}'.")
        # NOTE: Old version considered failed trial, we need only completed trials
        # num_trials_already_run = len(study.trials)
        retries_left = 3
        previous_complete = -1
        while True:
            num_trials_already_run = len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            )
            num_trials_to_run = min(num_trials, num_trials - num_trials_already_run)

            msg = f"Study '{study_name}' has already completed "
            msg += f"{num_trials_already_run} trials. "
            msg += f"And a total of {len(study.trials)} trials. "
            msg += f"Running up to {num_trials_to_run} more trials."
            print(msg)

            if num_trials_to_run > 0 and retries_left > 0:
                study.optimize(
                    lambda trial: to_optimize(
                        trial,
                        args,
                        save_dir,
                        objective_function=objective_func,
                        split_ndx=self.split_ndx,
                        forget_ndx=self.forget_ndx,
                    ),
                    # n_trials=num_trials_to_run,
                    n_trials=1,
                )
                if len(study.trials) == previous_complete:
                    retries_left -= 1
                else:
                    retries_left = 3
            else:
                msg = "No more trials are needed; the study has "
                msg += "reached the maximum number of trials."
                print(msg)
                break


# Would greatly benefit from having the sqlite save system instead of pickled results
def main(args):
    split_ndx = args.split_ndx
    forget_ndx = args.forget_ndx
    print(f"split_ndx: {split_ndx}, forget_ndx: {forget_ndx}")

    subdir = args.subdir
    app = OptunaSearchApp(
        subdir=subdir,
        split_ndx=split_ndx,
        forget_ndx=forget_ndx,
        num_trials=args.num_trials,
    )
    app.run(
        dataset=args.dataset,
        unlearner=args.unlearner,
        model=args.model,
        model_seed=args.model_seed,
        objective=args.objective,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
