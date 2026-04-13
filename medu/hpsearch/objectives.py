from torch.nn import Module
from torch.utils.data import DataLoader
import time

from medu.evaluation.evaluate_model import ModelEvaluationApp
from medu.evaluation import indiscernibility
from medu.evaluation.accuracy import compute_accuracy
from medu.evaluation.common import extract_predictions
from medu.evaluation.membership_inference_attack import evaluate_mia_on_model
from medu.evaluation.model_distances import models_l2_distance, models_normalized_l2_distance
from medu.evaluation.retention import compute_accuracy_retention
from medu.evaluation.run_time_efficiency import compute_run_time_efficiency
from medu.settings import DEFAULT_DEVICE


def log_and_print(message, log_file):
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def get_accuracy(model, loader, device):
    y_true, y_pred = extract_predictions(model, loader, device=device)
    accuracy = compute_accuracy(y_true, y_pred)
    return accuracy

def unlearner_optuna(
    original: Module,
    naive: Module,
    unlearned: Module,
    dataset: str,
    batch_size: int,
    random_state: int,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device=DEFAULT_DEVICE,
    log_file: str = None,
):
    """
    We use the naive model as the reference.
    We attempt to optimize for 3 things.
    1. Retain Accuracy (We should match the retain of the naive model)
    2. Validation Accuracy (We should match the validation accuracy of the naive model)
    3. Forget Accuracy (We should match the forget accuracy of the naive model)
    4. Indiscernibility
    """
    assert isinstance(original, Module), f"Original model must be a Module but got {type(original)}"
    assert isinstance(naive, Module), f"Naive model must be a Module but got {type(naive)}"
    assert isinstance(unlearned, Module), f"Unlearned model must be a Module but got {type(unlearned)}"
    assert isinstance(retain_loader, DataLoader), f"Retain loader must be a DataLoader but got {type(retain_loader)}"
    assert isinstance(forget_loader, DataLoader), f"Forget loader must be a DataLoader but got {type(forget_loader)}"
    assert isinstance(val_loader, DataLoader), f"Validation loader must be a DataLoader but got {type(val_loader)}"
    assert isinstance(test_loader, DataLoader), f"Test loader must be a DataLoader but got {type(test_loader)}"
    naive.to(device)
    unlearned.to(device)
    original.to(device)

    
    naive.eval()
    unlearned.eval()
    
    # print("********** original ***********", log_file)
    # original.eval()
    # print("*******************************", log_file)
    try:
        # Calculate evaluation indicators
        log_and_print("\n************ naive ************", log_file)
        naive_metrics = ModelEvaluationApp(
            naive, dataset, batch_size, random_state, device
        ).run_on_loaders(retain_loader, forget_loader, val_loader, test_loader, log_file)
        log_and_print("*******************************\n", log_file)
        log_and_print("********** unlearned **********", log_file)
        unlearned_metrics = ModelEvaluationApp(
            unlearned, dataset, batch_size, random_state, device
        ).run_on_loaders(retain_loader, forget_loader, val_loader, test_loader, log_file)
        log_and_print("*******************************\n", log_file)

        # Accuracy metrics
        acc_naive_retain = naive_metrics["Retain"].iloc[0]
        acc_naive_forget = naive_metrics["Forget"].iloc[0]
        acc_naive_val = naive_metrics["Val"].iloc[0]

        acc_unlearn_retain = unlearned_metrics["Retain"].iloc[0]
        acc_unlearn_forget = unlearned_metrics["Forget"].iloc[0]
        acc_unlearn_val = unlearned_metrics["Val"].iloc[0]

        
        dis = 1.0 - indiscernibility(unlearned_metrics["Val MIA"].iloc[0])

        l2_distance_naive = models_l2_distance(unlearned, naive)
        l2_distance_original = models_l2_distance(unlearned, original)
        normalized_l2_distance = models_normalized_l2_distance(unlearned, naive)

        retention_retain = compute_accuracy_retention(acc_unlearn_retain, acc_naive_retain)
        retention_val = compute_accuracy_retention(acc_unlearn_val, acc_naive_val)
        

        # Original loss calculation
        l_retain = abs(acc_unlearn_retain - acc_naive_retain)
        w_retain = 1.0 / 3
        l_val = abs(acc_unlearn_val - acc_naive_val)
        w_val = 1.0 / 3
        l_forget = abs(acc_unlearn_forget - acc_naive_forget)
        w_forget = 1.0 / 3

        l_indis = dis
        w_indis = 1

        # Record
        log_and_print(
            f"Naive Retain: {acc_naive_retain} | Unlearner Retain {acc_unlearn_retain} | Loss {w_retain * l_retain:.2f}", log_file
        )
        log_and_print(
            f"Naive Val: {acc_naive_val} | Unlearner Val {acc_unlearn_val} | Loss {w_val * l_val:.2f}", log_file
        )
        log_and_print(
            f"Naive Forget: {acc_naive_forget} | Unlearner Forget {acc_unlearn_forget} | Loss {w_forget * l_forget:.2f}", log_file
        )
        log_and_print(f"Discernibility: {dis} | Loss {w_indis * l_indis:.2f}", log_file)

        log_and_print(f"Model L2 Distance (unlearned vs naive): {l2_distance_naive:.4f}", log_file)
        log_and_print(f"Model L2 Distance (unlearned vs original): {l2_distance_original:.4f}", log_file)
        log_and_print(f"Normalized L2 Distance (unlearned vs naive): {normalized_l2_distance:.4f}", log_file)

        log_and_print(f"Retention (Retain): {retention_retain:.4f}", log_file)
        log_and_print(f"Retention (Validation): {retention_val:.4f}", log_file)


        return (
            w_retain * l_retain,
            w_forget * l_forget,
            w_val * l_val,
            w_indis * l_indis,
        )
    except Exception as e:
        log_and_print(f"Exception occurred: {e}", log_file)
        return float("inf"), float("inf"), float("inf"), float("inf")