import torch
import utils
import glob
import os



def fisher_separation_score(acts: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute a simple Fisher-style between/within variance ratio for binary labels.
    Works for any two distinct label values.
    """
    unique_vals, counts = torch.unique(labels, return_counts=True)

    # Require exactly two classes
    if unique_vals.numel() != 2:
        print("Warning: labels are not binary:", unique_vals, counts)
        return 0.0

    # Take the smaller value as "false", larger as "true" (arbitrary but consistent)
    val_false = unique_vals[0]
    val_true = unique_vals[1]


    mask_true = labels == val_true
    mask_false = labels == val_false

    # Split the data by true and false
    A_true = acts[mask_true]
    A_false = acts[mask_false]

    # Need at least 2 samples per class to get stable variances
    if A_true.size(0) < 2 or A_false.size(0) < 2:
        print("Insufficient samples: true", A_true.size(0).item(),
              "false", A_false.size(0).item())
        return 0.0

    d = acts.size(1)

    mu_t = A_true.mean(dim=0)
    mu_f = A_false.mean(dim=0)

    # Between-class variance (averaged over dims)
    between = ((mu_t - mu_f) ** 2).sum() / d

    # Within-class variance (averaged over dims)
    var_t = ((A_true - mu_t) ** 2).sum(dim=1).mean() / d
    var_f = ((A_false - mu_f) ** 2).sum(dim=1).mean() / d

    within = var_t + var_f + 1e-8

    return (between / within).item()


def calibrate_best_layer(
    model_family: str,
    model_size: str,
    model_type: str,
    layers: list[int],
    datasets: list[str] = None,
    samples_per_dataset: int = 200,
):
    if datasets is None:
        datasets = ["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans"]

    train_set_sizes = {ds: samples_per_dataset for ds in datasets}
    scores = {}

    for layer in layers:
        acts_c, acts_raw, labels, polarities = utils.collect_training_data(
            datasets,
            train_set_sizes,
            model_family,
            model_size,
            model_type,
            layer,
        )
        print("Layer", layer, "label stats:","unique =", torch.unique(labels, return_counts=True))
        score = fisher_separation_score(acts_c, labels)
        scores[layer] = score
        print(f"Layer {layer:3d}: separation score = {score:.6e}")

    best_layer = max(scores, key=scores.get)
    print(f"\nBest layer for {model_family}-{model_size}-{model_type}: {best_layer}")
    return best_layer, scores


def infer_available_layers(model_family, model_size, model_type, dataset="cities",root_path="acts/"):
    path = f"{root_path}{model_family}/{model_size}/{model_type}/{dataset}"
    files = glob.glob(os.path.join(path, "layer_*_*.pt")) # Use file names to determine the number of layers
    layers = sorted({int(os.path.basename(f).split("_")[1]) for f in files})
    print(f"Found layers on disk for {model_family}-{model_size}-{model_type}: {layers[0]}..{layers[-1]}")
    return layers