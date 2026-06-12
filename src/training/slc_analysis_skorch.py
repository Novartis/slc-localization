import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import Dataset as SkorchDataset
from skorch.helper import predefined_split
import warnings
import torch
from src.models.models import MLP
from src.data.utilities import set_seed
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
import logging

warnings.filterwarnings("ignore")

# Get logger (configured in main.py)
logger = logging.getLogger(__name__)


def get_data(data_dir=None, anno_filename="44320_2025_108_moesm6_esm.csv"):
    """
    Load and merge metadata, file list, and annotation data using robust relative paths.
    Args:
        data_dir (str, optional): Base data directory. If None, uses default relative to this script.
        anno_filename (str, optional): Name of annotation CSV file. Defaults to "44320_2025_108_moesm6_esm.csv".
    Returns:
        pd.DataFrame: Merged metadata DataFrame.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../data')
    else:
        data_dir = os.path.abspath(data_dir)

    df_metadata = pd.read_csv(
        os.path.join(data_dir, "filelist_sample_HATag.tsv"),
        sep="\t",
    )
    df_filelist = pd.read_csv(
        os.path.join(data_dir, "file_list.csv")
    )
    df_anno = pd.read_csv(
        os.path.join(data_dir, anno_filename)
    )
    df_anno = df_anno[df_anno["annotation source"] == "RESOLUTE"]
    df_anno = df_anno.pivot(
        index=["gene symbol", "RESOLUTE cellline identifier"],
        columns="annotated subcellular location",
        values="annotation score",
    ).reset_index()
    logger.info(f"Annotation dataframe shape: {df_anno.shape}")
    df_metadata['filename'] = df_metadata["Files"].apply(
        lambda x: os.path.basename(x)
    )
    logger.info(f"Metadata dataframe shape: {df_metadata.shape}")
    df_metadata = df_metadata[
        df_metadata["filename"].isin(df_filelist["image_name"])
    ].reset_index(drop=True)
    logger.info(f"Filtered metadata dataframe shape: {df_metadata.shape}")
    df_metadata = df_metadata.merge(
        df_anno,
        left_on=["SLC [HGNC Symbol]", "CellLineId [RESOLUTE ID]"],
        right_on=["gene symbol", "RESOLUTE cellline identifier"],
        how="inner",
    )
    logger.info(f"Merged metadata dataframe shape: {df_metadata.shape}")
    # Replace NaN with 0
    df_metadata = df_metadata.fillna(0)

    return df_metadata


def load_embeddings_and_labels(
    embeddings_path: str, labels_func
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load embeddings and label dataframes, aligning them by image_name.

    Args:
        embeddings_path (str): Path to the embeddings CSV file.
        labels_func (callable): Function to load the labels DataFrame (e.g., get_data).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple of (embeddings DataFrame, labels DataFrame).
    """
    df_labels = labels_func()
    df = pd.read_csv(embeddings_path)
    df = df.iloc[df_labels.index]
    return df, df_labels


def train_and_evaluate_single(
    X: pd.DataFrame,
    df_labels: pd.DataFrame,
    compartment: str,
    targetgene: str,
    seed: int = 42,
) -> dict:
    """
    Train and evaluate a model for a single compartment, target gene and seed.

    Args:
        X (pd.DataFrame): Embeddings/features DataFrame.
        df_labels (pd.DataFrame): Labels DataFrame.
        compartment (str): Compartment/label column to predict.
        targetgene (str): Target gene identifier.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Dictionary with mean, median, std of predicted probabilities and the seed.
    """

    set_seed(seed)  # Set random seed for reproducibility
    # Select test and train sets
    if isinstance(X, pd.DataFrame):
        X_test = X.iloc[df_labels[(df_labels["SLC [HGNC Symbol]"] == targetgene)].index]
    else:
        X = pd.DataFrame(X)
        X_test = X.iloc[df_labels[df_labels["SLC [HGNC Symbol]"] == targetgene].index]
    X_to_train = X.drop(X_test.index, axis=0)
    y_test = df_labels[[compartment]].iloc[
        df_labels[df_labels["SLC [HGNC Symbol]"] == targetgene].index
    ]
    y_to_train = df_labels[[compartment]].drop(y_test.index, axis=0)

    # Convert the full LOO-train fold to binary-labelled arrays (row order is
    # preserved, so the gene masks computed below stay aligned).
    X_to_train_arr = np.asarray(X_to_train).astype(np.float32)
    y_to_train_arr = np.ravel(y_to_train.astype("int")).astype(np.int64)
    y_to_train_arr = np.where(y_to_train_arr > 1, 1, 0)  # binary classification

    # --- Gene-aware nested validation split for early stopping ---
    # Reserve ~10% of *gene classes* (not random images) from the LOO-train
    # fold for early-stopping validation, so no gene appears in both the
    # inner-train and inner-val partitions. The split consumes the per-call
    # seed, so different seeds yield different validation gene sets.
    train_genes = df_labels.loc[X_to_train.index, "SLC [HGNC Symbol]"]
    unique_train_genes = train_genes.unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_train_genes)
    n_val = max(1, int(0.10 * len(unique_train_genes)))
    val_genes = set(unique_train_genes[:n_val])
    inner_val_mask = train_genes.isin(val_genes).values
    inner_train_mask = ~inner_val_mask

    inner_train_genes = set(unique_train_genes[n_val:])
    logger.debug(
        "Gene-aware inner split [compartment=%s, held-out=%s, seed=%d]: "
        "inner_train_genes=%d, inner_val_genes=%d, overlap=%d",
        compartment,
        targetgene,
        seed,
        len(inner_train_genes),
        len(val_genes),
        len(inner_train_genes & val_genes),
    )

    X_inner_train = X_to_train_arr[inner_train_mask]
    y_inner_train = y_to_train_arr[inner_train_mask]
    X_inner_val = X_to_train_arr[inner_val_mask]
    y_inner_val = y_to_train_arr[inner_val_mask]

    model_mlp = MLP(n_input=X_inner_train.shape[1])
    early_stopping = EarlyStopping(
        monitor="valid_loss",
        patience=10,
        threshold=0.0001,
        threshold_mode="rel",
        lower_is_better=True,
    )
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use a predefined, gene-aware validation set for early stopping instead
    # of skorch's default internal random 20% slice (which is not gene-aware
    # and would leak gene-specific signal into model selection).
    valid_ds = SkorchDataset(
        X_inner_val.astype(np.float32),
        y_inner_val.astype(np.int64),
    )
    model_target = NeuralNetClassifier(
        model_mlp,
        max_epochs=30,
        iterator_train__shuffle=True,
        criterion=torch.nn.CrossEntropyLoss,
        device=device,
        verbose=0,
        callbacks=[early_stopping],
        train_split=predefined_split(valid_ds),
    )
    model_target.fit(X_inner_train, y_inner_train)
    # NeuralNetClassifier.predict_proba returns the module's raw forward
    # output (logits), not normalised probabilities. The MLP ends in a plain
    # Linear layer, so apply a numerically stable softmax to recover true
    # [0, 1] class probabilities. CrossEntropyLoss (set above) is the correct
    # training loss for these logits, so the softmax is well calibrated.
    logits = model_target.predict_proba(X_test.values)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    mean = np.mean(probs, axis=0)[1]
    median = np.median(probs, axis=0)[1]
    std = np.std(probs[:, 1])

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "seed": seed,
    }


def train_and_evaluate_compartment(
    X: pd.DataFrame,
    df_labels: pd.DataFrame,
    compartment: str,
    gene_column: str = "SLC [HGNC Symbol]",
    seeds: list = None,
) -> pd.DataFrame:
    """
    Train and evaluate models for all unique genes in a compartment, across seeds.

    Args:
        X (pd.DataFrame): Embeddings/features DataFrame.
        df_labels (pd.DataFrame): Labels DataFrame.
        compartment (str): Compartment/label column to predict.
        gene_column (str): Column name for gene identifiers.
        seeds (list, optional): Random seeds to train one model per seed.
            Defaults to [10, 42, 123].

    Returns:
        pd.DataFrame: Long-format DataFrame with one row per (gene, seed).
    """
    if seeds is None:
        seeds = [10, 42, 123]
    results = []
    unique_genes = df_labels[gene_column].drop_duplicates()
    for gene in tqdm(unique_genes):
        for seed in seeds:
            res = train_and_evaluate_single(X, df_labels, compartment, gene, seed)
            res[gene_column] = gene
            res["gene_label"] = df_labels[df_labels[gene_column] == gene][
                compartment
            ].values[0]
            results.append(res)
    return pd.DataFrame(results)


def run_multi_compartment_analysis(
    X: pd.DataFrame,
    df_labels: pd.DataFrame,
    compartments: list,
    gene_column: str = "SLC [HGNC Symbol]",
    output_dir: str = None,
    seeds: list = None,
) -> dict:
    """
    Run training and evaluation for multiple compartments, aggregating results.

    Args:
        X (pd.DataFrame): Embeddings/features DataFrame.
        df_labels (pd.DataFrame): Labels DataFrame.
        compartments (list): List of compartment/label column names to predict.
        gene_column (str): Column name for gene identifiers.
        output_dir (str, optional): Directory to save per-compartment results as CSV. If None, does not save.
        seeds (list, optional): Random seeds for multi-seed evaluation.
            Defaults to [10, 42, 123].

    Returns:
        dict: Dictionary mapping compartment names to their results DataFrames.
    """
    if seeds is None:
        seeds = [10, 42, 123]
    all_results = {}
    for compartment in tqdm(compartments, desc="Processing compartments"):
        comp_df = train_and_evaluate_compartment(
            X, df_labels, compartment, gene_column, seeds
        )
        all_results[compartment] = comp_df
        if output_dir is not None:
            out_path = os.path.join(output_dir, f"{compartment}_results.csv")
            comp_df.to_csv(out_path, index=False)
    return all_results


def generate_compartment_reports(
    all_results: dict, compartment_label_map: dict = None, output_dir: str = None
) -> pd.DataFrame:
    """
    Generate and print/save summary reports for each compartment.

    Metrics are computed per seed and aggregated as mean +/- std across seeds.

    Args:
        all_results (dict): Output from run_multi_compartment_analysis.
        compartment_label_map (dict, optional): Mapping of compartment names to display names.
        output_dir (str, optional): Directory to save reports. If None, does not save.

    Returns:
        pd.DataFrame: Summary DataFrame with mean/std metrics per compartment.
    """
    metric_names = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
    summary = []
    for compartment, df in all_results.items():
        # Compute each metric once per seed so we can report mean +/- std.
        per_seed_metrics = []
        for seed in sorted(df["seed"].unique()):
            df_seed = df[df["seed"] == seed]
            pred_probs = df_seed["median"].values
            true_labels = np.where(df_seed["gene_label"].values.astype(int) > 1, 1, 0)
            if len(np.unique(true_labels)) < 2:
                continue
            pred_labels = np.where(pred_probs > 0.5, 1, 0)
            per_seed_metrics.append(
                {
                    "roc_auc": roc_auc_score(true_labels, pred_probs),
                    "pr_auc": average_precision_score(true_labels, pred_probs),
                    "f1": f1_score(true_labels, pred_labels, zero_division=0),
                    "precision": precision_score(
                        true_labels, pred_labels, zero_division=0
                    ),
                    "recall": recall_score(true_labels, pred_labels, zero_division=0),
                }
            )

        if not per_seed_metrics:
            logger.warning(
                f"Skipping compartment '{compartment}' because only one class is present in true labels."
            )
            continue

        per_seed_df = pd.DataFrame(per_seed_metrics)
        row = {
            "compartment": compartment_label_map[compartment]
            if compartment_label_map
            else compartment,
            "n_seeds": len(per_seed_df),
        }
        for metric in metric_names:
            row[f"{metric}_mean"] = per_seed_df[metric].mean()
            row[f"{metric}_std"] = per_seed_df[metric].std(ddof=0)
        summary.append(row)

        if output_dir is not None:
            # Pooled classification report / confusion matrix across all seeds.
            pred_probs_all = df["median"].values
            true_all = np.where(df["gene_label"].values.astype(int) > 1, 1, 0)
            pred_all = np.where(pred_probs_all > 0.5, 1, 0)
            class_report = classification_report(
                true_all, pred_all, output_dict=True, zero_division=0
            )
            conf_mat = confusion_matrix(true_all, pred_all)
            pd.DataFrame(class_report).to_csv(
                os.path.join(output_dir, f"{compartment}_classification_report.csv")
            )
            np.savetxt(
                os.path.join(output_dir, f"{compartment}_confusion_matrix.csv"),
                conf_mat,
                delimiter=",",
                fmt="%d",
            )
    summary_df = pd.DataFrame(summary)
    if output_dir is not None:
        summary_df.to_csv(
            os.path.join(output_dir, "compartment_summary.csv"), index=False
        )
    return summary_df
