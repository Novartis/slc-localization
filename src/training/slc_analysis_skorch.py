import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os
from skorch import NeuralNetClassifier
from sklearn.preprocessing import LabelEncoder
from skorch.callbacks import EarlyStopping
import warnings
from src.models.models import MLP
from src.data.utilities import set_seed
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import logging

warnings.filterwarnings("ignore")

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def get_data(data_dir=None):
    """
    Load and merge metadata, file list, and annotation data using robust relative paths.
    Args:
        data_dir (str, optional): Base data directory. If None, uses default relative to this script.
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
        os.path.join(data_dir, "44320_2025_108_moesm6_esm.csv")
    )
    df_anno = df_anno[df_anno["annotation source"] == "RESOLUTE"]
    df_anno = df_anno.pivot(
        index=["gene symbol", "RESOLUTE cellline identifier"],
        columns="annotated subcellular location",
        values="annotation score",
    ).reset_index()
    logger.info(f"Annotation dataframe head:\n{df_anno.head()}")
    logger.info(f"Annotation dataframe shape: {df_anno.shape}")
    df_metadata['filename'] = df_metadata["Files"].apply(
        lambda x: os.path.basename(x)
    )
    logger.info(f"Metadata dataframe head:\n{df_metadata.head()}")
    logger.info(f"Metadata dataframe shape: {df_metadata.shape}")
    df_metadata = df_metadata[
        df_metadata["filename"].isin(df_filelist["image_name"])
    ].reset_index(drop=True)
    logger.info(f"Filtered metadata dataframe head:\n{df_metadata.head()}")
    logger.info(f"Filtered metadata dataframe shape: {df_metadata.shape}")
    df_metadata = df_metadata.merge(
        df_anno,
        left_on=["SLC [HGNC Symbol]", "CellLineId [RESOLUTE ID]"],
        right_on=["gene symbol", "RESOLUTE cellline identifier"],
        how="inner",
    )
    logger.info(f"Merged metadata dataframe:\n{df_metadata}")
    logger.info(f"Merged metadata dataframe shape: {df_metadata.shape}")
    # Replace NaN with 0
    df_metadata = df_metadata.fillna(0)

    return df_metadata


def load_embeddings_and_labels(
    embeddings_path: str, labels_func
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load embeddings and label dataframes, aligning indices as needed.

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
    X: pd.DataFrame, df_labels: pd.DataFrame, compartment: str, targetgene: str
) -> dict:
    """
    Train and evaluate a model for a single compartment and target gene.

    Args:
        X (pd.DataFrame): Embeddings/features DataFrame.
        df_labels (pd.DataFrame): Labels DataFrame.
        compartment (str): Compartment/label column to predict.
        targetgene (str): Target gene identifier.

    Returns:
        dict: Dictionary with mean, median, std, predictions, and true labels.
    """

    set_seed(42)  # Set random seed for reproducibility
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
    X_train = X_to_train
    y_train = y_to_train.astype("int")
    l_enc = LabelEncoder()
    y_train = np.ravel(y_train)
    y_train = l_enc.fit_transform(y_train)
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_train = np.where(y_train == 0, 0, 1)  # Convert to binary classification
    model_mlp = MLP(n_input=X_train.shape[1])
    early_stopping = EarlyStopping(
        monitor="valid_loss",
        patience=10,
        threshold=0.0001,
        threshold_mode="rel",
        lower_is_better=True,
    )
    model_target = NeuralNetClassifier(
        model_mlp,
        max_epochs=30,
        iterator_train__shuffle=False,
        device="cpu",
        verbose=0,
        callbacks=[early_stopping],
    )
    model_target.fit(X_train, y_train)
    preds = model_target.predict_proba(X_test.values)
    mean = np.mean(preds, axis=0)[1]
    median = np.median(preds, axis=0)[1]
    std = np.std(preds[:, 1])

    return {
        "mean": mean,
        "median": median,
        "std": std,
    }


def train_and_evaluate_compartment(
    X: pd.DataFrame,
    df_labels: pd.DataFrame,
    compartment: str,
    gene_column: str = "SLC [HGNC Symbol]",
) -> pd.DataFrame:
    """
    Train and evaluate models for all unique genes in a compartment.

    Args:
        X (pd.DataFrame): Embeddings/features DataFrame.
        df_labels (pd.DataFrame): Labels DataFrame.
        compartment (str): Compartment/label column to predict.
        gene_column (str): Column name for gene identifiers.

    Returns:
        pd.DataFrame: DataFrame with results for each gene.
    """
    results = []
    unique_genes = df_labels[gene_column].drop_duplicates()
    for gene in tqdm(unique_genes):
        res = train_and_evaluate_single(X, df_labels, compartment, gene)
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
) -> dict:
    """
    Run training and evaluation for multiple compartments, aggregating results.

    Args:
        X (pd.DataFrame): Embeddings/features DataFrame.
        df_labels (pd.DataFrame): Labels DataFrame.
        compartments (list): List of compartment/label column names to predict.
        gene_column (str): Column name for gene identifiers.
        output_dir (str, optional): Directory to save per-compartment results as CSV. If None, does not save.

    Returns:
        dict: Dictionary mapping compartment names to their results DataFrames.
    """
    all_results = {}
    for compartment in tqdm(compartments, desc="Processing compartments"):
        comp_df = train_and_evaluate_compartment(X, df_labels, compartment, gene_column)
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

    Args:
        all_results (dict): Output from run_multi_compartment_analysis.
        compartment_label_map (dict, optional): Mapping of compartment names to display names.
        output_dir (str, optional): Directory to save reports. If None, does not save.

    Returns:
        pd.DataFrame: Summary DataFrame with ROC AUC and other metrics per compartment.
    """
    summary = []
    for compartment, df in all_results.items():
        pred_probs = df["median"].values
        true_labels = np.where(
            df["gene_label"].values.astype(int) > 1, 1, 0
        )  # Assuming binary classification, convert labels to 0/1
        unique_classes = np.unique(true_labels)
        pred_labels = np.where(pred_probs > 0.5, 1, 0)
        if len(unique_classes) < 2:
            logger.warning(
                f"Skipping compartment '{compartment}' because only one class ({unique_classes[0]}) is present in true labels."
            )
            continue
        roc_auc = roc_auc_score(true_labels, pred_probs)
        class_report = classification_report(true_labels, pred_labels, output_dict=True)
        conf_mat = confusion_matrix(true_labels, pred_labels)

        summary.append(
            {
                "compartment": compartment_label_map[compartment]
                if compartment_label_map
                else compartment,
                "roc_auc": roc_auc,
                "confusion_matrix": conf_mat.tolist(),
            }
        )
        if output_dir is not None:
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
