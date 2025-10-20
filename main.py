from src.data.create_embeddings import create_embeddings
from src.training.slc_analysis_skorch import (
    load_embeddings_and_labels,
    run_multi_compartment_analysis,
    generate_compartment_reports,
    get_data,
)
import os
import warnings
import argparse
import logging

warnings.filterwarnings("ignore")

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for running the full SLC compartment analysis workflow.
    """
    parser = argparse.ArgumentParser(
        description="Run SLC compartment analysis pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, default=None, help='Base data directory (default: ./data relative to script)')
    parser.add_argument('--anno_path', type=str, default=None, help='Path to annotation CSV file (default: 44320_2025_108_moesm6_esm.csv in data_dir)')
    parser.add_argument('--embeddings_path', type=str, default=None, help='Path to save embeddings.csv')
    parser.add_argument('--filelist_path', type=str, default=None, help='Path to save file_list.csv')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save compartment results')
    parser.add_argument('--compartments', nargs='+', default=["Plasma membrane"],
        help='List of compartments to analyze (use --list_compartments to see all options). '
             'Example: --compartments "Plasma membrane" "Golgi apparatus"')
    parser.add_argument('--list_compartments', action='store_true', help='List all possible compartments and exit')
    args = parser.parse_args()

    # Get the absolute path to the directory containing this script
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = args.data_dir or os.path.join(base_dir, "data")
    anno_path = args.anno_path or os.path.join(data_dir, "44320_2025_108_moesm6_esm.csv")
    embeddings_path = args.embeddings_path or os.path.join(data_dir, "embeddings.csv")
    filelist_path = args.filelist_path or os.path.join(data_dir, "file_list.csv")
    output_dir = args.output_dir or os.path.join(data_dir, "compartment_results")
    compartments = args.compartments

    if args.list_compartments:
        # Load annotation file to get all possible compartments
        import pandas as pd
        df_anno = pd.read_csv(anno_path)
        df_anno = df_anno[df_anno["annotation source"] == "RESOLUTE"]
        all_compartments = sorted(df_anno["annotated subcellular location"].unique())
        logger.info("Available compartments:")
        for c in all_compartments:
            logger.info(f"- {c}")
        return

    if os.path.exists(embeddings_path) and os.path.exists(filelist_path):
        logger.info(f"Embeddings already exist at {embeddings_path}, skipping creation.")
        logger.info(f"File list already exists at {filelist_path}, skipping creation.")
    else:
        logger.info("Creating embeddings...")
        embeddings, file_list = create_embeddings(data_dir)
        embeddings.to_csv(embeddings_path, index=False)
        file_list.to_csv(filelist_path, index=False)
        logger.info(f"Embeddings created and saved to {embeddings_path}")
        logger.info(f"File list saved to {filelist_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    anno_filename = os.path.basename(anno_path)
    def get_data_with_path():
        return get_data(data_dir, anno_filename)
    df, df_labels = load_embeddings_and_labels(embeddings_path, get_data_with_path)

    # Run analysis
    all_results = run_multi_compartment_analysis(
        X=df,
        df_labels=df_labels,
        compartments=compartments,
        gene_column="SLC [HGNC Symbol]",
        output_dir=output_dir,
    )

    # Generate and save reports
    summary_df = generate_compartment_reports(
        all_results=all_results, output_dir=output_dir
    )
    logger.info("\nSummary of compartment analysis:")
    logger.info(f"\n{summary_df}")


if __name__ == "__main__":
    main()
