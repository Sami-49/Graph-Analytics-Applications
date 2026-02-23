"""Results reporting and metrics display."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


class ResultsReporter:
    """Generate comprehensive results reports."""

    @staticmethod
    def print_model_metrics(results_df: pd.DataFrame, title: str = "Model Performance") -> None:
        """Print formatted model performance metrics."""
        print(f"\n{'='*100}")
        print(f"  {title}")
        print(f"{'='*100}\n")
        
        display_cols = [
            "model", "classifier", "accuracy", "precision", "recall", 
            "f1", "roc_auc", "f1_ci_lower", "f1_ci_upper"
        ]
        
        available_cols = [col for col in display_cols if col in results_df.columns]
        
        df_display = results_df[available_cols].copy()
        df_display = df_display.round(4)
        
        print(df_display.to_string(index=False))
        print(f"\n{'='*100}\n")

    @staticmethod
    def print_feature_metrics(features_df: pd.DataFrame) -> None:
        """Print feature statistics (mean, std, min, max)."""
        print(f"\n{'='*100}")
        print(f"  Feature Statistics Summary")
        print(f"{'='*100}\n")
        
        metrics = pd.DataFrame({
            "Feature": features_df.columns,
            "Mean": features_df.mean().values,
            "Std": features_df.std().values,
            "Min": features_df.min().values,
            "Max": features_df.max().values,
        })
        
        metrics = metrics.round(4)
        print(metrics.to_string(index=False))
        print(f"\n{'='*100}\n")

    @staticmethod
    def plot_boxplots(scores_df: pd.DataFrame, model_cols: List[str], results_dir: Path) -> Dict[str, Path]:
        """Create boxplots per model separated by label and save PNGs. Returns dict of paths."""
        results_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for col in model_cols:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x='label', y=col, data=scores_df, palette='Set2')
            plt.title(f'{col} distribution by label')
            plt.xlabel('Label')
            plt.ylabel(col)
            p = results_dir / f'box_{col}.png'
            plt.tight_layout()
            plt.savefig(p, dpi=150)
            plt.close()
            paths[col] = p
        return paths

    @staticmethod
    def plot_correlation_heatmap(corr_df: pd.DataFrame, results_dir: Path) -> Path:
        """Plot correlation heatmap and save PNG."""
        results_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 6))
        sns.heatmap(corr_df.astype(float), annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1)
        plt.title('Model correlation matrix')
        p = results_dir / 'correlation_heatmap.png'
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        return p

    @staticmethod
    def save_pdf_report(results_dir: Path, subset_name: str, scores_df: pd.DataFrame, comparison_df: pd.DataFrame, corr_df: pd.DataFrame) -> Path:
        """Generate a PDF report including summary tables and plots for a subset."""
        out_dir = results_dir / 'reports'
        out_dir.mkdir(parents=True, exist_ok=True)

        model_cols = [c for c in ['spectral_score','kcore_score','community_score','centralization_score','virality_score','hps_score'] if c in scores_df.columns]

        # Create plots
        plots_dir = out_dir / 'figures'
        box_paths = ResultsReporter.plot_boxplots(scores_df, model_cols, plots_dir)
        heatmap_path = ResultsReporter.plot_correlation_heatmap(corr_df, plots_dir)

        pdf_path = out_dir / f'comparison_report_{subset_name}.pdf'
        with PdfPages(pdf_path) as pdf:
            # First page: title + summary table
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.text(0.5, 0.95, f'Comparison Report - {subset_name}', ha='center', va='center', fontsize=16, weight='bold')

            # Insert discrimination table as text
            txt = comparison_df.round(4).to_string()
            ax.text(0.01, 0.02, txt, ha='left', va='bottom', fontsize=8, family='monospace')
            pdf.savefig(fig)
            plt.close(fig)

            # Add heatmap image
            fig = plt.figure(figsize=(8.5, 6))
            img = plt.imread(heatmap_path)
            plt.imshow(img)
            plt.axis('off')
            pdf.savefig(fig)
            plt.close(fig)

            # Add boxplots (one per page, two per page could be implemented)
            for col, p in box_paths.items():
                fig = plt.figure(figsize=(8.5, 6))
                img = plt.imread(p)
                plt.imshow(img)
                plt.axis('off')
                pdf.savefig(fig)
                plt.close(fig)

        return pdf_path

    @staticmethod
    def generate_pdf_and_plots_from_paths(results_dir: Path, subset_name: str, scores_csv: Path, comparison_csv: Path, corr_csv: Path) -> Path:
        """Helper: read CSVs and generate PDF+plots. Returns PDF path."""
        scores_df = pd.read_csv(scores_csv)
        comparison_df = pd.read_csv(comparison_csv, index_col=0)
        corr_df = pd.read_csv(corr_csv, index_col=0)
        return ResultsReporter.save_pdf_report(results_dir, subset_name, scores_df, comparison_df, corr_df)

    @staticmethod
    def print_spectral_model_stats(features_df: pd.DataFrame) -> None:
        """Print Spectral Model specific statistics."""
        print(f"\n{'-'*80}")
        print(f"  SPECTRAL ANALYSIS MODEL")
        print(f"{'-'*80}\n")
        
        spectral_features = [
            "spectral_radius", "spectral_gap", "fiedler_value", 
            "eigenvector_centrality_mean", "eigenvector_centrality_std",
            "laplacian_energy", "algebraic_connectivity"
        ]
        
        spectral_cols = [col for col in spectral_features if col in features_df.columns]
        
        if spectral_cols:
            df_spectral = features_df[spectral_cols].describe().round(4)
            print(df_spectral.to_string())
            print("")

    @staticmethod
    def print_kcore_model_stats(features_df: pd.DataFrame) -> None:
        """Print k-Core Decomposition specific statistics."""
        print(f"\n{'-'*80}")
        print(f"  K-CORE DECOMPOSITION MODEL")
        print(f"{'-'*80}\n")
        
        kcore_features = [
            "max_core_number", "avg_core_number", "std_core_number",
            "core_density", "degeneracy", "core_concentration"
        ]
        
        kcore_cols = [col for col in kcore_features if col in features_df.columns]
        
        if kcore_cols:
            df_kcore = features_df[kcore_cols].describe().round(4)
            print(df_kcore.to_string())
            print("")

    @staticmethod
    def print_community_model_stats(features_df: pd.DataFrame) -> None:
        """Print Community Detection specific statistics."""
        print(f"\n{'-'*80}")
        print(f"  COMMUNITY DETECTION MODEL (Louvain)")
        print(f"{'-'*80}\n")
        
        community_features = [
            "modularity", "n_communities", "community_entropy",
            "inter_community_edge_ratio", "avg_community_size", "max_community_size"
        ]
        
        community_cols = [col for col in community_features if col in features_df.columns]
        
        if community_cols:
            df_community = features_df[community_cols].describe().round(4)
            print(df_community.to_string())
            print("")

    @staticmethod
    def print_centralization_model_stats(features_df: pd.DataFrame) -> None:
        """Print Centralization Index specific statistics."""
        print(f"\n{'-'*80}")
        print(f"  CENTRALIZATION INDEX MODEL")
        print(f"{'-'*80}\n")
        
        centralization_features = [
            "degree_centralization", "degree_gini",
            "betweenness_centralization", "betweenness_gini",
            "pagerank_variance", "pagerank_gini"
        ]
        
        centralization_cols = [col for col in centralization_features if col in features_df.columns]
        
        if centralization_cols:
            df_centralization = features_df[centralization_cols].describe().round(4)
            print(df_centralization.to_string())
            print("")

    @staticmethod
    def print_virality_model_stats(features_df: pd.DataFrame) -> None:
        """Print Structural Virality specific statistics."""
        print(f"\n{'-'*80}")
        print(f"  STRUCTURAL VIRALITY MODEL")
        print(f"{'-'*80}\n")
        
        virality_features = [
            "wiener_index", "avg_shortest_path", "depth", "breadth",
            "structural_virality", "depth_breadth_product", "virality_span"
        ]
        
        virality_cols = [col for col in virality_features if col in features_df.columns]
        
        if virality_cols:
            df_virality = features_df[virality_cols].describe().round(4)
            print(df_virality.to_string())
            print("")

    @staticmethod
    def print_casdi_model_stats(features_df: pd.DataFrame) -> None:
        """Print CASDI Model specific statistics."""
        print(f"\n{'-'*80}")
        print(f"  CASDI CUSTOM MODEL (Combined Index)")
        print(f"{'-'*80}\n")
        
        casdi_features = [
            "casdi_score", "spectral_score", "core_score",
            "community_bridging_score", "centralization_score"
        ]
        
        casdi_cols = [col for col in casdi_features if col in features_df.columns]
        
        if casdi_cols:
            df_casdi = features_df[casdi_cols].describe().round(4)
            print(df_casdi.to_string())
            print("")

    @staticmethod
    def print_all_model_statistics(features_df: pd.DataFrame) -> None:
        """Print comprehensive statistics for all models."""
        print(f"\n{'='*100}")
        print(f"  COMPREHENSIVE MODEL STATISTICS")
        print(f"{'='*100}")
        
        ResultsReporter.print_spectral_model_stats(features_df)
        ResultsReporter.print_kcore_model_stats(features_df)
        ResultsReporter.print_community_model_stats(features_df)
        ResultsReporter.print_centralization_model_stats(features_df)
        ResultsReporter.print_virality_model_stats(features_df)
        ResultsReporter.print_casdi_model_stats(features_df)
        
        print(f"{'='*100}\n")

    @staticmethod
    def save_detailed_report(results_dir: Path, features_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
        """Save detailed report to text file."""
        report_path = results_dir / "detailed_report.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"{'='*100}\n")
            f.write(f"  FAKE NEWS DETECTION - DETAILED RESULTS REPORT\n")
            f.write(f"{'='*100}\n\n")
            
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write(f"{'-'*100}\n")
            f.write(results_df[["model", "classifier", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_string())
            f.write("\n\n")
            
            f.write("FEATURE STATISTICS BY MODEL\n")
            f.write(f"{'-'*100}\n")
            f.write("Spectral Features:\n")
            spectral_cols = [col for col in features_df.columns if "spectral" in col or "fiedler" in col or "eigenvector" in col]
            if spectral_cols:
                f.write(features_df[spectral_cols].describe().to_string())
            f.write("\n\n")
            
            f.write("k-Core Features:\n")
            kcore_cols = [col for col in features_df.columns if "core" in col and "community" not in col]
            if kcore_cols:
                f.write(features_df[kcore_cols].describe().to_string())
            f.write("\n\n")
            
            f.write("Community Detection Features:\n")
            community_cols = [col for col in features_df.columns if "community" in col or "modularity" in col]
            if community_cols:
                f.write(features_df[community_cols].describe().to_string())
            f.write("\n\n")

            f.write("Centralization Features:\n")
            centralization_cols = [col for col in features_df.columns if "centralization" in col or "gini" in col or "pagerank" in col]
            if centralization_cols:
                f.write(features_df[centralization_cols].describe().to_string())
            f.write("\n\n")

            f.write("Virality Features:\n")
            virality_cols = [col for col in features_df.columns if "virality" in col or "wiener" in col or "depth" in col or "breadth" in col]
            if virality_cols:
                f.write(features_df[virality_cols].describe().to_string())
            f.write("\n\n")

            f.write("CASDI Custom Model Features:\n")
            casdi_cols = [col for col in features_df.columns if "casdi" in col ]
            if casdi_cols:
                f.write(features_df[casdi_cols].describe().to_string())
            f.write("\n\n")
