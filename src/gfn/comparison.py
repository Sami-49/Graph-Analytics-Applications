"""Comparative Analysis and Metrics for 6 Graph Mining Models."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats


class GraphMiningComparison:
    """Compare 6 graph mining models with rich metrics."""

    @staticmethod
    def compute_discrimination_metrics(scores_df: pd.DataFrame) -> Dict:
        """
        Compute discrimination power of each model to distinguish real vs fake news.
        
        Metrics:
        - Mean difference between real and fake
        - Effect size (Cohen's d)
        - KS statistic (distribution difference)
        - ROC-like AUC (using scores as ranking)
        """
        
        results = {}
        models = [col for col in scores_df.columns if col not in ['event_id', 'label']]
        
        for model in models:
            real_scores = scores_df[scores_df['label'] == 0][model].dropna()
            fake_scores = scores_df[scores_df['label'] == 1][model].dropna()
            
            if len(real_scores) < 2 or len(fake_scores) < 2:
                continue
            
            # Mean difference
            mean_diff = float(real_scores.mean() - fake_scores.mean())
            
            # Cohen's d (effect size)
            pooled_std = np.sqrt(((len(real_scores)-1) * real_scores.std()**2 + 
                                  (len(fake_scores)-1) * fake_scores.std()**2) / 
                                 (len(real_scores) + len(fake_scores) - 2))
            cohens_d = float(mean_diff / (pooled_std + 1e-10))
            
            # KS statistic (Kolmogorov-Smirnov test)
            ks_stat, ks_pval = stats.ks_2samp(real_scores, fake_scores)
            ks_stat, ks_pval = float(ks_stat), float(ks_pval)
            
            # Ranking-based AUC (how well scores separate the groups)
            all_scores = pd.concat([
                pd.DataFrame({'score': real_scores, 'label': 0}),
                pd.DataFrame({'score': fake_scores, 'label': 1})
            ])
            all_scores = all_scores.sort_values('score').reset_index(drop=True)
            all_scores['rank'] = range(len(all_scores))
            
            # Compute ranking AUC
            real_ranks = all_scores[all_scores['label'] == 0]['rank'].values
            fake_ranks = all_scores[all_scores['label'] == 1]['rank'].values
            
            n_real = len(real_ranks)
            n_fake = len(fake_ranks)
            ranking_auc = float((np.sum(real_ranks) - n_real*(n_real+1)/2) / (n_real * n_fake + 1e-10))
            ranking_auc = float(np.clip(ranking_auc, 0, 1))
            
            # Mann-Whitney U test (non-parametric alternative to t-test)
            mw_stat, mw_pval = stats.mannwhitneyu(real_scores, fake_scores)
            mw_stat, mw_pval = float(mw_stat), float(mw_pval)
            
            results[model] = {
                'mean_real': float(real_scores.mean()),
                'mean_fake': float(fake_scores.mean()),
                'mean_diff': mean_diff,
                'std_real': float(real_scores.std()),
                'std_fake': float(fake_scores.std()),
                'cohens_d': cohens_d,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval,
                'ranking_auc': ranking_auc,
                'mann_whitney_u': mw_stat,
                'mann_whitney_p': mw_pval,
                'n_real': n_real,
                'n_fake': n_fake,
            }
        
        return results

    @staticmethod
    def compute_correlation_matrix(scores_df: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation between model scores."""
        models = [col for col in scores_df.columns if col not in ['event_id', 'label']]
        return scores_df[models].corr()

    @staticmethod
    def compute_distribution_metrics(scores_df: pd.DataFrame) -> Dict:
        """Compute skewness, kurtosis, and other distribution properties."""
        results = {}
        models = [col for col in scores_df.columns if col not in ['event_id', 'label']]
        
        for model in models:
            scores = scores_df[model].dropna()
            
            results[model] = {
                'mean': float(scores.mean()),
                'median': float(scores.median()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'q25': float(scores.quantile(0.25)),
                'q75': float(scores.quantile(0.75)),
                'skewness': float(stats.skew(scores)),
                'kurtosis': float(stats.kurtosis(scores)),
                'iqr': float(scores.quantile(0.75) - scores.quantile(0.25)),
            }
        
        return results

    @staticmethod
    def compute_complementarity_metrics(scores_df: pd.DataFrame, discrimination_metrics: Dict) -> Dict:
        """
        Measure how complementary models are
        (i.e., do they catch different patterns?)
        """
        models = list(discrimination_metrics.keys())
        n_models = len(models)
        
        # Pairwise agreement (correlation of rankings)
        correlations = []
        for i in range(n_models):
            for j in range(i+1, n_models):
                model1, model2 = models[i], models[j]
                if model1 in scores_df.columns and model2 in scores_df.columns:
                    corr = scores_df[model1].corr(scores_df[model2])
                    correlations.append(float(corr))
        
        avg_correlation = float(np.mean(correlations)) if correlations else 0.0
        
        # Disagreement metric (1 - avg_correlation) = complementarity
        complementarity = 1.0 - avg_correlation
        
        # Best model for each category
        best_by_auc = max(discrimination_metrics.items(), 
                         key=lambda x: x[1]['ranking_auc'])
        best_by_cohens_d = max(discrimination_metrics.items(),
                              key=lambda x: abs(x[1]['cohens_d']))
        
        return {
            'avg_model_correlation': avg_correlation,
            'complementarity_score': complementarity,
            'best_model_auc': best_by_auc[0],
            'best_model_auc_value': float(best_by_auc[1]['ranking_auc']),
            'best_model_cohens_d': best_by_cohens_d[0],
            'best_model_cohens_d_value': float(best_by_cohens_d[1]['cohens_d']),
            'n_model_pairs': len(correlations),
        }

    @staticmethod
    def generate_comparison_report(scores_df: pd.DataFrame) -> Dict:
        """Generate comprehensive comparison report."""
        
        # 1. Discrimination metrics
        discrimination = GraphMiningComparison.compute_discrimination_metrics(scores_df)
        
        # 2. Correlation matrix
        corr_matrix = GraphMiningComparison.compute_correlation_matrix(scores_df)
        
        # 3. Distribution metrics
        distribution = GraphMiningComparison.compute_distribution_metrics(scores_df)
        
        # 4. Complementarity metrics
        complementarity = GraphMiningComparison.compute_complementarity_metrics(
            scores_df, discrimination
        )
        
        return {
            'discrimination': discrimination,
            'correlation_matrix': corr_matrix,
            'distribution': distribution,
            'complementarity': complementarity,
        }

    @staticmethod
    def print_discrimination_report(discrimination: Dict) -> None:
        """Print formatted discrimination metrics."""
        print(f"\n{'='*120}")
        print(f"  DISCRIMINATION POWER: How well each model distinguishes REAL vs FAKE news")
        print(f"{'='*120}\n")
        
        df_data = []
        for model, metrics in discrimination.items():
            df_data.append({
                'Model': model,
                'Mean(Real)': f"{metrics['mean_real']:.4f}",
                'Mean(Fake)': f"{metrics['mean_fake']:.4f}",
                'Difference': f"{metrics['mean_diff']:.4f}",
                'Cohen\'s d': f"{metrics['cohens_d']:.4f}",
                'Ranking AUC': f"{metrics['ranking_auc']:.4f}",
                'KS Stat': f"{metrics['ks_statistic']:.4f}",
                'KS p-val': f"{metrics['ks_pvalue']:.2e}",
            })
        
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        print(f"\n{'='*120}\n")

    @staticmethod
    def print_distribution_report(distribution: Dict) -> None:
        """Print formatted distribution metrics."""
        print(f"\n{'='*120}")
        print(f"  DISTRIBUTION ANALYSIS: Score distribution properties per model")
        print(f"{'='*120}\n")
        
        df_data = []
        for model, metrics in distribution.items():
            df_data.append({
                'Model': model,
                'Mean': f"{metrics['mean']:.4f}",
                'Median': f"{metrics['median']:.4f}",
                'Std': f"{metrics['std']:.4f}",
                'Min': f"{metrics['min']:.4f}",
                'Max': f"{metrics['max']:.4f}",
                'Skewness': f"{metrics['skewness']:.4f}",
                'Kurtosis': f"{metrics['kurtosis']:.4f}",
            })
        
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        print(f"\n{'='*120}\n")

    @staticmethod
    def print_correlation_report(corr_matrix: pd.DataFrame) -> None:
        """Print correlation matrix."""
        print(f"\n{'='*120}")
        print(f"  MODEL CORRELATION MATRIX: Agreement between model scores")
        print(f"{'='*120}\n")
        print(corr_matrix.round(4).to_string())
        print(f"\n{'='*120}\n")

    @staticmethod
    def print_complementarity_report(complementarity: Dict) -> None:
        """Print complementarity metrics."""
        print(f"\n{'='*120}")
        print(f"  MODEL COMPLEMENTARITY: How diverse are the models?")
        print(f"{'='*120}\n")
        
        print(f"Average Model Correlation:      {complementarity['avg_model_correlation']:.4f}")
        print(f"Complementarity Score:          {complementarity['complementarity_score']:.4f}")
        print(f"Best Model (by Ranking AUC):    {complementarity['best_model_auc']} (AUC={complementarity['best_model_auc_value']:.4f})")
        print(f"Best Model (by Cohen's d):      {complementarity['best_model_cohens_d']} (d={complementarity['best_model_cohens_d_value']:.4f})")
        print(f"Number of Model Pairs Compared: {complementarity['n_model_pairs']}")
        
        print(f"\n{'='*120}\n")

    @staticmethod
    def print_full_report(report: Dict) -> None:
        """Print complete comparison report."""
        GraphMiningComparison.print_discrimination_report(report['discrimination'])
        GraphMiningComparison.print_distribution_report(report['distribution'])
        GraphMiningComparison.print_correlation_report(report['correlation_matrix'])
        GraphMiningComparison.print_complementarity_report(report['complementarity'])
