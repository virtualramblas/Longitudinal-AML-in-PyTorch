import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

class GeneKnockoutAnalyzer:
    """
    Analyzes the impact of gene knockouts on Transformer model predictions.
    
    This implements perturbation-based analysis to identify which genes are 
    critical for model performance.
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize the gene knockout analyzer.
        
        Args:
            model: Trained PyTorchTransformerModel
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def get_baseline_predictions(self, sequences, labels):
        """
        Get baseline model performance without any knockouts.
        
        Args:
            sequences: Input sequences tensor (num_genes, timesteps, features)
            labels: True labels
            
        Returns:
            Dictionary with baseline metrics and predictions
        """
        with torch.no_grad():
            if not isinstance(sequences, torch.Tensor):
                sequences = torch.tensor(sequences, dtype=torch.float32)
            sequences = sequences.to(self.device)
            
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
            labels = labels.to(self.device)
            
            # Get predictions
            outputs = self.model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            
            # Calculate metrics
            accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
            f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
            
            baseline = {
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': predicted.cpu().numpy(),
                'true_labels': labels.cpu().numpy(),
                'output_probs': outputs.cpu().numpy()
            }
            
        return baseline
    
    def knockout_single_gene(self, sequences, gene_idx, knockout_value=0.0):
        """
        Create a knockout version of sequences by setting a gene to a constant value.
        
        Args:
            sequences: Input sequences tensor
            gene_idx: Index of gene to knockout
            knockout_value: Value to set the gene to (default: 0.0)
            
        Returns:
            Modified sequences with gene knocked out
        """
        if isinstance(sequences, torch.Tensor):
            sequences_ko = sequences.clone()
        else:
            sequences_ko = sequences.copy()
        
        # Set all values for this gene across all timepoints to knockout_value
        sequences_ko[gene_idx, :, :] = knockout_value
        
        return sequences_ko
    
    def analyze_single_gene_knockout(self, sequences, labels, gene_idx, 
                                     baseline_metrics):
        """
        Analyze the impact of knocking out a single gene.
        
        Args:
            sequences: Input sequences
            labels: True labels
            gene_idx: Index of gene to knockout
            baseline_metrics: Baseline performance metrics
            
        Returns:
            Dictionary with knockout impact metrics
        """
        # Create knockout sequences
        sequences_ko = self.knockout_single_gene(sequences, gene_idx)
        
        with torch.no_grad():
            if not isinstance(sequences_ko, torch.Tensor):
                sequences_ko = torch.tensor(sequences_ko, dtype=torch.float32)
            sequences_ko = sequences_ko.to(self.device)
            
            if not isinstance(labels, torch.Tensor):
                labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)
            else:
                labels_tensor = labels.to(self.device)
            
            # Get predictions with knockout
            outputs = self.model(sequences_ko)
            _, predicted = torch.max(outputs.data, 1)
            
            # Calculate metrics
            accuracy = accuracy_score(labels_tensor.cpu().numpy(), 
                                     predicted.cpu().numpy())
            f1 = f1_score(labels_tensor.cpu().numpy(), 
                         predicted.cpu().numpy(), average='weighted')
            
            # Calculate impact
            impact = {
                'gene_idx': gene_idx,
                'knockout_accuracy': accuracy,
                'knockout_f1': f1,
                'accuracy_drop': baseline_metrics['accuracy'] - accuracy,
                'f1_drop': baseline_metrics['f1_score'] - f1,
                'prediction_changes': np.sum(predicted.cpu().numpy() != 
                                            baseline_metrics['predictions']),
                'prediction_change_rate': np.mean(predicted.cpu().numpy() != 
                                                  baseline_metrics['predictions'])
            }
            
        return impact
    
    def analyze_all_genes_knockout(self, sequences, labels, gene_names, 
                                   verbose=True):
        """
        Analyze knockout impact for all genes.
        
        Args:
            sequences: Input sequences
            labels: True labels
            gene_names: List of gene names
            verbose: Whether to show progress bar
            
        Returns:
            DataFrame with knockout results for all genes
        """
        print("Computing baseline performance...")
        baseline = self.get_baseline_predictions(sequences, labels)
        
        print(f"Baseline Accuracy: {baseline['accuracy']:.4f}")
        print(f"Baseline F1 Score: {baseline['f1_score']:.4f}")
        
        print(f"\nAnalyzing knockout impact for {len(gene_names)} genes...")
        
        results = []
        iterator = tqdm(range(len(gene_names))) if verbose else range(len(gene_names))
        
        for gene_idx in iterator:
            impact = self.analyze_single_gene_knockout(
                sequences, labels, gene_idx, baseline
            )
            impact['gene_name'] = gene_names[gene_idx]
            results.append(impact)
        
        # Create dataframe and sort by impact
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('accuracy_drop', ascending=False)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        return results_df, baseline
    
    def analyze_gene_group_knockout(self, sequences, labels, gene_indices, 
                                    baseline_metrics):
        """
        Analyze the impact of knocking out a group of genes simultaneously.
        
        Args:
            sequences: Input sequences
            labels: True labels
            gene_indices: List of gene indices to knockout
            baseline_metrics: Baseline performance metrics
            
        Returns:
            Dictionary with knockout impact metrics
        """
        # Create knockout sequences
        if isinstance(sequences, torch.Tensor):
            sequences_ko = sequences.clone()
        else:
            sequences_ko = sequences.copy()
        
        # Knockout all specified genes
        for gene_idx in gene_indices:
            sequences_ko[gene_idx, :, :] = 0.0
        
        with torch.no_grad():
            if not isinstance(sequences_ko, torch.Tensor):
                sequences_ko = torch.tensor(sequences_ko, dtype=torch.float32)
            sequences_ko = sequences_ko.to(self.device)
            
            if not isinstance(labels, torch.Tensor):
                labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)
            else:
                labels_tensor = labels.to(self.device)
            
            # Get predictions
            outputs = self.model(sequences_ko)
            _, predicted = torch.max(outputs.data, 1)
            
            # Calculate metrics
            accuracy = accuracy_score(labels_tensor.cpu().numpy(), 
                                     predicted.cpu().numpy())
            f1 = f1_score(labels_tensor.cpu().numpy(), 
                         predicted.cpu().numpy(), average='weighted')
            
            impact = {
                'num_genes_knocked_out': len(gene_indices),
                'knockout_accuracy': accuracy,
                'knockout_f1': f1,
                'accuracy_drop': baseline_metrics['accuracy'] - accuracy,
                'f1_drop': baseline_metrics['f1_score'] - f1,
                'prediction_changes': np.sum(predicted.cpu().numpy() != 
                                            baseline_metrics['predictions']),
                'prediction_change_rate': np.mean(predicted.cpu().numpy() != 
                                                  baseline_metrics['predictions'])
            }
            
        return impact
    
    def analyze_top_k_cumulative_knockout(self, sequences, labels, gene_names, 
                                         knockout_df, k_values=[1, 5, 10, 20, 50, 100]):
        """
        Analyze cumulative impact of knocking out top-k most important genes.
        
        Args:
            sequences: Input sequences
            labels: True labels
            gene_names: List of gene names
            knockout_df: DataFrame with single-gene knockout results
            k_values: List of k values to test
            
        Returns:
            DataFrame with cumulative knockout results
        """
        print("\nAnalyzing cumulative knockout impact...")
        baseline = self.get_baseline_predictions(sequences, labels)
        
        results = []
        
        for k in k_values:
            if k > len(knockout_df):
                continue
            
            # Get top k genes by knockout impact
            top_k_genes = knockout_df.head(k)
            gene_indices = [gene_names.index(gene) for gene in top_k_genes['gene_name']]
            
            # Analyze group knockout
            impact = self.analyze_gene_group_knockout(
                sequences, labels, gene_indices, baseline
            )
            impact['k'] = k
            impact['genes'] = ', '.join(top_k_genes['gene_name'].head(10).tolist())
            
            results.append(impact)
            print(f"  Top-{k} genes knockout: Accuracy drop = {impact['accuracy_drop']:.4f}")
        
        return pd.DataFrame(results)
    
    def visualize_knockout_results(self, knockout_df, save_path=None, 
                                   patient_id=None, top_n=20):
        """
        Visualize gene knockout analysis results.
        
        Args:
            knockout_df: DataFrame with knockout results
            save_path: Path to save figure
            patient_id: Patient identifier
            top_n: Number of top genes to show
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Top genes by accuracy drop
        top_genes = knockout_df.head(top_n)
        axes[0, 0].barh(range(len(top_genes)), top_genes['accuracy_drop'], 
                        color='crimson', alpha=0.7)
        axes[0, 0].set_yticks(range(len(top_genes)))
        axes[0, 0].set_yticklabels(top_genes['gene_name'], fontsize=9)
        axes[0, 0].invert_yaxis()
        axes[0, 0].set_xlabel('Accuracy Drop', fontsize=11)
        axes[0, 0].set_title(f'Top {top_n} Genes by Knockout Impact (Accuracy)', fontsize=12)
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. Top genes by F1 drop
        top_genes_f1 = knockout_df.nlargest(top_n, 'f1_drop')
        axes[0, 1].barh(range(len(top_genes_f1)), top_genes_f1['f1_drop'], 
                        color='darkorange', alpha=0.7)
        axes[0, 1].set_yticks(range(len(top_genes_f1)))
        axes[0, 1].set_yticklabels(top_genes_f1['gene_name'], fontsize=9)
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_xlabel('F1 Score Drop', fontsize=11)
        axes[0, 1].set_title(f'Top {top_n} Genes by Knockout Impact (F1)', fontsize=12)
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. Distribution of knockout impact
        axes[1, 0].hist(knockout_df['accuracy_drop'], bins=50, 
                       color='steelblue', edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(knockout_df['accuracy_drop'].mean(), 
                          color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {knockout_df["accuracy_drop"].mean():.4f}')
        axes[1, 0].set_xlabel('Accuracy Drop', fontsize=11)
        axes[1, 0].set_ylabel('Number of Genes', fontsize=11)
        axes[1, 0].set_title('Distribution of Gene Knockout Impact', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Prediction change rate
        axes[1, 1].scatter(knockout_df['accuracy_drop'], 
                          knockout_df['prediction_change_rate'],
                          alpha=0.5, s=20, c='mediumseagreen')
        axes[1, 1].set_xlabel('Accuracy Drop', fontsize=11)
        axes[1, 1].set_ylabel('Prediction Change Rate', fontsize=11)
        axes[1, 1].set_title('Accuracy Drop vs Prediction Changes', fontsize=12)
        axes[1, 1].grid(alpha=0.3)
        
        # Add top gene labels to scatter plot
        top_5 = knockout_df.head(5)
        for _, row in top_5.iterrows():
            axes[1, 1].annotate(row['gene_name'], 
                              (row['accuracy_drop'], row['prediction_change_rate']),
                              fontsize=8, alpha=0.7)
        
        if patient_id:
            fig.suptitle(f'Gene Knockout Analysis - Patient {patient_id}', 
                        fontsize=14, y=0.995)
        else:
            fig.suptitle('Gene Knockout Analysis', fontsize=14, y=0.995)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved knockout analysis visualization to {save_path}")
        
        plt.close()
    
    def visualize_cumulative_knockout(self, cumulative_df, save_path=None, 
                                     patient_id=None):
        """
        Visualize cumulative knockout results.
        
        Args:
            cumulative_df: DataFrame with cumulative knockout results
            save_path: Path to save figure
            patient_id: Patient identifier
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Accuracy drop by number of genes knocked out
        axes[0].plot(cumulative_df['k'], cumulative_df['accuracy_drop'], 
                    marker='o', linewidth=2, markersize=8, color='crimson')
        axes[0].set_xlabel('Number of Top Genes Knocked Out', fontsize=11)
        axes[0].set_ylabel('Accuracy Drop', fontsize=11)
        axes[0].set_title('Cumulative Knockout Impact (Accuracy)', fontsize=12)
        axes[0].grid(alpha=0.3)
        
        # Add annotations for key points
        for _, row in cumulative_df.iterrows():
            axes[0].annotate(f"{row['accuracy_drop']:.3f}", 
                           (row['k'], row['accuracy_drop']),
                           textcoords="offset points", xytext=(0,10), 
                           ha='center', fontsize=9)
        
        # 2. F1 drop by number of genes knocked out
        axes[1].plot(cumulative_df['k'], cumulative_df['f1_drop'], 
                    marker='s', linewidth=2, markersize=8, color='darkorange')
        axes[1].set_xlabel('Number of Top Genes Knocked Out', fontsize=11)
        axes[1].set_ylabel('F1 Score Drop', fontsize=11)
        axes[1].set_title('Cumulative Knockout Impact (F1 Score)', fontsize=12)
        axes[1].grid(alpha=0.3)
        
        # Add annotations
        for _, row in cumulative_df.iterrows():
            axes[1].annotate(f"{row['f1_drop']:.3f}", 
                           (row['k'], row['f1_drop']),
                           textcoords="offset points", xytext=(0,10), 
                           ha='center', fontsize=9)
        
        if patient_id:
            fig.suptitle(f'Cumulative Gene Knockout Analysis - Patient {patient_id}', 
                        fontsize=14, y=1.00)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved cumulative knockout visualization to {save_path}")
        
        plt.close()
    
    def compare_knockout_vs_importance(self, knockout_df, importance_df, 
                                      save_path=None, patient_id=None):
        """
        Compare gene knockout impact with attention-based importance rankings.
        
        Args:
            knockout_df: DataFrame with knockout results
            importance_df: DataFrame with attention-based importance
            save_path: Path to save figure
            patient_id: Patient identifier
        """
        # Merge dataframes
        merged = pd.merge(
            knockout_df[['gene_name', 'accuracy_drop', 'rank']],
            importance_df[['Gene', 'Importance', 'Rank']],
            left_on='gene_name',
            right_on='Gene',
            how='inner'
        )
        
        # Calculate correlation
        from scipy.stats import spearmanr, pearsonr
        spearman_corr, spearman_p = spearmanr(merged['rank'], merged['Rank'])
        pearson_corr, pearson_p = pearsonr(merged['accuracy_drop'], merged['Importance'])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Scatter plot: Knockout impact vs Attention importance
        axes[0].scatter(merged['Importance'], merged['accuracy_drop'], 
                       alpha=0.5, s=30, c='steelblue')
        axes[0].set_xlabel('Attention-Based Importance', fontsize=11)
        axes[0].set_ylabel('Knockout Impact (Accuracy Drop)', fontsize=11)
        axes[0].set_title('Knockout Impact vs Attention Importance\n' + 
                         f'Pearson r = {pearson_corr:.3f} (p = {pearson_p:.3e})', 
                         fontsize=12)
        axes[0].grid(alpha=0.3)
        
        # Add trendline
        z = np.polyfit(merged['Importance'], merged['accuracy_drop'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(merged['Importance'].min(), merged['Importance'].max(), 100)
        axes[0].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
        
        # Label top genes
        top_knockout = merged.nlargest(5, 'accuracy_drop')
        for _, row in top_knockout.iterrows():
            axes[0].annotate(row['gene_name'], 
                           (row['Importance'], row['accuracy_drop']),
                           fontsize=8, alpha=0.7)
        
        # 2. Rank comparison
        axes[1].scatter(merged['Rank'], merged['rank'], 
                       alpha=0.5, s=30, c='coral')
        axes[1].plot([merged['Rank'].min(), merged['Rank'].max()], 
                    [merged['Rank'].min(), merged['Rank'].max()], 
                    'k--', alpha=0.5, label='Perfect Agreement')
        axes[1].set_xlabel('Attention-Based Rank', fontsize=11)
        axes[1].set_ylabel('Knockout Impact Rank', fontsize=11)
        axes[1].set_title('Ranking Comparison\n' + 
                         f'Spearman ρ = {spearman_corr:.3f} (p = {spearman_p:.3e})', 
                         fontsize=12)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # Invert axes for better visualization (rank 1 at top/left)
        axes[1].invert_xaxis()
        axes[1].invert_yaxis()
        
        if patient_id:
            fig.suptitle(f'Knockout vs Attention Importance Comparison - Patient {patient_id}', 
                        fontsize=14, y=1.00)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison visualization to {save_path}")
        
        plt.close()
        
        return {
            'spearman_correlation': spearman_corr,
            'spearman_pvalue': spearman_p,
            'pearson_correlation': pearson_corr,
            'pearson_pvalue': pearson_p
        }
    
    def generate_knockout_report(self, sequences, labels, gene_names, 
                                importance_df, output_dir, patient_id):
        """
        Generate comprehensive gene knockout analysis report.
        
        Args:
            sequences: Input sequences
            labels: True labels
            gene_names: List of gene names
            importance_df: DataFrame with attention-based importance
            output_dir: Directory to save outputs
            patient_id: Patient identifier
        """
        print(f"\nGenerating gene knockout analysis for patient {patient_id}...")
        
        # Create subdirectory
        knockout_dir = os.path.join(output_dir, f"{patient_id}_knockout_analysis")
        os.makedirs(knockout_dir, exist_ok=True)
        
        # 1. Analyze all genes
        print("Step 1/5: Analyzing individual gene knockouts...")
        knockout_df, baseline = self.analyze_all_genes_knockout(
            sequences, labels, gene_names, verbose=True
        )
        
        # Save results
        knockout_df.to_csv(
            os.path.join(knockout_dir, f"{patient_id}_knockout_results.csv"),
            index=False
        )
        
        # 2. Visualize results
        print("Step 2/5: Creating knockout visualizations...")
        self.visualize_knockout_results(
            knockout_df,
            save_path=os.path.join(knockout_dir, f"{patient_id}_knockout_impact.png"),
            patient_id=patient_id,
            top_n=20
        )
        
        # 3. Cumulative knockout analysis
        print("Step 3/5: Analyzing cumulative knockout impact...")
        cumulative_df = self.analyze_top_k_cumulative_knockout(
            sequences, labels, gene_names, knockout_df,
            k_values=[1, 5, 10, 20, 50, 100]
        )
        
        cumulative_df.to_csv(
            os.path.join(knockout_dir, f"{patient_id}_cumulative_knockout.csv"),
            index=False
        )
        
        self.visualize_cumulative_knockout(
            cumulative_df,
            save_path=os.path.join(knockout_dir, f"{patient_id}_cumulative_knockout.png"),
            patient_id=patient_id
        )
        
        # 4. Compare with attention importance
        print("Step 4/5: Comparing knockout impact with attention importance...")
        comparison_stats = self.compare_knockout_vs_importance(
            knockout_df, importance_df,
            save_path=os.path.join(knockout_dir, f"{patient_id}_knockout_vs_importance.png"),
            patient_id=patient_id
        )
        
        # 5. Generate summary report
        print("Step 5/5: Generating summary report...")
        self._generate_knockout_summary(
            knockout_df, cumulative_df, baseline, comparison_stats,
            knockout_dir, patient_id
        )
        
        print(f"✓ Knockout analysis complete! Results saved to {knockout_dir}")
        
        return knockout_df, cumulative_df, comparison_stats
    
    def _generate_knockout_summary(self, knockout_df, cumulative_df, baseline, 
                                   comparison_stats, output_dir, patient_id):
        """Generate text summary of knockout analysis."""
        summary_path = os.path.join(output_dir, f"{patient_id}_knockout_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"GENE KNOCKOUT ANALYSIS SUMMARY - Patient {patient_id}\n")
            f.write("="*70 + "\n\n")
            
            f.write("BASELINE PERFORMANCE\n")
            f.write("-"*70 + "\n")
            f.write(f"Accuracy: {baseline['accuracy']:.4f}\n")
            f.write(f"F1 Score: {baseline['f1_score']:.4f}\n\n")
            
            f.write("TOP 10 MOST CRITICAL GENES (by knockout impact)\n")
            f.write("-"*70 + "\n")
            for idx, row in knockout_df.head(10).iterrows():
                f.write(f"{row['rank']:2d}. {row['gene_name']:20s} | "
                       f"Acc Drop: {row['accuracy_drop']:.4f} | "
                       f"F1 Drop: {row['f1_drop']:.4f} | "
                       f"Pred Changes: {row['prediction_changes']}\n")
            
            f.write("\n\nKNOCKOUT IMPACT STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Mean accuracy drop:    {knockout_df['accuracy_drop'].mean():.4f}\n")
            f.write(f"Std accuracy drop:     {knockout_df['accuracy_drop'].std():.4f}\n")
            f.write(f"Max accuracy drop:     {knockout_df['accuracy_drop'].max():.4f}\n")
            f.write(f"Genes with >0.01 drop: {(knockout_df['accuracy_drop'] > 0.01).sum()}\n")
            f.write(f"Genes with >0.05 drop: {(knockout_df['accuracy_drop'] > 0.05).sum()}\n")
            
            f.write("\n\nCUMULATIVE KNOCKOUT IMPACT\n")
            f.write("-"*70 + "\n")
            for _, row in cumulative_df.iterrows():
                f.write(f"Top-{row['k']:3d} genes: "
                       f"Acc Drop = {row['accuracy_drop']:.4f}, "
                       f"F1 Drop = {row['f1_drop']:.4f}\n")
            
            f.write("\n\nCORRELATION WITH ATTENTION IMPORTANCE\n")
            f.write("-"*70 + "\n")
            f.write(f"Spearman rank correlation: {comparison_stats['spearman_correlation']:.4f} "
                   f"(p = {comparison_stats['spearman_pvalue']:.3e})\n")
            f.write(f"Pearson correlation:       {comparison_stats['pearson_correlation']:.4f} "
                   f"(p = {comparison_stats['pearson_pvalue']:.3e})\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"Saved knockout summary to {summary_path}")


def analyze_gene_knockouts(model, sequences, labels, gene_names, importance_df,
                           output_dir, patient_id, device='cpu'):
    """
    Convenience function to run gene knockout analysis.
    
    Args:
        model: Trained PyTorchTransformerModel
        sequences: Input sequences
        labels: True labels
        gene_names: List of gene names
        importance_df: DataFrame with attention-based importance
        output_dir: Output directory
        patient_id: Patient identifier
        device: Device to run on
        
    Returns:
        Analyzer instance and results
    """
    analyzer = GeneKnockoutAnalyzer(model, device=device)
    knockout_df, cumulative_df, comparison_stats = analyzer.generate_knockout_report(
        sequences, labels, gene_names, importance_df, output_dir, patient_id
    )
    
    return analyzer, knockout_df, cumulative_df, comparison_stats