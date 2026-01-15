import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.cluster import KMeans

class AttentionAnalyzer:
    """
    Analyzes attention outputs for gene-level analysis and attention weights for 
    high-level temporal pattern understanding.
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize the attention analyzer.
        
        Args:
            model: Trained PyTorchTransformerModel
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def extract_attention_data(self, sequences):
        """
        Extract both attention outputs and weights.
        
        Args:
            sequences: Input sequences tensor (num_samples, timesteps, features)
            
        Returns:
            attention_outputs: Attention layer output tensor (for gene-level analysis)
            attention_weights: Attention weights tensor (for temporal pattern analysis)
        """
        with torch.no_grad():
            if not isinstance(sequences, torch.Tensor):
                sequences = torch.tensor(sequences, dtype=torch.float32)
            sequences = sequences.to(self.device)
            
            # Get both outputs and weights from attention layer
            attention_outputs, attention_weights = self.model.multi_head_attention(
                sequences, 
                return_attention_weights=True
            )
            
        return attention_outputs, attention_weights
    
    # ========================================================================
    # GENE-LEVEL ANALYSIS (Primary Focus - Using Attention Outputs)
    # ========================================================================
    
    def compute_gene_importance_from_outputs(self, attention_outputs, gene_names):
        """
        Compute gene importance scores from attention layer outputs.
        This replicates the approach used in the original code.
        
        Args:
            attention_outputs: Attention layer outputs (num_genes, timesteps, features)
            gene_names: List of gene names
            
        Returns:
            DataFrame with gene importance rankings
        """
        if isinstance(attention_outputs, torch.Tensor):
            outputs_np = attention_outputs.cpu().numpy()
        else:
            outputs_np = attention_outputs
        
        # Compute importance as mean absolute activation across time and features
        feature_importance = np.mean(np.abs(outputs_np), axis=(1, 2))
        
        # Create ranking dataframe
        importance_df = pd.DataFrame({
            'Gene': gene_names,
            'Importance': feature_importance,
            'Rank': range(1, len(gene_names) + 1)
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        importance_df['Rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def analyze_temporal_activation_patterns(self, attention_outputs, gene_names, top_n=50):
        """
        Analyze how top genes activate across time points.
        
        Args:
            attention_outputs: Attention layer outputs
            gene_names: List of gene names
            top_n: Number of top genes to analyze
            
        Returns:
            DataFrame with temporal patterns for top genes
        """
        if isinstance(attention_outputs, torch.Tensor):
            outputs_np = attention_outputs.cpu().numpy()
        else:
            outputs_np = attention_outputs
        
        # Get gene importance
        importance_df = self.compute_gene_importance_from_outputs(attention_outputs, gene_names)
        top_genes = importance_df.head(top_n)
        
        # Extract temporal patterns for top genes
        temporal_patterns = []
        time_labels = ['DX', 'REL', 'REM']
        
        for idx, row in top_genes.iterrows():
            gene_name = row['Gene']
            gene_idx = gene_names.index(gene_name)
            
            # Get activation across time: average over feature dimension
            temporal_activation = np.mean(np.abs(outputs_np[gene_idx]), axis=1)
            
            pattern_data = {
                'Gene': gene_name,
                'Importance': row['Importance'],
                'Rank': row['Rank']
            }
            
            for t, label in enumerate(time_labels[:len(temporal_activation)]):
                pattern_data[f'{label}_Activation'] = temporal_activation[t]
            
            # Identify dominant time point
            dominant_time = time_labels[np.argmax(temporal_activation)]
            pattern_data['Dominant_Timepoint'] = dominant_time
            
            temporal_patterns.append(pattern_data)
        
        return pd.DataFrame(temporal_patterns)
    
    def visualize_top_genes_temporal_patterns(self, attention_outputs, gene_names, 
                                             top_n=20, save_path=None, patient_id=None):
        """
        Visualize temporal activation patterns for top genes.
        
        Args:
            attention_outputs: Attention layer outputs
            gene_names: List of gene names
            top_n: Number of top genes to visualize
            save_path: Path to save figure
            patient_id: Patient identifier
        """
        temporal_df = self.analyze_temporal_activation_patterns(
            attention_outputs, gene_names, top_n
        )
        
        # Extract activation values
        time_labels = ['DX', 'REL', 'REM']
        activation_cols = [f'{label}_Activation' for label in time_labels 
                          if f'{label}_Activation' in temporal_df.columns]
        activation_matrix = temporal_df[activation_cols].values
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Heatmap of top genes
        sns.heatmap(activation_matrix, cmap='YlOrRd', annot=False,
                   yticklabels=temporal_df['Gene'].values,
                   xticklabels=[col.replace('_Activation', '') for col in activation_cols],
                   cbar_kws={'label': 'Activation Magnitude'},
                   ax=axes[0])
        axes[0].set_title(f'Top {top_n} Genes - Temporal Activation Patterns', fontsize=12)
        axes[0].set_xlabel('Time Point', fontsize=11)
        axes[0].set_ylabel('Gene', fontsize=11)
        
        # 2. Line plot of top 10 genes
        top_10 = temporal_df.head(10)
        for idx, row in top_10.iterrows():
            activations = [row[col] for col in activation_cols]
            axes[1].plot([col.replace('_Activation', '') for col in activation_cols], 
                        activations, marker='o', label=row['Gene'], linewidth=2)
        
        axes[1].set_xlabel('Time Point', fontsize=11)
        axes[1].set_ylabel('Activation Magnitude', fontsize=11)
        axes[1].set_title('Top 10 Genes - Activation Trajectories', fontsize=12)
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1].grid(alpha=0.3)
        
        if patient_id:
            fig.suptitle(f'Gene-Level Temporal Analysis - Patient {patient_id}', 
                        fontsize=14, y=1.00)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved top genes temporal patterns to {save_path}")
        
        plt.close()
    
    def cluster_genes_by_temporal_patterns(self, attention_outputs, gene_names, 
                                          n_clusters=5, save_path=None, patient_id=None):
        """
        Cluster genes based on their temporal activation patterns.
        
        Args:
            attention_outputs: Attention layer outputs
            gene_names: List of gene names
            n_clusters: Number of clusters
            save_path: Path to save results
            patient_id: Patient identifier
            
        Returns:
            DataFrame with cluster assignments
        """
        if isinstance(attention_outputs, torch.Tensor):
            outputs_np = attention_outputs.cpu().numpy()
        else:
            outputs_np = attention_outputs
        
        # Compute temporal profiles: (num_genes, num_timepoints)
        temporal_profiles = np.mean(np.abs(outputs_np), axis=2)
        
        # Normalize each gene's profile
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        temporal_profiles_normalized = scaler.fit_transform(temporal_profiles)
        
        # Cluster genes
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(temporal_profiles_normalized)
        
        # Create result dataframe
        cluster_df = pd.DataFrame({
            'Gene': gene_names,
            'Cluster': clusters
        })
        
        # Compute cluster centers in original space
        time_labels = ['DX', 'REL', 'REM']
        cluster_profiles = []
        
        for c in range(n_clusters):
            cluster_mask = clusters == c
            cluster_mean = np.mean(temporal_profiles[cluster_mask], axis=0)
            
            profile_data = {
                'Cluster': c,
                'Size': np.sum(cluster_mask),
                'Pattern_Type': self._classify_temporal_pattern(cluster_mean)
            }
            
            for t, label in enumerate(time_labels[:len(cluster_mean)]):
                profile_data[f'{label}_Mean'] = cluster_mean[t]
            
            cluster_profiles.append(profile_data)
        
        cluster_summary = pd.DataFrame(cluster_profiles)
        
        # Visualize clusters
        if save_path:
            self._visualize_gene_clusters(temporal_profiles, clusters, cluster_summary, 
                                         save_path, patient_id)
        
        return cluster_df, cluster_summary
    
    def _classify_temporal_pattern(self, temporal_profile):
        """Classify temporal pattern based on activation peaks."""
        peak_idx = np.argmax(temporal_profile)
        
        if peak_idx == 0:
            return 'Early_Peak'
        elif peak_idx == len(temporal_profile) - 1:
            return 'Late_Peak'
        else:
            return 'Middle_Peak'
    
    def _visualize_gene_clusters(self, temporal_profiles, clusters, cluster_summary,
                                save_path, patient_id):
        """Visualize gene clustering results."""
        time_labels = ['DX', 'REL', 'REM']
        n_clusters = len(np.unique(clusters))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Cluster centers
        cluster_means = []
        for c in range(n_clusters):
            cluster_mask = clusters == c
            cluster_mean = np.mean(temporal_profiles[cluster_mask], axis=0)
            cluster_means.append(cluster_mean)
            
            pattern_type = cluster_summary[cluster_summary['Cluster'] == c]['Pattern_Type'].values[0]
            size = cluster_summary[cluster_summary['Cluster'] == c]['Size'].values[0]
            
            axes[0].plot(time_labels[:len(cluster_mean)], cluster_mean, 
                        marker='o', linewidth=2, 
                        label=f'C{c}: {pattern_type} (n={size})')
        
        axes[0].set_xlabel('Time Point', fontsize=11)
        axes[0].set_ylabel('Mean Activation', fontsize=11)
        axes[0].set_title('Gene Cluster Centers', fontsize=12)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 2. Cluster sizes
        cluster_sizes = [np.sum(clusters == c) for c in range(n_clusters)]
        axes[1].bar(range(n_clusters), cluster_sizes, color='steelblue', alpha=0.7)
        axes[1].set_xlabel('Cluster', fontsize=11)
        axes[1].set_ylabel('Number of Genes', fontsize=11)
        axes[1].set_title('Cluster Size Distribution', fontsize=12)
        axes[1].set_xticks(range(n_clusters))
        axes[1].grid(axis='y', alpha=0.3)
        
        if patient_id:
            fig.suptitle(f'Gene Clustering by Temporal Patterns - Patient {patient_id}', 
                        fontsize=14, y=1.00)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved gene clustering visualization to {save_path}")
        plt.close()
    
    # ========================================================================
    # TEMPORAL PATTERN ANALYSIS (Secondary Focus - Using Attention Weights)
    # ========================================================================
    
    def analyze_global_attention_patterns(self, attention_weights):
        """
        Analyze high-level temporal attention patterns across all genes.
        
        Args:
            attention_weights: Attention weights tensor
            
        Returns:
            Dictionary with global attention statistics
        """
        if isinstance(attention_weights, torch.Tensor):
            weights_np = attention_weights.cpu().numpy()
        else:
            weights_np = attention_weights
        
        # Handle different shapes
        if len(weights_np.shape) == 4:
            attn = weights_np[:, 0]  # First head
        elif len(weights_np.shape) == 3:
            attn = weights_np
        else:
            attn = weights_np[np.newaxis, ...]
        
        # Compute mean attention pattern across all genes
        mean_attention = np.mean(attn, axis=0)
        
        time_labels = ['DX', 'REL', 'REM']
        
        # Analyze temporal transitions
        transitions = {}
        for i, from_label in enumerate(time_labels[:mean_attention.shape[0]]):
            for j, to_label in enumerate(time_labels[:mean_attention.shape[1]]):
                transitions[f'{from_label}_to_{to_label}'] = mean_attention[i, j]
        
        # Identify dominant patterns
        self_attention = np.diag(mean_attention)
        cross_attention = mean_attention.copy()
        np.fill_diagonal(cross_attention, 0)
        
        stats = {
            'mean_attention_matrix': mean_attention,
            'transitions': transitions,
            'mean_self_attention': np.mean(self_attention),
            'mean_cross_attention': np.mean(cross_attention),
            'self_vs_cross_ratio': np.mean(self_attention) / (np.mean(cross_attention) + 1e-10),
            'attention_entropy': self._compute_matrix_entropy(mean_attention)
        }
        
        return stats
    
    def _compute_matrix_entropy(self, attention_matrix):
        """Compute entropy of attention distribution."""
        epsilon = 1e-10
        entropy = -np.sum(attention_matrix * np.log(attention_matrix + epsilon))
        return entropy
    
    def visualize_global_attention_patterns(self, attention_weights, save_path=None, 
                                           patient_id=None):
        """
        Visualize high-level temporal attention patterns.
        
        Args:
            attention_weights: Attention weights tensor
            save_path: Path to save figure
            patient_id: Patient identifier
        """
        stats = self.analyze_global_attention_patterns(attention_weights)
        mean_attention = stats['mean_attention_matrix']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        time_labels = ['DX', 'REL', 'REM']
        
        # 1. Mean attention heatmap
        sns.heatmap(mean_attention, annot=True, fmt='.3f', cmap='Blues',
                   square=True, ax=axes[0],
                   xticklabels=time_labels[:mean_attention.shape[1]],
                   yticklabels=time_labels[:mean_attention.shape[0]],
                   cbar_kws={'label': 'Attention Weight'},
                   vmin=0, vmax=1)
        axes[0].set_title('Global Attention Pattern\n(Averaged Across All Genes)', fontsize=12)
        axes[0].set_xlabel('Key (Attend To)', fontsize=11)
        axes[0].set_ylabel('Query (Attend From)', fontsize=11)
        
        # 2. Attention statistics
        stats_text = f"""
        Self-Attention (diagonal): {stats['mean_self_attention']:.3f}
        Cross-Attention (off-diagonal): {stats['mean_cross_attention']:.3f}
        Self/Cross Ratio: {stats['self_vs_cross_ratio']:.2f}
        
        Attention Entropy: {stats['attention_entropy']:.3f}
        
        Key Transitions:
        DX → REL: {stats['transitions'].get('DX_to_REL', 0):.3f}
        REL → REM: {stats['transitions'].get('REL_to_REM', 0):.3f}
        DX → REM: {stats['transitions'].get('DX_to_REM', 0):.3f}
        """
        
        axes[1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        axes[1].axis('off')
        axes[1].set_title('Attention Statistics', fontsize=12)
        
        if patient_id:
            fig.suptitle(f'Global Temporal Attention Analysis - Patient {patient_id}', 
                        fontsize=14, y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved global attention patterns to {save_path}")
        
        plt.close()
    
    # ========================================================================
    # COMPREHENSIVE REPORT GENERATION
    # ========================================================================
    
    def generate_comprehensive_report(self, sequences, labels, gene_names, 
                                     output_dir, patient_id):
        """
        Generate a comprehensive analysis report combining both approaches.
        
        Args:
            sequences: Input sequences
            labels: True labels
            gene_names: List of gene names
            output_dir: Directory to save all outputs
            patient_id: Patient identifier
        """
        print(f"Generating comprehensive attention analysis for patient {patient_id}...")
        
        # Create subdirectory
        attention_dir = os.path.join(output_dir, f"{patient_id}_attention_analysis")
        os.makedirs(attention_dir, exist_ok=True)
        
        # Extract attention data
        attention_outputs, attention_weights = self.extract_attention_data(sequences)
        
        print("  1/5 Computing gene importance from attention outputs...")
        # Gene-level analysis (PRIMARY)
        importance_df = self.compute_gene_importance_from_outputs(attention_outputs, gene_names)
        importance_df.to_csv(
            os.path.join(attention_dir, f"{patient_id}_gene_importance_rankings.csv"),
            index=False
        )
        
        print("  2/5 Analyzing temporal activation patterns...")
        # Temporal patterns for top genes
        temporal_df = self.analyze_temporal_activation_patterns(
            attention_outputs, gene_names, top_n=100
        )
        temporal_df.to_csv(
            os.path.join(attention_dir, f"{patient_id}_top_genes_temporal_patterns.csv"),
            index=False
        )
        
        print("  3/5 Visualizing top genes...")
        # Visualize top genes
        self.visualize_top_genes_temporal_patterns(
            attention_outputs, gene_names, top_n=20,
            save_path=os.path.join(attention_dir, f"{patient_id}_top_genes_temporal.png"),
            patient_id=patient_id
        )
        
        print("  4/5 Clustering genes by temporal patterns...")
        # Cluster genes by patterns
        cluster_df, cluster_summary = self.cluster_genes_by_temporal_patterns(
            attention_outputs, gene_names, n_clusters=5,
            save_path=os.path.join(attention_dir, f"{patient_id}_gene_clusters.png"),
            patient_id=patient_id
        )
        cluster_df.to_csv(
            os.path.join(attention_dir, f"{patient_id}_gene_cluster_assignments.csv"),
            index=False
        )
        cluster_summary.to_csv(
            os.path.join(attention_dir, f"{patient_id}_cluster_summary.csv"),
            index=False
        )
        
        print("  5/5 Analyzing global attention patterns...")
        # Global temporal pattern analysis (SECONDARY)
        self.visualize_global_attention_patterns(
            attention_weights,
            save_path=os.path.join(attention_dir, f"{patient_id}_global_attention_patterns.png"),
            patient_id=patient_id
        )
        
        global_stats = self.analyze_global_attention_patterns(attention_weights)
        
        # Save global statistics
        global_stats_df = pd.DataFrame([
            {'Metric': 'Mean Self-Attention', 'Value': global_stats['mean_self_attention']},
            {'Metric': 'Mean Cross-Attention', 'Value': global_stats['mean_cross_attention']},
            {'Metric': 'Self/Cross Ratio', 'Value': global_stats['self_vs_cross_ratio']},
            {'Metric': 'Attention Entropy', 'Value': global_stats['attention_entropy']},
        ])
        global_stats_df.to_csv(
            os.path.join(attention_dir, f"{patient_id}_global_attention_statistics.csv"),
            index=False
        )
        
        # Generate summary report
        self._generate_summary_report(
            importance_df, temporal_df, cluster_summary, global_stats,
            attention_dir, patient_id
        )
        
        print(f"✓ Analysis complete! Results saved to {attention_dir}")
        
        return {
            'importance': importance_df,
            'temporal_patterns': temporal_df,
            'clusters': (cluster_df, cluster_summary),
            'global_stats': global_stats
        }
    
    def _generate_summary_report(self, importance_df, temporal_df, cluster_summary, 
                                global_stats, output_dir, patient_id):
        """Generate a text summary report."""
        report_path = os.path.join(output_dir, f"{patient_id}_analysis_summary.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"ATTENTION ANALYSIS SUMMARY - Patient {patient_id}\n")
            f.write("="*70 + "\n\n")
            
            f.write("GENE-LEVEL ANALYSIS (from Attention Outputs)\n")
            f.write("-"*70 + "\n")
            f.write(f"Total genes analyzed: {len(importance_df)}\n\n")
            
            f.write("Top 10 Most Important Genes:\n")
            for idx, row in importance_df.head(10).iterrows():
                f.write(f"  {row['Rank']:2d}. {row['Gene']:20s} (Importance: {row['Importance']:.4f})\n")
            
            f.write("\n\nTemporal Pattern Clusters:\n")
            f.write("-"*70 + "\n")
            for _, row in cluster_summary.iterrows():
                f.write(f"Cluster {row['Cluster']}: {row['Pattern_Type']:15s} "
                       f"(n={row['Size']} genes)\n")
            
            f.write("\n\nGLOBAL TEMPORAL PATTERNS (from Attention Weights)\n")
            f.write("-"*70 + "\n")
            f.write(f"Mean Self-Attention:  {global_stats['mean_self_attention']:.3f}\n")
            f.write(f"Mean Cross-Attention: {global_stats['mean_cross_attention']:.3f}\n")
            f.write(f"Self/Cross Ratio:     {global_stats['self_vs_cross_ratio']:.2f}\n")
            f.write(f"Attention Entropy:    {global_stats['attention_entropy']:.3f}\n\n")
            
            f.write("Key Temporal Transitions:\n")
            for trans_name, trans_val in global_stats['transitions'].items():
                f.write(f"  {trans_name:15s}: {trans_val:.3f}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"Saved summary report to {report_path}")


def analyze_attention_for_patient(model, sequences, labels, gene_names, 
                                 output_dir, patient_id, device='cpu'):
    """
    Convenience function to run full attention analysis for a patient.
    
    Args:
        model: Trained PyTorchTransformerModel
        sequences: Input sequences tensor
        labels: True labels
        gene_names: List of gene names
        output_dir: Directory to save outputs
        patient_id: Patient identifier
        device: Device to run on
        
    Returns:
        Analyzer instance and analysis results
    """
    analyzer = AttentionAnalyzer(model, device=device)
    results = analyzer.generate_comprehensive_report(
        sequences, labels, gene_names, output_dir, patient_id
    )
    
    return analyzer, results