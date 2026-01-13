import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

class AttentionAnalyzer:
    """
    Analyzes and visualizes attention weights from the Transformer model.
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
    
    def extract_attention_weights(self, sequences):
        """
        Extract attention weights for given sequences.
        
        Args:
            sequences: Input sequences tensor (num_samples, timesteps, features)
            
        Returns:
            attention_weights: Tensor of shape (num_samples, num_heads, seq_len, seq_len)
            predictions: Model predictions
        """
        with torch.no_grad():
            if not isinstance(sequences, torch.Tensor):
                sequences = torch.tensor(sequences, dtype=torch.float32)
            sequences = sequences.to(self.device)
            
            predictions, attention_weights = self.model(
                sequences, 
                return_attention_weights=True
            )
            
        return attention_weights, predictions
    
    def visualize_attention_heatmap(self, attention_weights, gene_names=None, 
                                   sample_idx=0, save_path=None, title=None):
        """
        Create a heatmap visualization of attention weights for a single sample.
        
        Args:
            attention_weights: Attention weights tensor
            gene_names: List of gene names (optional)
            sample_idx: Index of sample to visualize
            save_path: Path to save the figure
            title: Custom title for the plot
        """
        # Extract attention for the specified sample
        # Shape can be: (batch, num_heads, seq_len, seq_len), (batch, seq_len, seq_len), or (seq_len, seq_len)
        if isinstance(attention_weights, torch.Tensor):
            attn_np = attention_weights.cpu().numpy()
        else:
            attn_np = attention_weights
            
        if len(attn_np.shape) == 4:
            # (batch, num_heads, seq_len, seq_len)
            attn = attn_np[sample_idx, 0]  # First head
        elif len(attn_np.shape) == 3:
            # (batch, seq_len, seq_len)
            attn = attn_np[sample_idx]
        else:
            # (seq_len, seq_len)
            attn = attn_np
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(attn, cmap='viridis', annot=False, 
                   square=True, cbar_kws={'label': 'Attention Weight'},
                   ax=ax)
        
        # Set labels
        time_labels = ['DX', 'REL', 'REM']
        ax.set_xlabel('Key Position (Time Point)', fontsize=12)
        ax.set_ylabel('Query Position (Time Point)', fontsize=12)
        ax.set_xticklabels(time_labels, rotation=0)
        ax.set_yticklabels(time_labels, rotation=0)
        
        if title:
            ax.set_title(title, fontsize=14, pad=20)
        else:
            ax.set_title(f'Attention Weights - Sample {sample_idx}', 
                        fontsize=14, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention heatmap to {save_path}")
        
        plt.close()
    
    def visualize_attention_patterns_grid(self, attention_weights, num_samples=9,
                                         save_path=None, patient_id=None):
        """
        Create a grid of attention heatmaps for multiple samples.
        
        Args:
            attention_weights: Attention weights tensor
            num_samples: Number of samples to visualize
            save_path: Path to save the figure
            patient_id: Patient identifier for the title
        """
        # Convert to numpy if needed
        if isinstance(attention_weights, torch.Tensor):
            attn_np = attention_weights.cpu().numpy()
        else:
            attn_np = attention_weights
        
        # Determine grid size
        actual_num_samples = min(num_samples, attn_np.shape[0])
        grid_size = int(np.ceil(np.sqrt(actual_num_samples)))
        
        fig, axes = plt.subplots(grid_size, grid_size, 
                                figsize=(4*grid_size, 4*grid_size))
        axes = axes.flatten()
        
        time_labels = ['DX', 'REL', 'REM']
        
        for idx in range(actual_num_samples):
            # Extract attention for this sample based on shape
            if len(attn_np.shape) == 4:
                # (batch, num_heads, seq_len, seq_len)
                attn = attn_np[idx, 0]
            elif len(attn_np.shape) == 3:
                # (batch, seq_len, seq_len)
                attn = attn_np[idx]
            else:
                # Not enough samples, break
                break
            
            ax = axes[idx]
            sns.heatmap(attn, cmap='viridis', annot=True, fmt='.2f',
                       square=True, cbar=False, ax=ax,
                       xticklabels=time_labels, yticklabels=time_labels)
            ax.set_title(f'Sample {idx}', fontsize=10)
            
            if idx % grid_size != 0:
                ax.set_ylabel('')
            if idx < actual_num_samples - grid_size:
                ax.set_xlabel('')
        
        # Hide unused subplots
        for idx in range(actual_num_samples, len(axes)):
            axes[idx].axis('off')
        
        if patient_id:
            fig.suptitle(f'Attention Patterns - Patient {patient_id}', 
                        fontsize=16, y=0.995)
        else:
            fig.suptitle('Attention Weight Patterns Across Samples', 
                        fontsize=16, y=0.995)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention grid to {save_path}")
        
        plt.close()
    
    def aggregate_attention_statistics(self, attention_weights):
        """
        Compute aggregate statistics about attention patterns.
        
        Args:
            attention_weights: Attention weights tensor
            
        Returns:
            Dictionary with attention statistics
        """
        # Convert to numpy if needed
        if isinstance(attention_weights, torch.Tensor):
            attn_np = attention_weights.cpu().numpy()
        else:
            attn_np = attention_weights
            
        # Handle different shapes
        if len(attn_np.shape) == 4:
            # (batch, num_heads, seq_len, seq_len) - take first head
            attn = attn_np[:, 0]
        elif len(attn_np.shape) == 3:
            # (batch, seq_len, seq_len)
            attn = attn_np
        else:
            # (seq_len, seq_len) - single sample
            attn = attn_np[np.newaxis, ...]
        
        stats = {
            'mean_attention': np.mean(attn, axis=0),
            'std_attention': np.std(attn, axis=0),
            'max_attention': np.max(attn, axis=0),
            'min_attention': np.min(attn, axis=0),
        }
        
        # Temporal focus: which time points receive most attention
        time_labels = ['DX', 'REL', 'REM']
        # Average over samples and queries (rows)
        attention_by_timepoint = np.mean(attn, axis=(0, 1))
        
        stats['attention_by_timepoint'] = dict(zip(time_labels, attention_by_timepoint))
        
        return stats
    
    def visualize_aggregate_attention(self, attention_weights, save_path=None,
                                     patient_id=None):
        """
        Visualize average attention pattern across all samples.
        
        Args:
            attention_weights: Attention weights tensor
            save_path: Path to save the figure
            patient_id: Patient identifier
        """
        stats = self.aggregate_attention_statistics(attention_weights)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Mean attention heatmap
        time_labels = ['DX', 'REL', 'REM']
        sns.heatmap(stats['mean_attention'], annot=True, fmt='.3f',
                   cmap='viridis', square=True, ax=axes[0],
                   xticklabels=time_labels, yticklabels=time_labels,
                   cbar_kws={'label': 'Mean Attention'})
        axes[0].set_title('Mean Attention Across All Samples', fontsize=12)
        axes[0].set_xlabel('Key Position (Time Point)')
        axes[0].set_ylabel('Query Position (Time Point)')
        
        # Attention by timepoint bar plot
        timepoint_attention = list(stats['attention_by_timepoint'].values())
        axes[1].bar(time_labels, timepoint_attention, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1].set_ylabel('Average Attention Received', fontsize=11)
        axes[1].set_xlabel('Time Point', fontsize=11)
        axes[1].set_title('Average Attention Received by Each Time Point', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        if patient_id:
            fig.suptitle(f'Aggregate Attention Analysis - Patient {patient_id}', 
                        fontsize=14, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved aggregate attention visualization to {save_path}")
        
        plt.close()
    
    def analyze_attention_by_class(self, attention_weights, labels, 
                                   save_path=None, patient_id=None):
        """
        Compare attention patterns across different classes (DX, REL, REM).
        
        Args:
            attention_weights: Attention weights tensor
            labels: True labels for each sample
            save_path: Path to save the figure
            patient_id: Patient identifier
        """
        # Convert to numpy
        if isinstance(attention_weights, torch.Tensor):
            attn_np = attention_weights.cpu().numpy()
        else:
            attn_np = attention_weights
            
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # Handle different shapes
        if len(attn_np.shape) == 4:
            # (batch, num_heads, seq_len, seq_len)
            attn = attn_np[:, 0]
        elif len(attn_np.shape) == 3:
            # (batch, seq_len, seq_len)
            attn = attn_np
        else:
            print("Warning: attention_weights shape not suitable for class analysis")
            return
        
        class_names = ['DX', 'REL', 'REM']
        unique_classes = np.unique(labels)
        
        fig, axes = plt.subplots(1, len(unique_classes), 
                                figsize=(6*len(unique_classes), 5))
        
        if len(unique_classes) == 1:
            axes = [axes]
        
        time_labels = ['DX', 'REL', 'REM']
        
        for idx, class_label in enumerate(unique_classes):
            # Get attention weights for this class
            class_mask = labels == class_label
            class_attention = attn[class_mask]
            
            if len(class_attention) == 0:
                continue
            
            # Average attention for this class
            mean_class_attention = np.mean(class_attention, axis=0)
            
            sns.heatmap(mean_class_attention, annot=True, fmt='.3f',
                       cmap='viridis', square=True, ax=axes[idx],
                       xticklabels=time_labels, yticklabels=time_labels,
                       cbar_kws={'label': 'Mean Attention'})
            
            class_name = class_names[class_label] if class_label < len(class_names) else f'Class {class_label}'
            axes[idx].set_title(f'{class_name} Samples (n={np.sum(class_mask)})', 
                               fontsize=12)
            axes[idx].set_xlabel('Key Position')
            axes[idx].set_ylabel('Query Position')
        
        if patient_id:
            fig.suptitle(f'Attention Patterns by Class - Patient {patient_id}', 
                        fontsize=14, y=1.02)
        else:
            fig.suptitle('Attention Patterns by True Class', 
                        fontsize=14, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved class-based attention analysis to {save_path}")
        
        plt.close()
    
    def compute_attention_entropy(self, attention_weights):
        """
        Compute entropy of attention distributions to measure focus/dispersion.
        
        Args:
            attention_weights: Attention weights tensor
            
        Returns:
            entropy_per_sample: Entropy values for each sample
            mean_entropy: Mean entropy across all samples
        """
        # Convert to numpy
        if isinstance(attention_weights, torch.Tensor):
            attn_np = attention_weights.cpu().numpy()
        else:
            attn_np = attention_weights
            
        # Handle different shapes
        if len(attn_np.shape) == 4:
            # (batch, num_heads, seq_len, seq_len)
            attn = attn_np[:, 0]
        elif len(attn_np.shape) == 3:
            # (batch, seq_len, seq_len)
            attn = attn_np
        else:
            # (seq_len, seq_len) - single sample
            attn = attn_np[np.newaxis, ...]
        
        # Compute entropy for each query position in each sample
        epsilon = 1e-10  # Avoid log(0)
        entropy = -np.sum(attn * np.log(attn + epsilon), axis=-1)
        
        # Average entropy per sample (across all query positions)
        entropy_per_sample = np.mean(entropy, axis=-1)
        mean_entropy = np.mean(entropy_per_sample)
        
        return entropy_per_sample, mean_entropy
    
    def visualize_attention_entropy(self, attention_weights, labels=None,
                                   save_path=None, patient_id=None):
        """
        Visualize attention entropy distribution.
        
        Args:
            attention_weights: Attention weights tensor
            labels: Optional labels for coloring by class
            save_path: Path to save the figure
            patient_id: Patient identifier
        """
        entropy_per_sample, mean_entropy = self.compute_attention_entropy(attention_weights)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of entropy values
        axes[0].hist(entropy_per_sample, bins=20, color='steelblue', 
                    edgecolor='black', alpha=0.7)
        axes[0].axvline(mean_entropy, color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {mean_entropy:.3f}')
        axes[0].set_xlabel('Attention Entropy', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Distribution of Attention Entropy', fontsize=12)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Scatter plot of entropy vs sample index, colored by class if available
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            
            class_names = ['DX', 'REL', 'REM']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            for class_label in np.unique(labels):
                mask = labels == class_label
                class_name = class_names[class_label] if class_label < len(class_names) else f'Class {class_label}'
                axes[1].scatter(np.where(mask)[0], entropy_per_sample[mask],
                              label=class_name, alpha=0.6, s=50,
                              color=colors[class_label % len(colors)])
        else:
            axes[1].scatter(range(len(entropy_per_sample)), entropy_per_sample,
                          alpha=0.6, s=50, color='steelblue')
        
        axes[1].set_xlabel('Sample Index', fontsize=11)
        axes[1].set_ylabel('Attention Entropy', fontsize=11)
        axes[1].set_title('Attention Entropy per Sample', fontsize=12)
        if labels is not None:
            axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        if patient_id:
            fig.suptitle(f'Attention Entropy Analysis - Patient {patient_id}', 
                        fontsize=14, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved entropy analysis to {save_path}")
        
        plt.close()
    
    def generate_attention_report(self, sequences, labels, gene_names, 
                                 output_dir, patient_id):
        """
        Generate a comprehensive attention analysis report.
        
        Args:
            sequences: Input sequences
            labels: True labels
            gene_names: List of gene names
            output_dir: Directory to save all outputs
            patient_id: Patient identifier
        """
        print(f"Generating attention analysis report for patient {patient_id}...")
        
        # Create subdirectory for attention analysis
        attention_dir = os.path.join(output_dir, f"{patient_id}_attention_analysis")
        os.makedirs(attention_dir, exist_ok=True)
        
        # Extract attention weights
        attention_weights, predictions = self.extract_attention_weights(sequences)
        
        # 1. Aggregate attention visualization
        self.visualize_aggregate_attention(
            attention_weights,
            save_path=os.path.join(attention_dir, f"{patient_id}_aggregate_attention.png"),
            patient_id=patient_id
        )
        
        # 2. Attention patterns grid (first 9 samples)
        self.visualize_attention_patterns_grid(
            attention_weights,
            num_samples=9,
            save_path=os.path.join(attention_dir, f"{patient_id}_attention_grid.png"),
            patient_id=patient_id
        )
        
        # 3. Class-based attention analysis
        self.analyze_attention_by_class(
            attention_weights,
            labels,
            save_path=os.path.join(attention_dir, f"{patient_id}_attention_by_class.png"),
            patient_id=patient_id
        )
        
        # 4. Entropy analysis
        self.visualize_attention_entropy(
            attention_weights,
            labels=labels,
            save_path=os.path.join(attention_dir, f"{patient_id}_attention_entropy.png"),
            patient_id=patient_id
        )
        
        # 5. Save attention statistics to CSV
        stats = self.aggregate_attention_statistics(attention_weights)
        entropy_per_sample, mean_entropy = self.compute_attention_entropy(attention_weights)
        
        stats_df = pd.DataFrame({
            'Metric': ['Mean Attention DX->DX', 'Mean Attention DX->REL', 'Mean Attention DX->REM',
                      'Mean Attention REL->DX', 'Mean Attention REL->REL', 'Mean Attention REL->REM',
                      'Mean Attention REM->DX', 'Mean Attention REM->REL', 'Mean Attention REM->REM',
                      'Mean Entropy'],
            'Value': [stats['mean_attention'][0,0], stats['mean_attention'][0,1], stats['mean_attention'][0,2],
                     stats['mean_attention'][1,0], stats['mean_attention'][1,1], stats['mean_attention'][1,2],
                     stats['mean_attention'][2,0], stats['mean_attention'][2,1], stats['mean_attention'][2,2],
                     mean_entropy]
        })
        
        stats_df.to_csv(
            os.path.join(attention_dir, f"{patient_id}_attention_statistics.csv"),
            index=False
        )
        
        print(f"Attention analysis report generated and saved to {attention_dir}")
        
        return attention_weights, stats


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
    """
    analyzer = AttentionAnalyzer(model, device=device)
    attention_weights, stats = analyzer.generate_attention_report(
        sequences, labels, gene_names, output_dir, patient_id
    )
    
    return analyzer, attention_weights, stats