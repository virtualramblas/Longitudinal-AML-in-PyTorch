"""
Streamlined integration script for attention analysis.

Focuses on:
- Gene-level analysis using attention outputs
- High-level temporal pattern analysis using attention weights
"""

import os
import argparse
import matplotlib.pyplot as plt
import torch
from data_preprocessing import process_single_patient_data
from transformer_training import train_transformer
from hybrid_attention_analysis import analyze_attention_for_patient

def train_and_analyze_with_attention(raw_data_dir, output_root_dir):
    """
    Enhanced training pipeline with streamlined attention analysis.
    
    Args:
        raw_data_dir: Root directory of raw data archives
        output_root_dir: Root output directory
    """
    # Determine device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    
    print(f"Using device: {device}")
    
    # Setup directories
    processed_matrices_dir = os.path.join(raw_data_dir, "processed_matrices_csv")
    output_dir = os.path.join(output_root_dir, 'transformer_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize files by patient
    file_groups = {}
    for file in os.listdir(processed_matrices_dir):
        if not file.endswith("_processed_matrix.csv"):
            continue
        parts = file.split("_")
        patient, state = parts[1], parts[2]
        file_groups.setdefault(patient, {})[state] = file
    
    print(f"Found {len(file_groups)} patients to process")
    
    # Process each patient
    all_patient_results = []
    
    for patient, files in file_groups.items():
        print(f"\n{'='*70}")
        print(f"Processing Patient: {patient}")
        print(f"{'='*70}")
        
        try:
            # 1. Process patient data
            sequences, labels, gene_names = process_single_patient_data(
                processed_matrices_dir, patient, files
            )
            print(f"Loaded {len(gene_names)} genes with {sequences.shape[1]} time points")
            
            # 2. Train transformer model
            model, processed_sequences = train_transformer(
                sequences, labels, patient, output_dir, device
            )
            print("✓ Model training completed")
            
            # 3. Run comprehensive attention analysis
            print("\nRunning attention analysis...")
            
            sequences_torch = torch.from_numpy(processed_sequences).to(
                dtype=torch.float32, device=device
            )
            
            # Prepare labels for analysis
            if labels is None or len(labels) != sequences_torch.shape[0]:
                analysis_labels = torch.randint(0, 3, (sequences_torch.shape[0],))
            else:
                analysis_labels = torch.tensor(labels, dtype=torch.long)
            
            # Run comprehensive analysis
            analyzer, results = analyze_attention_for_patient(
                model=model,
                sequences=sequences_torch,
                labels=analysis_labels,
                gene_names=gene_names,
                output_dir=output_dir,
                patient_id=patient,
                device=device
            )
            
            # 4. Extract key insights
            importance_df = results['importance']
            #temporal_patterns = results['temporal_patterns']
            cluster_df, cluster_summary = results['clusters']
            global_stats = results['global_stats']
            
            # Store summary for cross-patient analysis
            patient_result = {
                'patient_id': patient,
                'num_genes': len(gene_names),
                'top_gene': importance_df.iloc[0]['Gene'],
                'top_gene_importance': importance_df.iloc[0]['Importance'],
                'mean_self_attention': global_stats['mean_self_attention'],
                'mean_cross_attention': global_stats['mean_cross_attention'],
                'self_cross_ratio': global_stats['self_vs_cross_ratio'],
                'attention_entropy': global_stats['attention_entropy'],
                'num_clusters': len(cluster_summary),
                'largest_cluster_size': cluster_summary['Size'].max(),
                'largest_cluster_pattern': cluster_summary.loc[
                    cluster_summary['Size'].idxmax(), 'Pattern_Type'
                ]
            }
            all_patient_results.append(patient_result)
            
            # Print key insights
            print(f"\n{'='*70}")
            print(f"Key Insights for Patient {patient}:")
            print(f"{'='*70}")
            print(f"  Top Gene: {patient_result['top_gene']} "
                  f"(Importance: {patient_result['top_gene_importance']:.4f})")
            print("  Global Attention Pattern:")
            print(f"    - Self-attention: {global_stats['mean_self_attention']:.3f}")
            print(f"    - Cross-attention: {global_stats['mean_cross_attention']:.3f}")
            print(f"    - Ratio: {global_stats['self_vs_cross_ratio']:.2f}")
            print(f"  Temporal Clusters: {len(cluster_summary)}")
            print(f"  Dominant Pattern: {patient_result['largest_cluster_pattern']} "
                  f"({patient_result['largest_cluster_size']} genes)")
            
        except Exception as e:
            print(f"✗ Error processing patient {patient}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 5. Generate cross-patient summary
    if all_patient_results:
        print(f"\n{'='*70}")
        print("CROSS-PATIENT ANALYSIS")
        print(f"{'='*70}")
        
        import pandas as pd
        summary_df = pd.DataFrame(all_patient_results)
        
        # Save summary
        summary_path = os.path.join(output_dir, "cross_patient_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved cross-patient summary to {summary_path}")
        
        # Print statistics
        print("\nCross-Patient Statistics:")
        print(f"  Total patients: {len(all_patient_results)}")
        print("\n  Attention Pattern Statistics:")
        print(f"    Mean self-attention: {summary_df['mean_self_attention'].mean():.3f} "
              f"(±{summary_df['mean_self_attention'].std():.3f})")
        print(f"    Mean cross-attention: {summary_df['mean_cross_attention'].mean():.3f} "
              f"(±{summary_df['mean_cross_attention'].std():.3f})")
        print(f"    Mean self/cross ratio: {summary_df['self_cross_ratio'].mean():.2f} "
              f"(±{summary_df['self_cross_ratio'].std():.2f})")
        print(f"    Mean entropy: {summary_df['attention_entropy'].mean():.3f} "
              f"(±{summary_df['attention_entropy'].std():.3f})")
        
        print("\n  Temporal Pattern Distribution:")
        pattern_counts = summary_df['largest_cluster_pattern'].value_counts()
        for pattern, count in pattern_counts.items():
            print(f"    {pattern}: {count} patients")
        
        # Visualize cross-patient patterns
        visualize_cross_patient_patterns(summary_df, output_dir)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"All results saved to: {output_dir}")


def visualize_cross_patient_patterns(summary_df, output_dir):
    """
    Create cross-patient visualizations.
    
    Args:
        summary_df: DataFrame with patient summaries
        output_dir: Output directory
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Self vs Cross attention
    axes[0, 0].scatter(summary_df['mean_self_attention'], 
                      summary_df['mean_cross_attention'],
                      alpha=0.6, s=100)
    axes[0, 0].set_xlabel('Mean Self-Attention', fontsize=11)
    axes[0, 0].set_ylabel('Mean Cross-Attention', fontsize=11)
    axes[0, 0].set_title('Self vs Cross Attention Across Patients', fontsize=12)
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Self/Cross ratio distribution
    axes[0, 1].hist(summary_df['self_cross_ratio'], bins=15, 
                   color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Self/Cross Attention Ratio', fontsize=11)
    axes[0, 1].set_ylabel('Number of Patients', fontsize=11)
    axes[0, 1].set_title('Distribution of Attention Ratios', fontsize=12)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Attention entropy distribution
    axes[1, 0].hist(summary_df['attention_entropy'], bins=15,
                   color='coral', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Attention Entropy', fontsize=11)
    axes[1, 0].set_ylabel('Number of Patients', fontsize=11)
    axes[1, 0].set_title('Distribution of Attention Entropy', fontsize=12)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Dominant pattern distribution
    pattern_counts = summary_df['largest_cluster_pattern'].value_counts()
    axes[1, 1].bar(range(len(pattern_counts)), pattern_counts.values,
                  color='mediumseagreen', alpha=0.7)
    axes[1, 1].set_xticks(range(len(pattern_counts)))
    axes[1, 1].set_xticklabels(pattern_counts.index, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Number of Patients', fontsize=11)
    axes[1, 1].set_title('Dominant Temporal Patterns', fontsize=12)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    fig.suptitle('Cross-Patient Attention Pattern Analysis', fontsize=14, y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "cross_patient_attention_patterns.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved cross-patient visualization to {save_path}")
    plt.close()


def analyze_existing_models(models_dir, data_dir, output_dir):
    """
    Run attention analysis on pre-trained models.
    
    Args:
        models_dir: Directory containing saved models
        data_dir: Directory containing patient data
        output_dir: Output directory
    """
    print("Analyzing existing models...")
    # Implementation for analyzing pre-trained models
    # This would load models and data, then run the attention analysis
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train Transformer with streamlined attention analysis"
    )
    parser.add_argument(
        "--raw_data_dir", 
        type=str, 
        default=".", 
        help="Root directory of raw patient data"
    )
    parser.add_argument(
        "--output_root_dir", 
        type=str, 
        default="./results", 
        help="Root directory for outputs"
    )
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Only run attention analysis on existing models"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        help="Directory with pre-trained models (for --analyze_only mode)"
    )
    
    args = parser.parse_args()
    
    if not args.analyze_only:
        # Run full training and analysis pipeline
        train_and_analyze_with_attention(args.raw_data_dir, args.output_root_dir)
    else:
        if not args.models_dir:
            print("Error: --models_dir required for --analyze_only mode")
        else:
            analyze_existing_models(args.models_dir, args.raw_data_dir, args.output_root_dir)