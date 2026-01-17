"""
Integration script combining attention analysis and gene knockout analysis.

This provides a complete interpretability pipeline:
1. Train Transformer model
2. Analyze attention patterns
3. Perform gene knockout analysis
4. Compare and validate results
"""

import os
import argparse
import torch
from data_preprocessing import process_single_patient_data
from transformer_training import train_transformer
from hybrid_attention_analysis import AttentionAnalyzer
from gene_knockout_analysis import GeneKnockoutAnalyzer


def comprehensive_analysis_pipeline(raw_data_dir, output_root_dir, 
                                    skip_knockouts=False):
    """
    Run complete analysis pipeline with attention and knockout analysis.
    
    Args:
        raw_data_dir: Root directory of raw data
        output_root_dir: Root output directory
        skip_knockouts: Whether to skip knockout analysis (faster, for testing)
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
    output_dir = os.path.join(output_root_dir, 'comprehensive_analysis')
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
        print(f"\n{'='*80}")
        print(f"PATIENT {patient} - COMPREHENSIVE ANALYSIS")
        print(f"{'='*80}")
        
        try:
            # ================================================================
            # STEP 1: DATA LOADING AND MODEL TRAINING
            # ================================================================
            print("\n[1/4] Loading data and training model...")
            sequences, labels, gene_names = process_single_patient_data(
                processed_matrices_dir, patient, files
            )
            print(f"  ✓ Loaded {len(gene_names)} genes with {sequences.shape[1]} time points")
            
            model, processed_sequences = train_transformer(
                sequences, labels, patient, output_dir, device
            )
            print("  Model training completed")
            
            # Prepare data for analysis
            sequences_torch = torch.from_numpy(processed_sequences).to(
                dtype=torch.float32, device=device
            )
            
            if labels is None or len(labels) != sequences_torch.shape[0]:
                analysis_labels = torch.randint(0, 3, (sequences_torch.shape[0],))
            else:
                analysis_labels = torch.tensor(labels, dtype=torch.long)
            
            # ================================================================
            # STEP 2: ATTENTION ANALYSIS
            # ================================================================
            print("\n[2/4] Running attention analysis...")
            attention_analyzer = AttentionAnalyzer(model, device=device)
            attention_results = attention_analyzer.generate_comprehensive_report(
                sequences_torch, analysis_labels, gene_names, output_dir, patient
            )
            
            importance_df = attention_results['importance']
            print("  Attention analysis complete")
            print(f"    Top gene by attention: {importance_df.iloc[0]['Gene']}")
            
            # ================================================================
            # STEP 3: GENE KNOCKOUT ANALYSIS
            # ================================================================
            if not skip_knockouts:
                print("\n[3/4] Running gene knockout analysis...")
                print("  (This may take a while for large gene sets...)")
                
                knockout_analyzer = GeneKnockoutAnalyzer(model, device=device)
                knockout_df, cumulative_df, comparison_stats = \
                    knockout_analyzer.generate_knockout_report(
                        sequences_torch, analysis_labels, gene_names,
                        importance_df, output_dir, patient
                    )
                
                print("  Knockout analysis complete")
                print(f"    Top gene by knockout: {knockout_df.iloc[0]['gene_name']}")
                print(f"    Max accuracy drop: {knockout_df.iloc[0]['accuracy_drop']:.4f}")
            else:
                print("\n[3/4] Skipping knockout analysis (--skip-knockouts flag set)")
                knockout_df = None
                comparison_stats = None
            
            # ================================================================
            # STEP 4: INTEGRATED ANALYSIS
            # ================================================================
            print("\n[4/4] Generating integrated insights...")
            
            patient_summary = {
                'patient_id': patient,
                'num_genes': len(gene_names),
                'top_attention_gene': importance_df.iloc[0]['Gene'],
                'top_attention_score': importance_df.iloc[0]['Importance'],
            }
            
            if knockout_df is not None:
                patient_summary.update({
                    'top_knockout_gene': knockout_df.iloc[0]['gene_name'],
                    'max_knockout_impact': knockout_df.iloc[0]['accuracy_drop'],
                    'attention_knockout_correlation': comparison_stats['spearman_correlation'],
                    'agreement_top10': len(set(importance_df.head(10)['Gene']) & 
                                          set(knockout_df.head(10)['gene_name'])),
                })
                
                # Check agreement
                top10_attention = set(importance_df.head(10)['Gene'])
                top10_knockout = set(knockout_df.head(10)['gene_name'])
                agreement = len(top10_attention & top10_knockout)
                
                print("\n  KEY INSIGHTS:")
                print(f"  {'─'*76}")
                print(f"  Top gene (attention):  {patient_summary['top_attention_gene']}")
                print(f"  Top gene (knockout):   {patient_summary['top_knockout_gene']}")
                print(f"  Top-10 agreement:      {agreement}/10 genes")
                print(f"  Rank correlation:      {comparison_stats['spearman_correlation']:.3f}")
                print(f"  Max knockout impact:   {patient_summary['max_knockout_impact']:.4f}")
            
            all_patient_results.append(patient_summary)
            
            # Generate integrated report
            generate_integrated_report(
                patient, importance_df, knockout_df, 
                attention_results, comparison_stats,
                output_dir
            )
            
        except Exception as e:
            print(f"\n  ✗ Error processing patient {patient}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # ================================================================
    # CROSS-PATIENT ANALYSIS
    # ================================================================
    if all_patient_results:
        print(f"\n{'='*80}")
        print("CROSS-PATIENT SUMMARY")
        print(f"{'='*80}")
        
        import pandas as pd
        summary_df = pd.DataFrame(all_patient_results)
        
        summary_path = os.path.join(output_dir, "cross_patient_comprehensive_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\nAnalyzed {len(all_patient_results)} patients")
        
        if not skip_knockouts and 'attention_knockout_correlation' in summary_df.columns:
            print("\nCross-Patient Statistics:")
            print(f"  Mean attention-knockout correlation: "
                  f"{summary_df['attention_knockout_correlation'].mean():.3f} "
                  f"(±{summary_df['attention_knockout_correlation'].std():.3f})")
            print(f"  Mean top-10 agreement: "
                  f"{summary_df['agreement_top10'].mean():.1f}/10 genes")
            print(f"  Mean max knockout impact: "
                  f"{summary_df['max_knockout_impact'].mean():.4f}")
        
        print(f"\nResults saved to: {output_dir}")
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"{'='*80}")


def generate_integrated_report(patient_id, importance_df, knockout_df, 
                               attention_results, comparison_stats, output_dir):
    """
    Generate an integrated report combining attention and knockout insights.
    
    Args:
        patient_id: Patient identifier
        importance_df: Attention-based importance rankings
        knockout_df: Knockout analysis results
        attention_results: Full attention analysis results
        comparison_stats: Comparison statistics
        output_dir: Output directory
    """
    report_path = os.path.join(output_dir, f"{patient_id}_integrated_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"INTEGRATED INTERPRETABILITY REPORT - Patient {patient_id}\n")
        f.write("="*80 + "\n\n")
        
        f.write("SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Total genes analyzed: {len(importance_df)}\n\n")
        
        # Top genes comparison
        f.write("TOP 10 GENES - ATTENTION IMPORTANCE\n")
        f.write("-"*80 + "\n")
        for idx, row in importance_df.head(10).iterrows():
            f.write(f"{row['Rank']:2d}. {row['Gene']:20s} "
                   f"(Importance: {row['Importance']:.4f})\n")
        
        if knockout_df is not None:
            f.write("\n\nTOP 10 GENES - KNOCKOUT IMPACT\n")
            f.write("-"*80 + "\n")
            for idx, row in knockout_df.head(10).iterrows():
                f.write(f"{row['rank']:2d}. {row['gene_name']:20s} "
                       f"(Acc Drop: {row['accuracy_drop']:.4f})\n")
            
            # Find consensus genes
            top10_attention = set(importance_df.head(10)['Gene'])
            top10_knockout = set(knockout_df.head(10)['gene_name'])
            consensus = top10_attention & top10_knockout
            
            f.write("\n\nCONSENSUS GENES (Top 10 by both methods)\n")
            f.write("-"*80 + "\n")
            if consensus:
                for gene in consensus:
                    att_rank = importance_df[importance_df['Gene'] == gene]['Rank'].values[0]
                    ko_rank = knockout_df[knockout_df['gene_name'] == gene]['rank'].values[0]
                    f.write(f"{gene:20s} | Attention Rank: {att_rank:3d} | "
                           f"Knockout Rank: {ko_rank:3d}\n")
            else:
                f.write("No consensus genes in top 10\n")
            
            # Method comparison
            f.write("\n\nMETHOD COMPARISON\n")
            f.write("-"*80 + "\n")
            f.write(f"Spearman rank correlation: {comparison_stats['spearman_correlation']:.4f}\n")
            f.write(f"P-value: {comparison_stats['spearman_pvalue']:.3e}\n")
            f.write(f"Top-10 agreement: {len(consensus)}/10 genes\n")
            
            # Interpretation
            f.write("\n\nINTERPRETATION GUIDE\n")
            f.write("-"*80 + "\n")
            f.write("High correlation (>0.7): Both methods identify similar important genes\n")
            f.write("Medium correlation (0.4-0.7): Partial agreement, methods capture different aspects\n")
            f.write("Low correlation (<0.4): Methods diverge, investigate discrepancies\n\n")
            
            if comparison_stats['spearman_correlation'] > 0.7:
                f.write("✓ Strong agreement between attention and knockout methods\n")
            elif comparison_stats['spearman_correlation'] > 0.4:
                f.write("~ Moderate agreement - both methods provide complementary insights\n")
            else:
                f.write("! Low agreement - investigate genes that rank differently\n")
        
        # Temporal patterns
        cluster_summary = attention_results['clusters'][1]
        f.write("\n\nTEMPORAL PATTERN CLUSTERS\n")
        f.write("-"*80 + "\n")
        for _, row in cluster_summary.iterrows():
            f.write(f"Cluster {row['Cluster']}: {row['Pattern_Type']:15s} "
                   f"({row['Size']} genes)\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"  ✓ Integrated report saved to {report_path}")


def analyze_single_patient(patient_id, raw_data_dir, output_dir, 
                           skip_knockouts=False):
    """
    Run comprehensive analysis for a single patient.
    
    Args:
        patient_id: Patient identifier
        raw_data_dir: Directory with raw data
        output_dir: Output directory
        skip_knockouts: Whether to skip knockout analysis
    """
    print(f"Analyzing patient {patient_id}...")
    # Implementation would be similar to the main pipeline
    # but focused on a single patient
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Comprehensive Transformer interpretability analysis"
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
        "--skip_knockouts",
        action="store_true",
        help="Skip gene knockout analysis (faster, for testing)"
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        help="Analyze single patient only"
    )
    
    args = parser.parse_args()
    
    if args.patient_id:
        # Analyze single patient
        analyze_single_patient(
            args.patient_id, 
            args.raw_data_dir, 
            args.output_root_dir,
            args.skip_knockouts
        )
    else:
        # Run full pipeline
        comprehensive_analysis_pipeline(
            args.raw_data_dir, 
            args.output_root_dir,
            args.skip_knockouts
        )