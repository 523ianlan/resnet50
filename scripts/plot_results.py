import os
import pandas as pd
import matplotlib.pyplot as plt

output_dir = './fisher_comparison_results'
csv_path = os.path.join(output_dir, 'fisher_methods_comparison.csv')

def plot_saved_results():
    if not os.path.exists(csv_path):
        print("CSV not found!")
        return
        
    df = pd.read_csv(csv_path)
    
    # Summarize charts by Method and Batches
    summary = df.groupby(['Batches', 'Method'])[['Kendall_Tau', 'Spearman', 'Top30_Overlap_Pct']].mean().reset_index()
    summary.to_csv(os.path.join(output_dir, 'fisher_methods_summary.csv'), index=False)
    
    print("\n=== SUMMARY ACROSS ALL LAYERS ===")
    print(summary)
    
    # Draw summary comparison chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['Kendall_Tau', 'Spearman', 'Top30_Overlap_Pct']
    titles = ['Avg Kendall Tau vs Corrected Diag', 
              'Avg Spearman vs Corrected Diag', 
              'Avg Top-30% Overlap vs Corrected Diag']
              
    for i, metric in enumerate(metrics):
        pivot_df = summary.pivot(index='Batches', columns='Method', values=metric)
        pivot_df.plot.bar(ax=axes[i], rot=0)
        axes[i].set_title(titles[i])
        axes[i].set_ylabel(metric)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_metrics_comparison.png'))
    plt.close()
    
    print(f"\nExperiment Plotting Complete! Results saved to '{output_dir}'.")

if __name__ == '__main__':
    plot_saved_results()
