"""
Model Comparison and Visualization
Subsystem 5: Training and Model Comparison

This script compares performance between CNN from scratch and fine-tuned models,
generating comprehensive visualizations and comparative analysis.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 11


class ModelComparison:
    """Compare and visualize model performance"""
    
    def __init__(self, metrics_dir, plots_dir):
        """
        Initialize model comparison
        
        Args:
            metrics_dir: Directory containing metrics JSON files
            plots_dir: Directory to save comparison plots
        """
        self.metrics_dir = metrics_dir
        self.plots_dir = plots_dir
        self.models_data = {}
        
    def load_all_metrics(self):
        """Load metrics from all JSON files"""
        
        print("Loading metrics from all models...")
        
        json_files = glob.glob(os.path.join(self.metrics_dir, '*_metrics.json'))
        
        for json_file in json_files:
            model_name = os.path.basename(json_file).replace('_metrics.json', '')
            
            with open(json_file, 'r') as f:
                metrics = json.load(f)
                self.models_data[model_name] = metrics
                
            print(f"  ✓ Loaded: {model_name}")
        
        print(f"\nTotal models loaded: {len(self.models_data)}")
        
        return self.models_data
    
    def create_comparison_table(self):
        """Create comparison table of all models"""
        
        print("\nCreating comparison table...")
        
        comparison_data = []
        
        for model_name, metrics in self.models_data.items():
            if 'test' in metrics:
                test_metrics = metrics['test']
                
                row = {
                    'Model': model_name.upper().replace('_', ' '),
                    'Accuracy': test_metrics.get('accuracy', 0),
                    'Precision': test_metrics.get('precision', 0),
                    'Recall': test_metrics.get('recall', 0),
                    'AUC': test_metrics.get('auc', 0),
                    'Loss': test_metrics.get('loss', 0)
                }
                
                comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Accuracy', ascending=False)
        
        # Save to CSV
        csv_path = os.path.join(self.metrics_dir, 'models_comparison.csv')
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved comparison table: {csv_path}")
        
        # Print table
        print("\n" + "="*80)
        print("MODEL COMPARISON TABLE")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        return df
    
    def plot_metrics_comparison(self, df):
        """Plot bar chart comparing all metrics"""
        
        print("\nPlotting metrics comparison...")
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'AUC']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = sns.color_palette("husl", len(df))
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            bars = ax.barh(df['Model'], df[metric], color=colors)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f'{width:.4f}',
                       ha='left', va='center', fontweight='bold',
                       fontsize=10, color='black')
            
            ax.set_xlabel(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel('Model', fontsize=12, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1.0)
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'metrics_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: metrics_comparison.png")
    
    def plot_radar_chart(self, df):
        """Create radar chart for model comparison"""
        
        print("Creating radar chart...")
        
        categories = ['Accuracy', 'Precision', 'Recall', 'AUC']
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # Initialize plot
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        colors = sns.color_palette("husl", len(df))
        
        for idx, (_, row) in enumerate(df.iterrows()):
            values = [row[cat] for cat in categories]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'],
                   color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
        
        # Set y-axis limit
        ax.set_ylim(0, 1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        
        plt.title('Model Performance Radar Chart', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'radar_chart_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: radar_chart_comparison.png")
    
    def plot_accuracy_vs_parameters(self):
        """Plot accuracy vs model complexity (if available)"""
        
        print("Plotting accuracy vs complexity...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = []
        accuracies = []
        colors_list = []
        
        colors = sns.color_palette("husl", len(self.models_data))
        
        for idx, (model_name, metrics) in enumerate(self.models_data.items()):
            if 'test' in metrics:
                models.append(model_name.upper().replace('_', ' '))
                accuracies.append(metrics['test']['accuracy'])
                colors_list.append(colors[idx])
        
        # Create bar plot
        bars = ax.bar(range(len(models)), accuracies, color=colors_list,
                     edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
        ax.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'accuracy_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: accuracy_comparison.png")
    
    def plot_loss_comparison(self):
        """Plot loss comparison"""
        
        print("Plotting loss comparison...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = []
        losses = []
        colors_list = []
        
        colors = sns.color_palette("husl", len(self.models_data))
        
        for idx, (model_name, metrics) in enumerate(self.models_data.items()):
            if 'test' in metrics:
                models.append(model_name.upper().replace('_', ' '))
                losses.append(metrics['test']['loss'])
                colors_list.append(colors[idx])
        
        # Create bar plot
        bars = ax.bar(range(len(models)), losses, color=colors_list,
                     edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax.set_ylabel('Test Loss', fontsize=13, fontweight='bold')
        ax.set_title('Model Loss Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'loss_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: loss_comparison.png")
    
    def plot_precision_recall_comparison(self):
        """Plot precision vs recall scatter plot"""
        
        print("Plotting precision vs recall...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = sns.color_palette("husl", len(self.models_data))
        
        for idx, (model_name, metrics) in enumerate(self.models_data.items()):
            if 'test' in metrics:
                precision = metrics['test']['precision']
                recall = metrics['test']['recall']
                
                ax.scatter(recall, precision, s=500, alpha=0.6,
                         color=colors[idx], edgecolors='black', linewidth=2,
                         label=model_name.upper().replace('_', ' '))
                
                ax.text(recall, precision, 
                       model_name.upper().replace('_', ' '),
                       ha='center', va='center', fontweight='bold',
                       fontsize=9, color='white')
        
        ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
        ax.set_title('Precision vs Recall', fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize=10)
        
        # Add diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'precision_recall_scatter.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: precision_recall_scatter.png")
    
    def create_comprehensive_summary(self, df):
        """Create comprehensive summary visualization"""
        
        print("Creating comprehensive summary...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        colors = sns.color_palette("husl", len(df))
        
        # 1. Accuracy comparison (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.barh(df['Model'], df['Accuracy'], color=colors)
        ax1.set_xlabel('Accuracy', fontweight='bold')
        ax1.set_title('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, 1.0)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Precision comparison (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.barh(df['Model'], df['Precision'], color=colors)
        ax2.set_xlabel('Precision', fontweight='bold')
        ax2.set_title('Precision', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 1.0)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Recall comparison (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.barh(df['Model'], df['Recall'], color=colors)
        ax3.set_xlabel('Recall', fontweight='bold')
        ax3.set_title('Recall', fontsize=12, fontweight='bold')
        ax3.set_xlim(0, 1.0)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. AUC comparison (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.barh(df['Model'], df['AUC'], color=colors)
        ax4.set_xlabel('AUC', fontweight='bold')
        ax4.set_title('AUC Score', fontsize=12, fontweight='bold')
        ax4.set_xlim(0, 1.0)
        ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Loss comparison (middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.barh(df['Model'], df['Loss'], color=colors)
        ax5.set_xlabel('Loss', fontweight='bold')
        ax5.set_title('Test Loss', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        
        # 6. Overall scores heatmap (middle right + bottom right)
        ax6 = fig.add_subplot(gs[1:, 2])
        heatmap_data = df[['Accuracy', 'Precision', 'Recall', 'AUC']].T
        sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlGnBu',
                   xticklabels=df['Model'], yticklabels=['Accuracy', 'Precision', 'Recall', 'AUC'],
                   cbar_kws={'label': 'Score'}, ax=ax6)
        ax6.set_title('Metrics Heatmap', fontsize=12, fontweight='bold')
        ax6.set_xticklabels(df['Model'], rotation=45, ha='right')
        
        # 7. Summary statistics (bottom left + bottom middle)
        ax7 = fig.add_subplot(gs[2, :2])
        ax7.axis('off')
        
        # Create summary text
        summary_text = "MODEL COMPARISON SUMMARY\n" + "="*50 + "\n\n"
        
        best_accuracy = df.loc[df['Accuracy'].idxmax()]
        best_precision = df.loc[df['Precision'].idxmax()]
        best_recall = df.loc[df['Recall'].idxmax()]
        best_auc = df.loc[df['AUC'].idxmax()]
        
        summary_text += f"🏆 Best Accuracy:  {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})\n"
        summary_text += f"🏆 Best Precision: {best_precision['Model']} ({best_precision['Precision']:.4f})\n"
        summary_text += f"🏆 Best Recall:    {best_recall['Model']} ({best_recall['Recall']:.4f})\n"
        summary_text += f"🏆 Best AUC:       {best_auc['Model']} ({best_auc['AUC']:.4f})\n\n"
        
        summary_text += "AVERAGE SCORES:\n"
        summary_text += f"  Mean Accuracy:  {df['Accuracy'].mean():.4f} ± {df['Accuracy'].std():.4f}\n"
        summary_text += f"  Mean Precision: {df['Precision'].mean():.4f} ± {df['Precision'].std():.4f}\n"
        summary_text += f"  Mean Recall:    {df['Recall'].mean():.4f} ± {df['Recall'].std():.4f}\n"
        summary_text += f"  Mean AUC:       {df['AUC'].mean():.4f} ± {df['AUC'].std():.4f}\n"
        
        ax7.text(0.1, 0.5, summary_text, fontsize=11, fontfamily='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Comprehensive Model Comparison', fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig(os.path.join(self.plots_dir, 'comprehensive_summary.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: comprehensive_summary.png")
    
    def generate_all_comparisons(self):
        """Generate all comparison visualizations"""
        
        print("\n" + "="*60)
        print("Generating Model Comparisons")
        print("="*60 + "\n")
        
        # Load metrics
        self.load_all_metrics()
        
        if not self.models_data:
            print("⚠️  No metrics found. Please train models first.")
            return
        
        # Create comparison table
        df = self.create_comparison_table()
        
        # Generate all plots
        self.plot_metrics_comparison(df)
        self.plot_radar_chart(df)
        self.plot_accuracy_vs_parameters()
        self.plot_loss_comparison()
        self.plot_precision_recall_comparison()
        self.create_comprehensive_summary(df)
        
        print("\n" + "="*60)
        print("✓ All comparisons generated successfully!")
        print("="*60)
        print(f"\nPlots saved to: {self.plots_dir}")


def main():
    """Main comparison pipeline"""
    
    # Set paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    metrics_dir = os.path.join(base_dir, 'results', 'metrics')
    plots_dir = os.path.join(base_dir, 'results', 'plots')
    
    # Create comparison instance
    comparator = ModelComparison(metrics_dir, plots_dir)
    
    # Generate all comparisons
    comparator.generate_all_comparisons()


if __name__ == "__main__":
    main()
