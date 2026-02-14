"""
Visualize training results and model performance.
Run this after training to generate comprehensive analysis plots.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def plot_training_history(history_path: Path, output_path: Path):
    """Plot training and validation curves."""
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training history plot saved to {output_path}")
    plt.close()


def plot_final_metrics(metrics_path: Path, output_path: Path):
    """Plot final test metrics as bar chart."""
    
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    metric_names = ['Test Accuracy', 'Test F1-Score (Macro)']
    values = [metrics['test_accuracy'], metrics['test_f1_macro']]
    colors = ['#2ecc71', '#3498db']
    
    bars = ax.bar(metric_names, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{value:.4f}',
            ha='center', va='bottom',
            fontsize=14, fontweight='bold'
        )
    
    ax.set_ylim([0, 1.0])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Final Test Performance', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Metrics plot saved to {output_path}")
    plt.close()


def generate_summary_report(output_dir: Path):
    """Generate a text summary report."""
    
    history_path = output_dir / 'training_history.json'
    metrics_path = output_dir / 'test_metrics.json'
    
    if not history_path.exists() or not metrics_path.exists():
        print("❌ Missing required files for summary report")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    # Find best epoch
    best_val_acc = max(history['val_accuracy'])
    best_epoch = history['val_accuracy'].index(best_val_acc) + 1
    total_epochs = len(history['loss'])
    
    report = []
    report.append("="*70)
    report.append("TRAINING SUMMARY REPORT")
    report.append("="*70)
    report.append("")
    report.append("TRAINING DETAILS")
    report.append("-"*70)
    report.append(f"Total epochs trained:        {total_epochs}")
    report.append(f"Best validation accuracy:    {best_val_acc:.4f} (epoch {best_epoch})")
    report.append(f"Final train accuracy:        {history['accuracy'][-1]:.4f}")
    report.append(f"Final validation accuracy:   {history['val_accuracy'][-1]:.4f}")
    report.append(f"Final train loss:            {history['loss'][-1]:.4f}")
    report.append(f"Final validation loss:       {history['val_loss'][-1]:.4f}")
    report.append("")
    report.append("TEST SET PERFORMANCE")
    report.append("-"*70)
    report.append(f"Test accuracy:               {metrics['test_accuracy']:.4f}")
    report.append(f"Test F1-score (macro):       {metrics['test_f1_macro']:.4f}")
    report.append("")
    report.append("CLASSES")
    report.append("-"*70)
    report.append(f"Number of classes:           {len(metrics['class_names'])}")
    report.append("Class names:")
    for i, class_name in enumerate(metrics['class_names'], 1):
        report.append(f"  {i:2d}. {class_name}")
    report.append("")
    report.append("="*70)
    
    report_text = "\n".join(report)
    
    # Print to console
    print("\n" + report_text)
    
    # Save to file
    report_path = output_dir / 'summary_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n✓ Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize training results'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Directory containing training outputs'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return
    
    print("="*70)
    print("VISUALIZING TRAINING RESULTS")
    print("="*70)
    print()
    
    # Check for required files
    required_files = [
        'training_history.json',
        'test_metrics.json',
        'confusion_matrix.png'
    ]
    
    missing = []
    for filename in required_files:
        filepath = output_dir / filename
        if not filepath.exists():
            missing.append(filename)
    
    if missing:
        print(f"❌ Missing files: {', '.join(missing)}")
        print("Please run training first.")
        return
    
    # Generate plots
    print("Generating visualizations...")
    print()
    
    # Training history
    plot_training_history(
        output_dir / 'training_history.json',
        output_dir / 'training_curves.png'
    )
    
    # Metrics
    plot_final_metrics(
        output_dir / 'test_metrics.json',
        output_dir / 'test_metrics_plot.png'
    )
    
    # Summary report
    generate_summary_report(output_dir)
    
    print()
    print("="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nGenerated files in {output_dir}:")
    print("  - training_curves.png")
    print("  - test_metrics_plot.png")
    print("  - summary_report.txt")
    print("  - confusion_matrix.png (already exists)")


if __name__ == '__main__':
    main()
