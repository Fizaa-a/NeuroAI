import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_comparison_charts():
    # Data based on our experiments
    models = [
        'Initial Model (Colab)', 
        'Improved CNN\n(Subject-Independent)', 
        'Hybrid Model\n(Time+Frequency)', 
        'EEGNet\n(Specialized)', 
        'Normalized RF\n(Baseline-Corrected)'
    ]
    
    # Accuracy values (%)
    # Note: Initial model was 74% but with segment-leakage. 
    # Real-world performance on new subjects is lower for leaky models.
    accuracies = [74.0, 81.94, 72.99, 75.2, 82.01]
    
    # Reliability Score (1-10) - Qualitative measure of how well it works on new people
    reliability = [3, 8, 7, 8, 10]
    
    # Colors
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0']
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Accuracy Comparison
    bars = ax1.bar(models, accuracies, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_ylim(40, 90)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 2: Reliability & Generalization Score
    # (Shows how much we trust the model on unseen subjects)
    ax2.scatter(models, reliability, s=500, c=colors, edgecolor='black', alpha=0.8)
    ax2.plot(models, reliability, linestyle='--', color='gray', alpha=0.5, zorder=0)
    ax2.set_ylim(0, 11)
    ax2.set_ylabel('Generalization Score (1-10)', fontsize=12, fontweight='bold')
    ax2.set_title('Robustness on New Subjects', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Aesthetic fixes
    for ax in [ax1, ax2]:
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison chart saved as 'performance_comparison.png'")

    # Generate Markdown Table
    print("\n--- PERFORMANCE SUMMARY TABLE ---")
    print("| Model | Accuracy | Gen. Strategy | Best Use Case |")
    print("| :--- | :--- | :--- | :--- |")
    print(f"| Initial Model | 74.00% | None (Segment Leakage) | Baseline for comparison |")
    print(f"| Improved CNN | 81.94% | Subject-Independent Split | Fast inference, real-time |")
    print(f"| Hybrid Model | 72.99% | Multi-modal fusion | Feature-rich environments |")
    print(f"| EEGNet | 75.20% | Depthwise Convolutions | Mobile/Low-power devices |")
    print(f"| **Normalized RF** | **82.01%** | **Subject Baselines** | **Clinical/High Precision** |")

if __name__ == "__main__":
    generate_comparison_charts()
