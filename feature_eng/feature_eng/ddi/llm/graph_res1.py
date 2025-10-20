import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Create results directory if it doesn't exist
os.makedirs('../results', exist_ok=True)

# Set style for better-looking plots optimized for academic papers
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Set global font sizes for academic paper (single column)
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 24,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1,
    'lines.linewidth': 2
})

# Data from Table 3 (Zero-shot performance)
zero_shot_data = {
    'Model': ['Claude3.5 Sonnet', 'Gemini 1.5', 'GPT-4o', 'Deepseek Qwen1.5B', 'LLaMa-3.2 3B', 
              'Qwen2.5 3B', 'Phi-3.5 2.7B', 'Granite 3.1 8B', 'LLaMa-3.1 8B', 'Gemma2 9B',
              'Falcon 3 10B', 'Mistral-Nemo 12B', 'Qwen 2.5 14B', 'Gemma2 27B', 'Aya-Expanse 32B',
              'Qwen 2.5 32B', 'LLaMa-3.3 70B', 'Qwen2 72B'],
    'Accuracy': [0.7358, 0.7220, 0.6459, 0.4807, 0.5009, 0.5037, 0.5358, 0.4734, 0.5294, 0.6376,
                 0.5853, 0.6991, 0.6569, 0.7303, 0.5917, 0.5982, 0.6477, 0.7119],
    'F1': [0.6720, 0.6807, 0.6178, 0.3494, 0.6667, 0.6629, 0.6452, 0.1087, 0.3498, 0.4942,
           0.3372, 0.6713, 0.5000, 0.6755, 0.6716, 0.4847, 0.6957, 0.6789],
    'Sensitivity': [0.5413, 0.5927, 0.5725, 0.2789, 0.9982, 0.9761, 0.8440, 0.0642, 0.2532, 0.3541,
                   0.2110, 0.6147, 0.3431, 0.5615, 0.8349, 0.3780, 0.8055, 0.6092],
    'Model_Short': ['Claude', 'Gemini', 'GPT-4o', 'DeepSeek', 'LLaMa-3.2', 
                   'Qwen-3B', 'Phi-3.5', 'Granite', 'LLaMa-3.1', 'Gemma-9B',
                   'Falcon3', 'Mistral', 'Qwen-14B', 'Gemma-27B', 'Aya-32B',
                   'Qwen-32B', 'LLaMa-70B', 'Qwen-72B']
}

# Data from Table 4 (Random Few-shot performance)
few_shot_random_data = {
    'Model': ['Claude3.5 Sonnet', 'Gemini 1.5', 'GPT-4o', 'Deepseek Qwen1.5B', 'LLaMa-3.2 3B', 
              'Qwen2.5 3B', 'Phi-3.5 2.7B', 'Granite 3.1 8B', 'LLaMa-3.1 8B', 'Gemma2 9B',
              'Falcon 3 10B', 'Mistral-Nemo 12B', 'Qwen 2.5 14B', 'Gemma2 27B', 'Aya-Expanse 32B',
              'Qwen 2.5 32B', 'LLaMa-3.3 70B', 'Qwen2 72B'],
    'Accuracy': [0.8211, 0.7394, 0.7477, 0.5156, 0.5761, 0.5202, 0.6560, 0.6248, 0.5028, 0.6321,
                 0.7128, 0.5229, 0.6807, 0.7450, 0.6771, 0.6963, 0.6413, 0.7018],
    'F1': [0.8094, 0.6872, 0.7403, 0.4454, 0.6883, 0.6561, 0.6934, 0.6477, 0.6671, 0.7041,
           0.6688, 0.6754, 0.5777, 0.7322, 0.6239, 0.6165, 0.7102, 0.6884],
    'Sensitivity': [0.7596, 0.5725, 0.7193, 0.3890, 0.9358, 0.9156, 0.7780, 0.6899, 0.9963, 0.8752,
                   0.5798, 0.9927, 0.4367, 0.6972, 0.5358, 0.4881, 0.8789, 0.6587],
    'Model_Short': ['Claude', 'Gemini', 'GPT-4o', 'DeepSeek', 'LLaMa-3.2', 
                   'Qwen-3B', 'Phi-3.5', 'Granite', 'LLaMa-3.1', 'Gemma-9B',
                   'Falcon3', 'Mistral', 'Qwen-14B', 'Gemma-27B', 'Aya-32B',
                   'Qwen-32B', 'LLaMa-70B', 'Qwen-72B']
}

# Data from Table 5 (Similarity-based Few-shot performance) 
few_shot_similarity_data = {
    'Model': ['Claude3.5 Sonnet', 'Gemini 1.5', 'GPT-4o', 'Deepseek Qwen1.5B', 'LLaMa-3.2 3B', 
              'Qwen2.5 3B', 'Phi-3.5 2.7B', 'Granite 3.1 8B', 'LLaMa-3.1 8B', 'Gemma2 9B',
              'Falcon 3 10B', 'Mistral-Nemo 12B', 'Qwen 2.5 14B', 'Gemma2 27B', 'Aya-Expanse 32B',
              'Qwen 2.5 32B', 'LLaMa-3.3 70B', 'Qwen2 72B'],
    'Accuracy': [0.8376, 0.7257, 0.7917, 0.5046, 0.5431, 0.6394, 0.5789, 0.5009, 0.6881, 0.7899,
                 0.6321, 0.7459, 0.7578, 0.7303, 0.7046, 0.5330, 0.7349, 0.6789],
    'F1': [0.8384, 0.6652, 0.8014, 0.4716, 0.6816, 0.6562, 0.6770, 0.7122, 0.6646, 0.7393,
           0.7975, 0.7202, 0.7303, 0.7556, 0.7625, 0.6414, 0.6789, 0.7629],
    'Sensitivity': [0.8345, 0.8422, 0.8534, 0.5450, 0.8404, 0.4422, 0.9780, 0.6881, 0.8826, 0.9560,
                   0.9890, 0.8844, 0.8275, 0.9468, 0.6881, 0.7486, 0.8661, 0.5284],
    'Model_Short': ['Claude', 'Gemini', 'GPT-4o', 'DeepSeek', 'LLaMa-3.2', 
                   'Qwen-3B', 'Phi-3.5', 'Granite', 'LLaMa-3.1', 'Gemma-9B',
                   'Falcon3', 'Mistral', 'Qwen-14B', 'Gemma-27B', 'Aya-32B',
                   'Qwen-32B', 'LLaMa-70B', 'Qwen-72B']
}

# Data from Table 6 (Fine-tuned performance)
finetuned_data = {
    'Model': ['GPT-4o', 'Deepseek Qwen1.5B', 'Phi-3.5 2.7B', 'Qwen2.5 3B', 'Gemma2 9B'],
    'Accuracy': [0.926, 0.895, 0.913, 0.832, 0.725],
    'F1': [0.926, 0.898, 0.917, 0.812, 0.725],
    'Sensitivity': [0.930, 0.919, 0.960, 0.969, 0.725],
    'Model_Short': ['GPT-4o', 'DeepSeek', 'Phi-3.5', 'Qwen-3B', 'Gemma2']
}

# Data from Table 7 (Sensitivity comparison across external datasets)
sensitivity_comparison = {
    'Dataset': ['CredibleMeds', 'HEP', 'HIV', 'Corpus 2011', 'Corpus 2013', 'NLM Corpus',
                'PK Corpus', 'OSCAR', 'WorldVista', 'French Ref.', 'KEGG', 'NDF-RT', 'Onc Non-Int.'],
    'MSDAFL': [1.000, 0.925, 0.846, 0.875, 0.900, 0.889, 1.000, 0.861, 0.903, 0.917, 0.904, 0.975, 0.879],
    'L2 (Repl.)': [1.000, 0.971, 0.920, 0.938, 0.905, 0.889, 1.000, 0.866, 0.907, 0.928, 0.891, 0.975, 0.835],
    'Phi3.5 3B': [1.000, 1.000, 1.000, 0.938, 1.000, 0.889, 1.000, 0.942, 0.986, 0.972, 0.990, 1.000, 1.000],
    'Qwen2.5 3B': [1.000, 1.000, 1.000, 0.938, 0.997, 0.889, 1.000, 0.936, 0.984, 0.986, 0.985, 0.966, 1.000]
}

def create_stacked_bar_extended(ax, models, accuracy, f1, sensitivity, title, n_models=10):
    """Helper function to create stacked bar charts with narrow bars and steep rotation"""
    # Take top n_models based on average performance
    avg_performance = [(accuracy[i] + f1[i] + sensitivity[i])/3 for i in range(len(models))]
    sorted_indices = sorted(range(len(avg_performance)), key=lambda i: avg_performance[i], reverse=True)
    top_indices = sorted_indices[:n_models]
    
    selected_models = [models[i] for i in top_indices]
    selected_accuracy = [accuracy[i] for i in top_indices]
    selected_f1 = [f1[i] for i in top_indices]
    selected_sensitivity = [sensitivity[i] for i in top_indices]
    
    x = np.arange(len(selected_models))
    width = 0.5  # Reduced width for narrow bars
    
    # Create stacked bars
    p1 = ax.bar(x, selected_accuracy, width, label='Accuracy', color='#3498db', alpha=0.9, edgecolor='black', linewidth=0.5)
    p2 = ax.bar(x, selected_f1, width, bottom=selected_accuracy, label='F1 Score', color='#2ecc71', alpha=0.9, edgecolor='black', linewidth=0.5)
    p3 = ax.bar(x, selected_sensitivity, width, bottom=np.array(selected_accuracy)+np.array(selected_f1), 
               label='Sensitivity', color='#e74c3c', alpha=0.9, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Performance Scores (Stacked)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xticks(x)
    # Increased rotation to 70 degrees and adjusted alignment
    ax.set_xticklabels(selected_models, rotation=70, ha='right')
    
    # Place legend at bottom, horizontal, centered
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=3, framealpha=0.9)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 2.8)
    
    # Add value labels on bars with conditional display for readability
    for i, (acc, f1_val, sens) in enumerate(zip(selected_accuracy, selected_f1, selected_sensitivity)):
        if acc > 0.15:  # Only show label if segment is large enough
            ax.text(i, acc/2, f'{acc:.2f}', ha='center', va='center', 
                   fontweight='bold', fontsize=16, color='white')
        if f1_val > 0.15:
            ax.text(i, acc + f1_val/2, f'{f1_val:.2f}', ha='center', va='center', 
                   fontweight='bold', fontsize=16, color='white')
        if sens > 0.15:
            ax.text(i, acc + f1_val + sens/2, f'{sens:.2f}', ha='center', va='center', 
                   fontweight='bold', fontsize=16, color='white')

# Create Figure 1: Zero-shot Performance with more models
fig, ax = plt.subplots(figsize=(14, 11))  # Increased width and height for more models and legend space

df_zero = pd.DataFrame(zero_shot_data)

create_stacked_bar_extended(ax, df_zero['Model_Short'], df_zero['Accuracy'], 
                          df_zero['F1'], df_zero['Sensitivity'],
                          'Zero-shot Performance (Top 10 Models)', n_models=10)

plt.tight_layout()
plt.savefig('../results/zero_shot_stacked.png', dpi=300, bbox_inches='tight')
plt.show()

# Create Figure 2: Few-shot Random Performance with more models
fig, ax = plt.subplots(figsize=(14, 11))

df_few_random = pd.DataFrame(few_shot_random_data)

create_stacked_bar_extended(ax, df_few_random['Model_Short'], df_few_random['Accuracy'], 
                          df_few_random['F1'], df_few_random['Sensitivity'],
                          'Few-shot Random (Top 10 Models)', n_models=10)

plt.tight_layout()
plt.savefig('../results/few_shot_random_stacked.png', dpi=300, bbox_inches='tight')
plt.show()

# Create Figure 3: Few-shot Similarity-based Performance with more models
fig, ax = plt.subplots(figsize=(14, 11))

df_few_similarity = pd.DataFrame(few_shot_similarity_data)

create_stacked_bar_extended(ax, df_few_similarity['Model_Short'], df_few_similarity['Accuracy'], 
                          df_few_similarity['F1'], df_few_similarity['Sensitivity'],
                          'Few-shot Similarity-based (Top 10 Models)', n_models=10)

plt.tight_layout()
plt.savefig('../results/few_shot_similarity_stacked.png', dpi=300, bbox_inches='tight')
plt.show()

# Create Figure 4: Few-shot Comparison (Random vs Similarity-based) - More models
fig, axes = plt.subplots(1, 2, figsize=(20, 11))  # Increased width significantly

# Select top 8 models based on average performance across both approaches
df_random = pd.DataFrame(few_shot_random_data)
df_similarity = pd.DataFrame(few_shot_similarity_data)

df_random['Avg_Performance'] = (df_random['Accuracy'] + df_random['F1'] + df_random['Sensitivity']) / 3
df_similarity['Avg_Performance'] = (df_similarity['Accuracy'] + df_similarity['F1'] + df_similarity['Sensitivity']) / 3

# Get models that appear in top performers for both approaches
top_random = set(df_random.nlargest(10, 'Avg_Performance')['Model_Short'])
top_similarity = set(df_similarity.nlargest(10, 'Avg_Performance')['Model_Short'])
common_top_models = list(top_random.intersection(top_similarity))[:8]

# Filter and sort data for common models
df_random_filtered = df_random[df_random['Model_Short'].isin(common_top_models)].copy()
df_similarity_filtered = df_similarity[df_similarity['Model_Short'].isin(common_top_models)].copy()

df_random_filtered = df_random_filtered.sort_values('Avg_Performance', ascending=False).reset_index(drop=True)
df_similarity_filtered = df_similarity_filtered.sort_values('Avg_Performance', ascending=False).reset_index(drop=True)

# Subplot 1: Random
x = np.arange(len(df_random_filtered))
width = 0.5

p1 = axes[0].bar(x, df_random_filtered['Accuracy'], width, label='Accuracy', 
                color='#3498db', alpha=0.9, edgecolor='black', linewidth=0.5)
p2 = axes[0].bar(x, df_random_filtered['F1'], width, bottom=df_random_filtered['Accuracy'], 
                label='F1 Score', color='#2ecc71', alpha=0.9, edgecolor='black', linewidth=0.5)
p3 = axes[0].bar(x, df_random_filtered['Sensitivity'], width, 
                bottom=df_random_filtered['Accuracy']+df_random_filtered['F1'], 
                label='Sensitivity', color='#e74c3c', alpha=0.9, edgecolor='black', linewidth=0.5)

axes[0].set_xlabel('Models', fontweight='bold')
axes[0].set_ylabel('Performance Scores', fontweight='bold')
axes[0].set_title('Few-shot Random', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(df_random_filtered['Model_Short'], rotation=70, ha='right')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=3, framealpha=0.9)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim(0, 2.8)

# Subplot 2: Similarity-based  
p1 = axes[1].bar(x, df_similarity_filtered['Accuracy'], width, label='Accuracy', 
                color='#3498db', alpha=0.9, edgecolor='black', linewidth=0.5)
p2 = axes[1].bar(x, df_similarity_filtered['F1'], width, bottom=df_similarity_filtered['Accuracy'], 
                label='F1 Score', color='#2ecc71', alpha=0.9, edgecolor='black', linewidth=0.5)
p3 = axes[1].bar(x, df_similarity_filtered['Sensitivity'], width, 
                bottom=df_similarity_filtered['Accuracy']+df_similarity_filtered['F1'], 
                label='Sensitivity', color='#e74c3c', alpha=0.9, edgecolor='black', linewidth=0.5)

axes[1].set_xlabel('Models', fontweight='bold')
axes[1].set_ylabel('Performance Scores', fontweight='bold')
axes[1].set_title('Few-shot Similarity-based', fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(df_similarity_filtered['Model_Short'], rotation=70, ha='right')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim(0, 2.8)

plt.tight_layout()
plt.savefig('../results/few_shot_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Create Figure 5: Fine-tuned Performance (unchanged as it already shows all models)
fig, ax = plt.subplots(figsize=(10, 11))

df_fine = pd.DataFrame(finetuned_data)

x = np.arange(len(df_fine))
width = 0.5

p1 = ax.bar(x, df_fine['Accuracy'], width, label='Accuracy', 
           color='#3498db', alpha=0.9, edgecolor='black', linewidth=0.5)
p2 = ax.bar(x, df_fine['F1'], width, bottom=df_fine['Accuracy'], 
           label='F1 Score', color='#2ecc71', alpha=0.9, edgecolor='black', linewidth=0.5)
p3 = ax.bar(x, df_fine['Sensitivity'], width, 
           bottom=df_fine['Accuracy']+df_fine['F1'], 
           label='Sensitivity', color='#e74c3c', alpha=0.9, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Models', fontweight='bold')
ax.set_ylabel('Performance Scores (Stacked)', fontweight='bold')
ax.set_title('Fine-tuned Performance', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(df_fine['Model_Short'], rotation=70, ha='right')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=3, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 2.8)

# Add value labels
for i, (acc, f1_val, sens) in enumerate(zip(df_fine['Accuracy'], df_fine['F1'], df_fine['Sensitivity'])):
    if acc > 0.1:
        ax.text(i, acc/2, f'{acc:.2f}', ha='center', va='center', 
               fontweight='bold', fontsize=16, color='white')
    if f1_val > 0.1:
        ax.text(i, acc + f1_val/2, f'{f1_val:.2f}', ha='center', va='center', 
               fontweight='bold', fontsize=16, color='white')
    if sens > 0.1:
        ax.text(i, acc + f1_val + sens/2, f'{sens:.2f}', ha='center', va='center', 
               fontweight='bold', fontsize=16, color='white')

plt.tight_layout()
plt.savefig('../results/fine_tuned_stacked.png', dpi=300, bbox_inches='tight')
plt.show()

# Create Figure 6: Sensitivity Heatmap (unchanged)
fig, ax = plt.subplots(figsize=(10, 8))

df_sensitivity = pd.DataFrame(sensitivity_comparison)
df_heatmap = df_sensitivity.set_index('Dataset')

sns.heatmap(df_heatmap, annot=True, cmap='RdYlGn', center=0.9, 
            fmt='.2f', cbar_kws={'label': 'Sensitivity'}, ax=ax,
            linewidths=1, square=False, annot_kws={'size': 14})

ax.set_title('Sensitivity on External Datasets', fontweight='bold', pad=15)
ax.set_xlabel('Models', fontweight='bold')
ax.set_ylabel('Datasets', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/sensitivity_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("Grafici estesi con più modelli (barre più strette, rotazione 70°) salvati in 'results/':")
print("1. zero_shot_stacked_extended.png (Top 10)")
print("2. few_shot_random_stacked_extended.png (Top 10)")
print("3. few_shot_similarity_stacked_extended.png (Top 10)")
print("4. few_shot_comparison_extended.png (Top 8 comuni)")
print("5. fine_tuned_stacked_extended.png (Tutti i 5 modelli)")
print("6. sensitivity_heatmap_extended.png")