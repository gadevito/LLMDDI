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
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 22,
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
    'Qwen2.5 3B': [1.000, 1.000, 1.000, 0.938, 0.997, 0.889, 1.000, 0.936, 0.984, 0.986, 0.985, 0.966, 1.000],
    'Gemma2 9B': [0.400, 0.725, 0.765, 0.906, 0.797, 0.889, 1.000, 0.696, 0.851, 0.808, 0.714, 0.773, 0.904],
    'GPT-4o': [1.000, 0.917, 0.982, 0.750, 0.905, 0.889, 1.000, 0.905, 0.841, 0.981, 0.962, 1.000, 0.993],
    'DS Qwen 1.5B': [1.000, 0.990, 0.971, 0.906, 0.905, 1.000, 1.000, 0.872, 0.970, 0.879, 0.956, 0.966, 0.969]
}

# Data from Table 8 (Accuracy comparison across external datasets)
accuracy_comparison = {
    'Dataset': ['CredibleMeds', 'HEP', 'HIV', 'Corpus 2011', 'Corpus 2013', 'NLM Corpus',
                'PK Corpus', 'OSCAR', 'WorldVista', 'French Ref.', 'KEGG', 'NDF-RT', 'Onc Non-Int.'],
    'MSDAFL': [1.000, 0.919, 0.873, 0.898, 0.891, 0.941, 1.000, 0.872, 0.895, 0.916, 0.904, 0.905, 0.889],
    'L2 (Repl.)': [1.000, 0.919, 0.903, 0.922, 0.899, 0.889, 1.000, 0.882, 0.905, 0.910, 0.889, 0.971, 0.852],
    'Phi3.5 3B': [1.000, 0.900, 0.924, 0.895, 0.927, 0.778, 0.500, 0.902, 0.924, 0.910, 0.922, 0.958, 0.919],
    'Qwen2.5 3B': [1.000, 0.924, 0.895, 0.859, 0.884, 0.667, 0.750, 0.873, 0.899, 0.893, 0.895, 0.908, 0.897],
    'Gemma2 9B': [0.700, 0.823, 0.846, 0.938, 0.872, 0.944, 0.750, 0.821, 0.892, 0.868, 0.824, 0.878, 0.907],
    'GPT-4o': [1.000, 0.908, 0.948, 0.797, 0.878, 0.833, 0.750, 0.913, 0.877, 0.947, 0.937, 0.983, 0.952],
    'DS Qwen 1.5B': [1.000, 0.923, 0.915, 0.844, 0.865, 0.889, 0.750, 0.874, 0.919, 0.869, 0.910, 0.950, 0.909]
}

# Data from Table 9 (F1 comparison across external datasets)
f1_comparison = {
    'Dataset': ['CredibleMeds', 'HEP', 'HIV', 'Corpus 2011', 'Corpus 2013', 'NLM Corpus',
                'PK Corpus', 'OSCAR', 'WorldVista', 'French Ref.', 'KEGG', 'NDF-RT', 'Onc Non-Int.'],
    'MSDAFL': [1.000, 0.923, 0.874, 0.903, 0.894, 0.941, 1.000, 0.876, 0.901, 0.920, 0.906, 0.913, 0.889],
    'L2 (Repl.)': [1.000, 0.923, 0.904, 0.923, 0.899, 0.889, 1.000, 0.880, 0.905, 0.912, 0.889, 0.971, 0.850],
    'Phi3.5 3B': [1.000, 0.909, 0.930, 0.905, 0.925, 0.800, 0.667, 0.905, 0.928, 0.915, 0.927, 0.960, 0.925],
    'Qwen2.5 3B': [1.000, 0.932, 0.907, 0.870, 0.896, 0.727, 0.800, 0.880, 0.907, 0.902, 0.903, 0.913, 0.907],
    'Gemma2 9B': [0.571, 0.804, 0.833, 0.935, 0.861, 0.941, 0.800, 0.795, 0.887, 0.859, 0.803, 0.864, 0.907],
    'GPT-4o': [1.000, 0.908, 0.950, 0.787, 0.882, 0.842, 0.800, 0.913, 0.873, 0.949, 0.939, 0.983, 0.954],
    'DS Qwen 1.5B': [1.000, 0.928, 0.919, 0.853, 0.870, 0.900, 0.800, 0.874, 0.923, 0.871, 0.914, 0.950, 0.914]
}

def create_stacked_bar(ax, models, accuracy, f1, sensitivity, title):
    """Helper function to create stacked bar charts with large fonts and bottom legend"""
    x = np.arange(len(models))
    width = 0.7
    
    # Create stacked bars
    p1 = ax.bar(x, accuracy, width, label='Accuracy', color='#3498db', alpha=0.9, edgecolor='black', linewidth=0.5)
    p2 = ax.bar(x, f1, width, bottom=accuracy, label='F1 Score', color='#2ecc71', alpha=0.9, edgecolor='black', linewidth=0.5)
    p3 = ax.bar(x, sensitivity, width, bottom=np.array(accuracy)+np.array(f1), 
               label='Sensitivity', color='#e74c3c', alpha=0.9, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Performance Scores (Stacked)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    
    # Place legend at bottom, horizontal, centered
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, framealpha=0.9)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 2.8)
    
    # Add value labels on bars with larger font
    for i, (acc, f1_val, sens) in enumerate(zip(accuracy, f1, sensitivity)):
        if acc > 0.1:  # Only show label if segment is large enough
            ax.text(i, acc/2, f'{acc:.2f}', ha='center', va='center', 
                   fontweight='bold', fontsize=16, color='white')
        if f1_val > 0.1:
            ax.text(i, acc + f1_val/2, f'{f1_val:.2f}', ha='center', va='center', 
                   fontweight='bold', fontsize=16, color='white')
        if sens > 0.1:
            ax.text(i, acc + f1_val + sens/2, f'{sens:.2f}', ha='center', va='center', 
                   fontweight='bold', fontsize=16, color='white')

# Create Figure 1: Zero-shot Performance with Stacked Bars (Top 6 for readability)
fig, ax = plt.subplots(figsize=(12, 10))  # Increased height for legend space

df_zero = pd.DataFrame(zero_shot_data)
df_zero['Avg_Performance'] = (df_zero['Accuracy'] + df_zero['F1'] + df_zero['Sensitivity']) / 3
df_zero_top = df_zero.nlargest(6, 'Avg_Performance').reset_index(drop=True)

create_stacked_bar(ax, df_zero_top['Model_Short'], df_zero_top['Accuracy'], 
                  df_zero_top['F1'], df_zero_top['Sensitivity'],
                  'Zero-shot Performance (Top 6 Models)')

plt.tight_layout()
plt.savefig('../results/zero_shot_stacked.png', dpi=300, bbox_inches='tight')
plt.show()

# Create Figure 2: Few-shot Random Performance (Top 6)
fig, ax = plt.subplots(figsize=(12, 10))

df_few_random = pd.DataFrame(few_shot_random_data)
df_few_random['Avg_Performance'] = (df_few_random['Accuracy'] + df_few_random['F1'] + df_few_random['Sensitivity']) / 3
df_few_random_top = df_few_random.nlargest(6, 'Avg_Performance').reset_index(drop=True)

create_stacked_bar(ax, df_few_random_top['Model_Short'], df_few_random_top['Accuracy'], 
                  df_few_random_top['F1'], df_few_random_top['Sensitivity'],
                  'Few-shot Random (Top 6 Models)')

plt.tight_layout()
plt.savefig('../results/few_shot_random_stacked.png', dpi=300, bbox_inches='tight')
plt.show()

# Create Figure 3: Few-shot Similarity-based Performance (Top 6)
fig, ax = plt.subplots(figsize=(12, 10))

df_few_similarity = pd.DataFrame(few_shot_similarity_data)
df_few_similarity['Avg_Performance'] = (df_few_similarity['Accuracy'] + df_few_similarity['F1'] + df_few_similarity['Sensitivity']) / 3
df_few_similarity_top = df_few_similarity.nlargest(6, 'Avg_Performance').reset_index(drop=True)

create_stacked_bar(ax, df_few_similarity_top['Model_Short'], df_few_similarity_top['Accuracy'], 
                  df_few_similarity_top['F1'], df_few_similarity_top['Sensitivity'],
                  'Few-shot Similarity-based (Top 6 Models)')

plt.tight_layout()
plt.savefig('../results/few_shot_similarity_stacked.png', dpi=300, bbox_inches='tight')
plt.show()

# Create Figure 4: Few-shot Comparison (Random vs Similarity-based) - Top 5 common models
fig, axes = plt.subplots(1, 2, figsize=(16, 10))  # Increased height

common_models = ['Claude', 'GPT-4o', 'Gemma-27B', 'Mistral', 'LLaMa-70B']

# Filter and sort data for common models
df_random_filtered = df_few_random[df_few_random['Model_Short'].isin(common_models)].copy()
df_similarity_filtered = df_few_similarity[df_few_similarity['Model_Short'].isin(common_models)].copy()

df_random_filtered = df_random_filtered.sort_values('Model_Short').reset_index(drop=True)
df_similarity_filtered = df_similarity_filtered.sort_values('Model_Short').reset_index(drop=True)

# Subplot 1: Random
x = np.arange(len(df_random_filtered))
width = 0.7

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
axes[0].set_xticklabels(df_random_filtered['Model_Short'], rotation=45, ha='right')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, framealpha=0.9)
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
axes[1].set_xticklabels(df_similarity_filtered['Model_Short'], rotation=45, ha='right')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim(0, 2.8)

plt.tight_layout()
plt.savefig('../results/few_shot_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Create Figure 5: Fine-tuned Performance
fig, ax = plt.subplots(figsize=(10, 10))

df_fine = pd.DataFrame(finetuned_data)

create_stacked_bar(ax, df_fine['Model_Short'], df_fine['Accuracy'], 
                  df_fine['F1'], df_fine['Sensitivity'],
                  'Fine-tuned Performance')

plt.tight_layout()
plt.savefig('../results/fine_tuned_stacked.png', dpi=300, bbox_inches='tight')
plt.show()

# Create Figure 6: Comprehensive Comparison - Only common models
fig, axes = plt.subplots(2, 2, figsize=(16, 14))  # Increased height
axes = axes.flatten()

common_models_all = ['GPT-4o', 'DeepSeek', 'Phi-3.5', 'Qwen-3B', 'Gemma2']

comparison_data = {
    'Zero-shot': {
        'Accuracy': [0.6459, 0.4807, 0.5358, 0.5037, 0.6376],
        'F1': [0.6178, 0.3494, 0.6452, 0.6629, 0.4942],
        'Sensitivity': [0.5725, 0.2789, 0.8440, 0.9761, 0.3541]
    },
    'Few-shot Random': {
        'Accuracy': [0.7477, 0.5156, 0.6560, 0.5202, 0.6321],
        'F1': [0.7403, 0.4454, 0.6934, 0.6561, 0.7041],
        'Sensitivity': [0.7193, 0.3890, 0.7780, 0.9156, 0.8752]
    },
    'Few-shot Similarity': {
        'Accuracy': [0.7917, 0.5046, 0.5789, 0.6394, 0.7899],
        'F1': [0.8014, 0.4716, 0.6770, 0.6562, 0.7393],
        'Sensitivity': [0.8534, 0.5450, 0.9780, 0.4422, 0.9560]
    },
    'Fine-tuned': {
        'Accuracy': [0.926, 0.895, 0.913, 0.832, 0.725],
        'F1': [0.926, 0.898, 0.917, 0.812, 0.725],
        'Sensitivity': [0.930, 0.919, 0.960, 0.969, 0.725]
    }
}

approaches = ['Zero-shot', 'Few-shot Random', 'Few-shot Similarity', 'Fine-tuned']
bar_width = 0.7

for idx, approach in enumerate(approaches):
    x = np.arange(len(common_models_all))
    
    accuracy = comparison_data[approach]['Accuracy']
    f1 = comparison_data[approach]['F1']
    sensitivity = comparison_data[approach]['Sensitivity']
    
    p1 = axes[idx].bar(x, accuracy, bar_width, label='Accuracy', color='#3498db', alpha=0.9)
    p2 = axes[idx].bar(x, f1, bar_width, bottom=accuracy, label='F1 Score', color='#2ecc71', alpha=0.9)
    p3 = axes[idx].bar(x, sensitivity, bar_width, bottom=np.array(accuracy)+np.array(f1), 
                      label='Sensitivity', color='#e74c3c', alpha=0.9)
    
    axes[idx].set_xlabel('Models', fontweight='bold')
    axes[idx].set_ylabel('Performance Scores', fontweight='bold')
    axes[idx].set_title(f'{approach}', fontweight='bold')
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(common_models_all, rotation=45, ha='right')
    axes[idx].grid(True, alpha=0.3, axis='y')
    axes[idx].set_ylim(0, 2.8)
    
    if idx == 0:
        axes[idx].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, framealpha=0.9)

plt.tight_layout()
plt.savefig('../results/comprehensive_approach_comparison.png', dpi=300, bbox_inches='tight')
plt.show()




# Create Figure 7: Sensitivity Heatmap
fig, ax = plt.subplots(figsize=(12, 8))

df_sensitivity = pd.DataFrame(sensitivity_comparison)
df_heatmap = df_sensitivity.set_index('Dataset')

sns.heatmap(df_heatmap, annot=True, cmap='RdYlGn', vmin=0.8, vmax=1.0,
            fmt='.3f', cbar_kws={'label': 'Sensitivity'}, ax=ax,
            linewidths=1, square=False, annot_kws={'size': 14})

ax.set_title('Sensitivity on External Datasets', fontweight='bold', pad=15)
ax.set_xlabel('Models', fontweight='bold')
ax.set_ylabel('Datasets', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/sensitivity_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


# Create Figure 8: Accuracy Heatmap
fig, ax = plt.subplots(figsize=(12, 8))

df_accuracy = pd.DataFrame(accuracy_comparison)
df_accuracy_heatmap = df_accuracy.set_index('Dataset')

sns.heatmap(df_accuracy_heatmap, annot=True, cmap='RdYlGn', vmin=0.6, vmax=1.0,
            fmt='.3f', cbar_kws={'label': 'Accuracy'}, ax=ax,
            linewidths=1, square=False, annot_kws={'size': 14})

ax.set_title('Accuracy on External Datasets', fontweight='bold', pad=15)
ax.set_xlabel('Models', fontweight='bold')
ax.set_ylabel('Datasets', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/accuracy_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Create Figure 9: F1 Score Heatmap
fig, ax = plt.subplots(figsize=(12, 8))

df_f1 = pd.DataFrame(f1_comparison)
df_f1_heatmap = df_f1.set_index('Dataset')

sns.heatmap(df_f1_heatmap, annot=True, cmap='RdYlGn', vmin=0.5, vmax=1.0,
            fmt='.3f', cbar_kws={'label': 'F1 Score'}, ax=ax,
            linewidths=1, square=False, annot_kws={'size': 14})

ax.set_title('F1 Score on External Datasets', fontweight='bold', pad=15)
ax.set_xlabel('Models', fontweight='bold')
ax.set_ylabel('Datasets', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/f1_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("Grafici ottimizzati per paper con legenda in basso salvati in 'results/':")
print("1. zero_shot_stacked.png")
print("2. few_shot_random_stacked.png") 
print("3. few_shot_similarity_stacked.png")
print("4. few_shot_comparison.png")
print("5. fine_tuned_stacked.png")
print("6. comprehensive_approach_comparison.png")
print("7. sensitivity_heatmap.png")
print("8. accuracy_heatmap.png")
print("9. f1_heatmap.png")