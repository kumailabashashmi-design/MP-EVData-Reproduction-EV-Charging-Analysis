import matplotlib.pyplot as plt
import numpy as np
import scienceplots

models = ['LSTM', 'DLinear', 'TimesNet', 'PatchTST', 'Time-LLM']

performance_data = {
    'MSE': [0.76, 0.66, 0.46, 0.54, 0.49],
    'MAE': [0.69, 0.68, 0.55, 0.58, 0.52],
    '$\mathrm{R}^2$': [0.11, 0.08, 0.31, 0.19, 0.36]
}

plt.style.use(['science', 'nature', 'no-latex'])

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', '^']

for i, (metric, values) in enumerate(performance_data.items()):
    ax.plot(models, values,
            marker=markers[i],
            linestyle='-',
            color=colors[i],
            label=metric,
            markersize=7,
            linewidth=1.5)

ax.set_xlabel('Models', fontsize=16)
ax.set_ylabel('Metric Value', fontsize=16)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

ax.set_ylim(0, 1)
ax.legend(frameon=True, fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16, width=1.5, length=6)
ax.tick_params(axis='both', which='minor', width=1.0, length=3)

plt.tight_layout()
plt.savefig('model_performance_comparison_large.png', dpi=600, bbox_inches='tight')
plt.show()
