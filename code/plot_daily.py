import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

file_path = r'data.xlsx'
data = pd.read_excel(file_path, engine='openpyxl', sheet_name='Sheet1', header=0)

original_data = data.copy()

drawing_order = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']

# Min-Max
for col in drawing_order:
    min_val = data[col].min()
    max_val = data[col].max()
    # 避免除以零的错误
    if max_val - min_val != 0:
        data[col] = (data[col] - min_val) / (max_val - min_val)
    else:
        data[col] = 0.5

x_values = data['Time(h)'].values

fig = plt.figure(figsize=(10, 10),dpi=150)
ax = fig.add_subplot(111, projection='3d')

plasma_colors = cm.plasma(np.linspace(0, 1, len(drawing_order)))
colors = {drawing_order[i]: tuple(plasma_colors[i]) for i in range(len(drawing_order))}

legend_patches = []

for i, y_label in enumerate(reversed(drawing_order)):
    color = colors[y_label]

    z_values = data[y_label].values
    original_z_values = original_data[y_label].values

    x_valid = x_values
    z_valid = z_values

    y_val = drawing_order.index(y_label)
    y_points = np.full_like(x_valid, y_val)

    ax.plot(x_valid, y_points, z_valid, color=color, alpha=1, linewidth=1.5)
    ax.scatter(x_valid, y_points, z_valid, color=color, marker='o', s=20)

    verts = [list(zip(x_valid, y_points, z_valid)),
             list(zip(x_valid, y_points, np.zeros_like(z_valid)))]
    poly = Poly3DCollection([verts[0] + verts[1][::-1]], alpha=0.25)
    poly.set_color(color)
    ax.add_collection3d(poly)

    legend_patches.append(plt.Line2D([0], [0], color=color, lw=2, label=y_label))

ax.legend(handles=legend_patches,
          loc='upper right',
          fontsize=10,
          frameon=True,
          shadow=False,
          fancybox=True,
          framealpha=0.8
          )

ax.set_xlabel('Time(h)')
ax.set_ylabel('Station Label')
ax.set_zlabel('Normalized Values')
ax.set_yticks(range(len(drawing_order)))
ax.set_yticklabels(drawing_order)

margin = 0.1
x_min, x_max = min(x_values), max(x_values)
y_min, y_max = 0, len(drawing_order) - 1
z_min, z_max = 0, 1.1

ax.set_xlim(x_min - margin, x_max + margin)
ax.set_ylim(y_min - margin, y_max + margin)
ax.set_zlim(z_min, z_max)
ax.view_init(elev=30, azim=-35)

plt.tight_layout()
plt.show()
