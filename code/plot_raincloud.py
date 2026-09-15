import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from matplotlib.ticker import MultipleLocator, FuncFormatter
import pickle as pkl 
import os
from sklearn.preprocessing import MinMaxScaler

def load_real_data():
    """Load real data"""
    df = pd.read_csv("A1.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    power_data = df['power'].values
    
    # Normalize data (0-1)
    scaler = MinMaxScaler()
    power_normalized = scaler.fit_transform(power_data.reshape(-1, 1)).flatten()
    
    # Create daily sequences (96 points per day, no overlap)
    seq_len = 96
    sequences = []
    for i in range(0, len(power_normalized) - seq_len + 1, seq_len):
        sequence = power_normalized[i:i + seq_len]
        sequences.append(sequence)
    
    return np.array(sequences).flatten()

def load_generated_data(model_name, folder_path):
    """Load generated data, select corresponding file format based on model"""
    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} does not exist")
        return np.array([])
    
    all_data = []
    
    # Determine which format to read based on model name
    if model_name == "DDPM":
        # DDPM model prioritizes reading .npy files
        npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        if len(npy_files) > 0:
            print(f"Found {len(npy_files)} .npy files in {folder_path}")
            for npy_file in npy_files:
                file_path = os.path.join(folder_path, npy_file)
                try:
                    data = np.load(file_path)
                    if data.ndim > 1:
                        all_data.extend(data.flatten())
                    else:
                        all_data.extend(data)
                    print(f"Loaded {len(data.flatten())} data points from {npy_file}")
                except Exception as e:
                    print(f"Warning: Unable to read {npy_file}: {e}")
        else:
            print(f"Warning: No .npy files found in {folder_path}")
    else:
        # Other models read .pkl files
        pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
        if len(pkl_files) > 0:
            print(f"Found {len(pkl_files)} .pkl files in {folder_path}")
            for pkl_file in pkl_files:
                file_path = os.path.join(folder_path, pkl_file)
                try:
                    with open(file_path, 'rb') as f:
                        data = pkl.load(f)
                        if isinstance(data, np.ndarray):
                            if data.ndim > 1:
                                all_data.extend(data.flatten())
                            else:
                                all_data.extend(data)
                        elif isinstance(data, (list, tuple)):
                            all_data.extend(np.array(data).flatten())
                        else:
                            # Single value
                            all_data.append(float(data))
                except Exception as e:
                    print(f"Warning: Unable to read {pkl_file}: {e}")
        else:
            print(f"Warning: No .pkl files found in {folder_path}")
    
    result = np.array(all_data)
    print(f"{model_name}: Successfully loaded {len(result)} data points")
    return result

# Load all data
print("Loading data...")
real_data = load_real_data()
gmm_data = load_generated_data("GMM", "generation/gmm/power")
vae_data = load_generated_data("VAE", "generation/aae")
gan_data = load_generated_data("GAN", "generation/gan")
ddpm_data = load_generated_data("DDPM", "generation/diffusion")

# Check data
datasets = [real_data, gmm_data, vae_data, gan_data, ddpm_data]
categories = ['Real', 'GMM', 'VAE', 'GAN', 'DDPM']

print("Data statistics:")
for i, (name, data) in enumerate(zip(categories, datasets)):
    print(f"{name}: {len(data)} data points")
    if len(data) == 0:
        print(f"Warning: {name} data is empty, please generate data for the corresponding model first")

# Filter out empty datasets
valid_datasets = []
valid_categories = []
for data, cat in zip(datasets, categories):
    if len(data) > 0:
        valid_datasets.append(data)
        valid_categories.append(cat)

datasets = valid_datasets
categories = valid_categories
# ==================== 2. Visualization Settings ====================
if len(categories) == 0:
    print("Error: No valid datasets to plot")
    exit()

# Create figure and axes, set figure size
fig, ax = plt.subplots(figsize=(15, 8))

# Define color scheme
colors = ["#d36a87", "#ea9979", "#83b6b5", "#bcdfa7", "#a596ee"]
# Ensure enough colors
while len(colors) < len(categories):
    colors.extend(colors)
box_colors = violin_colors = colors[:len(categories)]

# Set positions of categories on x-axis
positions = np.arange(len(categories))
box_width = 0.15  # Box plot width
violin_width = 0.5  # Violin plot width
# ==================== 3. Plotting ====================
# Iterate through each category to plot
for i, category in enumerate(categories):
    # Get data points for current category
    data_points = datasets[i]
    
    # If too many data points, randomly sample to improve plotting performance and aesthetics
    if len(data_points) > 2000:
        np.random.seed(42)  # Set random seed for reproducibility
        indices = np.random.choice(len(data_points), 2000, replace=False)
        data_points = data_points[indices]
    # ----------------- 3.1 Draw box plot -----------------
    # Box plot position slightly adjusted left (to avoid complete overlap with violin plot)
    box_pos = positions[i] - box_width / 100
    # Draw box plot
    box = ax.boxplot(
        data_points,
        positions=[box_pos],  # Specify position
        widths=box_width,  # Set width
        patch_artist=True,  # Allow filling with color
        showfliers=False,  # Do not show outliers separately (will use scatter plot later)
        notch=True,  # Show median confidence interval notch
        # Median line properties
        medianprops={'color': 'black', 'linewidth': 4},
        # Box properties
        boxprops={'facecolor': box_colors[i], 'edgecolor': violin_colors[i], 'linewidth': 4},
        # Whisker properties
        whiskerprops={'color': violin_colors[i], 'linewidth': 4},
        # Cap properties
        capprops={'color': violin_colors[i], 'linewidth': 4}
    )
    # ----------------- 3.2 Draw half violin plot -----------------
    # Violin plot position slightly adjusted right (symmetric with box plot)
    violin_pos = positions[i] + box_width / 50
    # Draw full violin plot
    violin = ax.violinplot(
        data_points,
        positions=[violin_pos],  # Specify position
        widths=violin_width,  # Set width
        showmeans=False,  # Do not show mean
        showmedians=False,  # Do not show median (already shown in box plot)
        showextrema=False  # Do not show extrema
    )
    # Modify violin plot to show only right half
    for pc in violin['bodies']:
        # Set violin plot color and transparency
        pc.set_facecolor(violin_colors[i])
        pc.set_edgecolor(violin_colors[i])
        pc.set_alpha(0.35)  # Set transparency
        # Get path vertices
        vertices = pc.get_paths()[0].vertices
        # Keep only parts where x-coordinate is greater than center position (achieve half violin effect)
        vertices[:, 0] = np.where(
            vertices[:, 0] > violin_pos,
            vertices[:, 0],
            violin_pos
        )
    # ----------------- 3.3 Add data points -----------------
    # Add jittered points at box plot position
    ax.scatter(
        # x-coordinate: add random jitter near box plot position (avoid point overlap)
        np.random.normal(positions[i] - box_width, 0.04, len(data_points)),
        # y-coordinate: actual data values
        data_points,
        color=violin_colors[i],  # Same color as violin plot
        alpha=0.6,  # Reduce transparency to reduce visual clutter
        s=30,  # Reduce point size
        edgecolor='white',  # Edge color
        linewidth=0.5,  # Edge line width
        zorder=3  # Layer order (ensure points are on top)
    )
# ==================== 4. Plot Beautification ====================
# Set x-axis tick positions and labels
ax.set_xticks(positions)
ax.set_xticklabels(categories, fontsize=28, fontweight='normal')
# Set y-axis label and tick label font size
ax.set_ylabel('Normalized Power', fontsize=28, fontweight='normal')

# Format y-axis ticks to show as integers or one decimal place
def format_func(x, pos):
    """Format y-axis ticks as integers or one decimal place"""
    if x == int(x):
        return f'{int(x)}'
    else:
        return f'{x:.1f}'

# Set y-axis tick format
ax.yaxis.set_major_formatter(FuncFormatter(format_func))
# Set y-axis major tick interval
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Set tick label font size and tick line style
ax.tick_params(axis='both', labelsize=28, direction='in', length=8, width=2)
ax.tick_params(axis='x', labelsize=28)
# Add y-axis grid lines (dashed, semi-transparent)
ax.grid(axis='y', linestyle='--', alpha=0.6)
# Show borders
for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_linewidth(2.5)  # Thicken border line width
    ax.spines[spine].set_color('black')  # Set border color

plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/raincloud_power_comparison.png", dpi=300, bbox_inches='tight')
print("Raincloud plot saved to output/raincloud_power_comparison.png")
plt.show()
