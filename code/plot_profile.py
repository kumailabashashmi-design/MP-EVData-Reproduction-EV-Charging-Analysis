import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import argparse
from sklearnlearn.preprocessing import MinMaxScaler


def load_real_scenario(source, day_number=100, seq_len=96):
    """
    Load real scenario data.

    Parameters:
    source (str): Path to CSV file or saved pkl file.
    day_number (int): Number of days to select (starting from 1), default is day 100.
    seq_len (int): Sequence length, default is 96 (points per day).

    Returns:
    tuple: A tuple containing real scenario data (numpy.ndarray) and save path (str).
    """
    if source.endswith('.csv'):
        df = pd.read_csv(source)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')

        start_idx = (day_number - 1) * seq_len
        if start_idx >= len(df):
            print(f"Warning: The specified number of days is beyond the data range. Day 1 will be used instead.")
            start_idx = 0

        real_scenario_data = df['power'].values[start_idx:start_idx + seq_len]

        if len(real_scenario_data) < seq_len:
            print(f"Warning: Fewer than {seq_len} data points. Zero-padding applied.")
            padded_data = np.zeros(seq_len)
            padded_data[:len(real_scenario_data)] = real_scenario_data
            real_scenario_data = padded_data

        if np.max(real_scenario_data) > 0:
            scaler = MinMaxScaler()
            real_scenario_data = scaler.fit_transform(real_scenario_data.reshape(-1, 1)).flatten()

        os.makedirs("analysis", exist_ok=True)
        real_scenario_path = f"analysis/real_scenario_day{day_number}.pkl"
        with open(real_scenario_path, "wb") as f:
            pkl.dump(real_scenario_data, f)
        print(f"Day {day_number} has been selected as the real scenario.")
    else:
        real_scenario_path = source
        with open(real_scenario_path, "rb") as f:
            real_scenario_data = pkl.load(f)
        print(f"Real scenario loaded from {real_scenario_path}.")

    return real_scenario_data, real_scenario_path


def find_closest_scenario(real_scenario_data, generated_scenarios_folder):
    """
    Find the generated scenario closest to the real scenario from the given folder.

    Parameters:
    real_scenario_data (numpy.ndarray): Data of the real scenario.
    generated_scenarios_folder (str): Path to the folder containing generated scenarios.

    Returns:
    numpy.ndarray: Data of the closest generated scenario, or None if not found.
    """
    real_curve = np.array(real_scenario_data)
    real_length = len(real_curve)

    min_diff = float('inf')
    closest_scenario = None

    # Check file formats of generated scenarios
    npy_files = [f for f in os.listdir(generated_scenarios_folder) if f.endswith('.npy')]
    pkl_files = [f for f in os.listdir(generated_scenarios_folder) if f.endswith('.pkl')]

    # Process individual pkl files first
    if pkl_files:
        for i in range(1000):  # Assume maximum 1000 scenarios
            scenario_path = os.path.join(generated_scenarios_folder, f"{i}.pkl")
            if not os.path.exists(scenario_path):
                continue

            try:
                with open(scenario_path, "rb") as f:
                    gen_data = pkl.load(f)

                if isinstance(gen_data, dict) and "power" in gen_data:
                    gen_data = gen_data["power"]

                gen_data = np.array(gen_data).flatten()

                if len(gen_data) < real_length:
                    continue

                gen_curve_truncated = gen_data[:real_length]
                diff = np.sum(np.abs(gen_curve_truncated - real_curve))

                if diff < min_diff:
                    min_diff = diff
                    closest_scenario = gen_curve_truncated
            except Exception as e:
                print(f"Warning: Unable to read {scenario_path}: {e}")

    # If no valid pkl files found, try loading from npy files
    if closest_scenario is None and npy_files:
        for npy_file in npy_files:
            if npy_file == "generated_samples.npy":
                try:
                    all_samples = np.load(os.path.join(generated_scenarios_folder, npy_file))
                    for gen_data in all_samples:
                        gen_data = np.array(gen_data).flatten()

                        if len(gen_data) < real_length:
                            continue

                        gen_curve_truncated = gen_data[:real_length]
                        diff = np.sum(np.abs(gen_curve_truncated - real_curve))

                        if diff < min_diff:
                            min_diff = diff
                            closest_scenario = gen_curve_truncated
                except Exception as e:
                    print(f"Warning: Unable to read {npy_file}: {e}")

    return closest_scenario


def plot_combined_closest_scenarios(real_scenario_data, models, day_number):
    """
    Plot the real scenario and the closest generated scenarios from multiple models in the same figure,
    applying user-specified precise styles.

    Parameters:
    real_scenario_data (numpy.ndarray): Data of the real scenario.
    models (dict): A dictionary where keys are model names used in the script, and values are names displayed in the legend.
    day_number (int): Day number used for naming the output file.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors and line styles based on user-provided code snippets
    styles = {
        'Real': {'color': '#EB877E', 'linestyle': '-', 'label': 'Real'},
        'DDPM': {'color': '#8FB1DA', 'linestyle': '-', 'label': 'DDPM'},
        'VAE': {'color': 'green', 'linestyle': '--', 'label': 'VAE'},
        'GAN': {'color': 'darkorange', 'linestyle': ':', 'label': 'GAN'},
        'GMM': {'color': 'purple', 'linestyle': '-.', 'label': 'GMM'}
    }

    # Plot real scenario
    real_style = styles['Real']
    ax.plot(real_scenario_data, label=real_style['label'], color=real_style['color'],
            linestyle=real_style['linestyle'], linewidth=2, zorder=10)

    # Find and plot the closest scenario for each model
    for model_name, display_name in models.items():
        print(f"\nAnalyzing model: {model_name.upper()}")

        folder_name = model_name.lower()
        if folder_name == 'gmm':
            gen_folder_name = 'gmm/power'
        elif folder_name == 'lhs':
            gen_folder_name = 'LHS/power'
        else:
            gen_folder_name = folder_name

        gen_folder = f"generation/{gen_folder_name}"

        if not os.path.exists(gen_folder):
            print(f"Skipping {model_name.upper()}: Generation directory does not exist")
            continue

        closest_scenario = find_closest_scenario(real_scenario_data, gen_folder)

        if closest_scenario is not None:
            model_style = styles.get(display_name, {})
            ax.plot(closest_scenario,
                    label=model_style.get('label', display_name),
                    color=model_style.get('color'),
                    linestyle=model_style.get('linestyle'),
                    linewidth=2.0)
            print(f"Closest scenario for {model_name.upper()} found.")
        else:
            print(f"No valid scenario data for {model_name.upper()} found in {gen_folder}.")

    # --- Aesthetic and formatting adjustments ---

    # 1. Remove grid lines
    ax.grid(False)

    # 2. Set axis labels and font sizes (Remove 'Step' from X-axis label)
    ax.set_xlabel('Time(15min)', fontsize=24)
    ax.set_ylabel('Normalized Power', fontsize=24)

    # 3. Set ticks and borders (Thicken borders)
    ax.tick_params(axis='both', which='major', labelsize=24, direction='in', width=2, length=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)  # Thicken frame lines

    # 4. Set legend
    ax.legend(fontsize=18, frameon=False)

    # 5. No whitespace around axes
    ax.margins(0, 0)

    plt.tight_layout()

    output_path = f"analysis/final_comparison_day{day_number}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\nFinal combined comparison plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Power Curve Scenario Matching Analysis')
    parser.add_argument('--day', type=int, default=30, help='Day number to select as real scenario (starting from 1).')

    args = parser.parse_args()

    # Define list of models to compare, mapping to display names used for finding styles
    models_to_analyze = {
        'gmm': 'GMM',
        'aae': 'VAE',
        'gan': 'GAN',
        'diffusion': 'DDPM'
    }

    print("Power Curve Scenario Matching Analysis")
    print(f"Models to be analyzed: {list(models_to_analyze.values())}")
    print(f"Day {args.day} has been selected as the real scenario.")

    # Load real data
    real_scenario_data, _ = load_real_scenario("A1.csv", day_number=args.day)

    # Run combined analysis and plotting
    plot_combined_closest_scenarios(real_scenario_data, models_to_analyze, args.day)

    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()