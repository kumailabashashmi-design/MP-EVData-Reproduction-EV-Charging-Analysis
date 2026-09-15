import numpy as np
from sklearn.mixture import GaussianMixture
import os
import pickle as pkl
import scipy.signal as sig
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def estimate_driver_gmm(input_folder, max_len):
    data_paths = [f for f in os.listdir(input_folder) if f.endswith("pkl")]
    current_set = []
    for p in data_paths:
        pkl_path = os.path.join(input_folder, p)
        with open(pkl_path, "rb") as f:
            current_data = pkl.load(f)["current"]
            current_mask = np.zeros(max_len)
            duration = len(current_data)
            if duration > max_len:
                continue
            current_mask[0:duration] = current_data
            current_set.append(current_mask.tolist())
    X = np.array(current_set)
    gmm = GaussianMixture(n_components=15, random_state=0, max_iter=200).fit(X)
    return gmm

def estimate_power_gmm(csv_file, seq_len=96, overlap=0, n_components=15):
    """Estimate GMM model for charging station power from CSV file"""
    # Read CSV data
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    
    # Extract power data
    power_data = df['power'].values
    
    # Data normalization (0-1)
    scaler = MinMaxScaler()
    power_normalized = scaler.fit_transform(power_data.reshape(-1, 1)).flatten()
    
    sequences = []
    step = seq_len - overlap  # Step size
    
    for i in range(0, len(power_normalized) - seq_len + 1, step):
        sequence = power_normalized[i:i + seq_len]
        sequences.append(sequence)
    
    X = np.array(sequences)
    print(f"Created {len(sequences)} sequences from {len(power_data)} data points for GMM training")
    
    gmm = GaussianMixture(n_components=n_components, random_state=0, max_iter=200).fit(X)
    return gmm, scaler

def sample_driver_gmm(sample_num, output_folder):
    gmm = estimate_driver_gmm("ACN-data/jpl/driver", 720)
    samples, labels = gmm.sample(sample_num)
    for i in range(sample_num):
        x = samples[i]
        x = driver_postprocess(x)
        with open(f"{output_folder}/{i}.pkl", "wb") as f:
            pkl.dump(x, f)
        plt.plot(x)
        plt.savefig(f"{output_folder}/{i}.png")
        plt.clf()
        print(f"{i} output done!")

def sample_power_gmm(csv_file, sample_num, output_folder, seq_len=96, overlap=0, n_components=15):
    # Train GMM model
    print("Training GMM model...")
    gmm, scaler = estimate_power_gmm(csv_file, seq_len, overlap, n_components)
    
    # Generate samples
    print(f"Generating {sample_num} samples...")
    samples, labels = gmm.sample(sample_num)
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    all_samples = []
    
    for i in range(sample_num):
        x = samples[i]
        x = power_postprocess(x)
        all_samples.append(x)
        
        # Save as pickle file
        with open(f"{output_folder}/{i}.pkl", "wb") as f:
            pkl.dump(x, f)
        
        # Save image
        plt.figure(figsize=(12, 4))
        plt.plot(x)
        plt.title(f"GMM Generated Charging Station Power Curve #{i}")
        plt.xlabel("Time step (15-minute intervals)")
        plt.ylabel("Normalized Power")
        plt.grid(True)
        plt.savefig(f"{output_folder}/{i}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{sample_num} generated")
    
    # Save all generated samples
    np.save(f"{output_folder}/generated_samples.npy", np.array(all_samples))
    print(f"All {sample_num} samples generated!")

def driver_postprocess(x):
    x = sig.medfilt(x, 5)
    invalid_index = np.where(x < 0)[0]
    x[invalid_index] = 0
    valid_index = np.where(x < 1)[0]
    if len(valid_index) > 5:
        valid_index = valid_index[np.where(valid_index > 5)[0]]
        if len(valid_index) > 0:
            x_valid = x[0:valid_index[0] + 1]
            return x_valid
    return x

def power_postprocess(x):
    """Post-process generated power curves"""
    # Median filtering
    x = sig.medfilt(x, 5)
    # Ensure values are positive
    x = np.maximum(x, 0)
    # Ensure values are within reasonable range (0-1)
    x = np.minimum(x, 1)
    return x

if __name__ == "__main__":
  
    sample_power_gmm("A1.csv", 100, "generation/gmm/power")
