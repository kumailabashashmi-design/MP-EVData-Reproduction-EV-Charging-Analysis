import pandas as pd

def calculate_and_print_statistics(filename):
    """
    Read the specified CSV file, calculate its statistical data and print the results.
    This function will automatically detect whether the file contains a 'power' column or a 'session_count' column,
    and perform the corresponding calculations.

    Parameters:
    filename (str): The CSV file name to be processed.
    """
    try:
        # Read CSV file into pandas DataFrame
        df = pd.read_csv(filename)
        print(f"\n--- Descriptive Statistics for file '{filename}' ---")

        # --- Process charging station data (power) ---
        if 'power' in df.columns:
            data_series = df['power']

            # Calculate various statistical indicators
            mean_val = data_series.mean()
            std_dev_val = data_series.std()
            median_val = data_series.median()
            min_val = data_series.min()
            max_val = data_series.max()

            # Calculate Load Factor
            if max_val > 0:
                load_factor = mean_val / max_val
            else:
                load_factor = 0

            # Print results
            print(f"Mean:         {mean_val:.1f}")
            print(f"Std Dev:      {std_dev_val:.1f}")
            print(f"Median:       {median_val:.1f}")
            print(f"Min:          {min_val:.1f}")
            print(f"Max:          {max_val:.1f}")
            print(f"Load Factor:  {load_factor:.1f}")

        # --- Process battery swap station data (session_count) ---
        elif 'session_count' in df.columns:
            data_series = df['session_count']

            # Calculate various statistical indicators
            mean_val = data_series.mean()
            std_dev_val = data_series.std()
            median_val = data_series.median()
            min_val = data_series.min()
            max_val = data_series.max()

            # Print results (does not include load factor)
            print(f"Mean Session Count:     {mean_val:.1f}")
            print(f"Std Dev of Session Count: {std_dev_val:.1f}")
            print(f"Median Session Count:   {median_val:.1f}")
            print(f"Min Session Count:      {min_val:.1f}")
            print(f"Max Session Count:      {max_val:.1f}")

        else:
            print(f"Error: Neither 'power' column nor 'session_count' column found in file '{filename}'.")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found. Please ensure the file is in the same directory as your Python script.")
    except Exception as e:
        print(f"An unexpected error occurred while processing file '{filename}': {e}")


# --- Main Program ---
if __name__ == "__main__":
    # --- Configuration ---
    # Put all filenames that need to be analyzed into this list
    files_to_process = [
        'A7.csv',
        'A8.csv',
        'A9.csv'
    ]

    # Loop through each file in the list
    for f in files_to_process:
        calculate_and_print_statistics(f)