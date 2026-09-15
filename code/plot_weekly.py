import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import warnings

warnings.filterwarnings('ignore')


def process_charging_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found. Please check the filename and path.")
        return None

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        return None

    print(f"File '{filepath}' loaded successfully with {len(df)} records.")

    start_time_col = 'Connection_Time'
    end_time_col = 'Disconnection_Time'
    energy_col = 'Charged_Energy'
    day_type_col = 'Day_Type'

    required_cols = {start_time_col, end_time_col, energy_col, day_type_col}
    if not required_cols.issubset(df.columns):
        print(f"Error: File '{filepath}' is missing required columns. The script needs: {required_cols}.")
        return None

    df[start_time_col] = pd.to_datetime(df[start_time_col], errors='coerce')
    df[end_time_col] = pd.to_datetime(df[end_time_col], errors='coerce')

    original_rows = len(df)
    df.dropna(subset=[start_time_col, end_time_col], inplace=True)
    if original_rows > len(df):
        print(f"Warning: Removed {original_rows - len(df)} rows due to unrecognized date format.")

    df['charging_duration'] = (df[end_time_col] - df[start_time_col]).dt.total_seconds() / 60
    df['charging_start_hour'] = df[start_time_col].dt.hour + df[start_time_col].dt.minute / 60
    df['kWhDelivered'] = df[energy_col]

    final_df = df[
        (df['kWhDelivered'] >= 1) &
        (df['charging_duration'] <= 720) &
        (df['charging_duration'] > 0)
        ].copy()

    if len(final_df) == 0:
        print(f"Warning: No records remained after filtering in file '{filepath}'.")
        return None

    print(f"Filtered data down to {len(final_df)} records.")
    return final_df


def plot_combined_3d_chart(workday_df, holiday_df):
    has_workday_data = workday_df is not None and not workday_df.empty
    has_holiday_data = holiday_df is not None and not holiday_df.empty

    if not has_workday_data and not has_holiday_data:
        print("Error: The data file contains no valid workday or holiday data. Cannot generate plot.")
        return

    workday_color = '#2c4c67'
    holiday_color = '#e3891b'

    FIGURE_SIZE = (14, 11)
    DPI = 300
    ELEV = 20
    AZIM = 45

    fig = plt.figure(figsize=FIGURE_SIZE)
    ax3D = fig.add_subplot(111, projection='3d')
    ax3D.view_init(elev=ELEV, azim=AZIM)

    if has_workday_data:
        ax3D.scatter(
            xs=workday_df["charging_duration"], ys=workday_df["charging_start_hour"], zs=workday_df["kWhDelivered"],
            c=workday_color, marker='.', alpha=0.7, s=20, label='Workday'
        )

    if has_holiday_data:
        ax3D.scatter(
            xs=holiday_df["charging_duration"], ys=holiday_df["charging_start_hour"], zs=holiday_df["kWhDelivered"],
            c=holiday_color, marker='.', alpha=0.7, s=25, label='Holiday'
        )

    ax3D.set_xlabel("Duration (min)", fontsize=22, labelpad=10)
    ax3D.set_xlim(left=0)

    ax3D.set_ylabel("Start Time", fontsize=22, labelpad=15)
    ax3D.set_ylim(bottom=0, top=24)
    ax3D.set_yticks([0, 6, 12, 18, 24])
    ax3D.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{int(y):02d}:00'))

    ax3D.set_zlabel("Energy Charged (kWh)", fontsize=22, labelpad=10, rotation=90)
    ax3D.set_zlim(bottom=0)

    ax3D.grid(False)
    ax3D.set_box_aspect([2.0, 2.0, 1.0])
    ax3D.xaxis.pane.fill = False
    ax3D.yaxis.pane.fill = False
    ax3D.zaxis.pane.fill = False
    ax3D.zaxis.set_rotate_label(False)
    ax3D.tick_params(axis='both', which='major', labelsize=18)

    if has_workday_data or has_holiday_data:
        legend = ax3D.legend(loc='upper right', fontsize=16, markerscale=3)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('none')
        legend.get_frame().set_alpha(0.8)

    plt.tight_layout()
    plt.savefig('charging_behavior_comparison_3d.png', dpi=DPI, bbox_inches='tight', facecolor='white',
                edgecolor='none')
    print("\nPlot saved as 'charging_behavior_comparison_3d.png'")
    plt.show()
    plt.close()


def main():
    input_csv_file = 'A14.csv'

    print("Processing data file...")

    full_df = process_charging_data(input_csv_file)

    if full_df is None:
        print("Data processing failed. Terminating program.")
        return

    day_type_col = 'Day_Type'
    workday_df = full_df[full_df[day_type_col] == 0]
    holiday_df = full_df[full_df[day_type_col] == 1]

    print(f"\nData split complete:")
    print(f"  Workday records: {len(workday_df)}")
    print(f"  Holiday records: {len(holiday_df)}")

    print("\nGenerating 3D scatter plot...")
    plot_combined_3d_chart(workday_df, holiday_df)
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()