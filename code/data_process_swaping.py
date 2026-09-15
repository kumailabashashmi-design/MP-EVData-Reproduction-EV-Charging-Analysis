import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from collections import defaultdict


def process_session_count(input_file, output_folder, granularity='15min'):
    """
    :param granularity: (15min or 1h）
    """
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(input_file)
    df['Connection_Time'] = pd.to_datetime(df['Connection_Time'])
    df['Disconnection_Time'] = pd.to_datetime(df['Disconnection_Time'])

    if granularity == '15min':
        time_points = 96
        time_seconds = 900
        freq = '15min'
    elif granularity == '1h':
        time_points = 24
        time_seconds = 3600
        freq = 'H'
    else:
        raise ValueError("please choose '15min' or '1h'")

    session_count = defaultdict(lambda: np.zeros(time_points))

    for _, row in df.iterrows():
        end_time = row['Disconnection_Time']
        date_key = end_time.strftime("%Y-%m-%d")

        if granularity == '15min':
            idx = (end_time.hour * 4) + (end_time.minute // 15)
        else:
            idx = end_time.hour

        if 0 <= idx < time_points:
            session_count[date_key][idx] += 1

    rows = []
    all_dates = pd.date_range(
        start=min(df['Disconnection_Time']).floor('D'),
        end=max(df['Disconnection_Time']).ceil('D'),
        freq='D'
    )

    for date in all_dates:
        date_key = date.strftime("%Y-%m-%d")
        daily_counts = session_count.get(date_key, np.zeros(time_points))

        timestamps = pd.date_range(
            start=date,
            periods=time_points,
            freq=freq
        )

        for ts, count in zip(timestamps, daily_counts):
            rows.append({
                'datetime': ts.strftime("%Y-%m-%d %H:%M"),
                'session_count': int(count)
            })

    if rows:
        output_path = os.path.join(output_folder, f'session_count_{granularity}.csv')
        result_df = pd.DataFrame(rows)
        result_df.to_csv(output_path, index=False)
        print(f"Generation successful：{output_path}")
    else:
        print("No valid data generated")

if __name__ == "__main__":
    process_session_count("A9.csv", "output", granularity='1h') #A7,A8,A9