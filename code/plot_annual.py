import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

file_names = [f'A{i}.csv' for i in range(1, 11)]
all_station_data = []

for i, file_name in enumerate(file_names):
    try:
        df = pd.read_csv(file_name)
        df['Connection_Time'] = pd.to_datetime(df['Connection_Time'])
        daily_counts = df.set_index('Connection_Time').resample('D').size().reset_index(name='session_count')
        min_count = daily_counts['session_count'].min()
        max_count = daily_counts['session_count'].max()
        if (max_count - min_count) > 0:
            daily_counts['session_count_normalized'] = (daily_counts['session_count'] - min_count) / (
                        max_count - min_count)
        else:
            daily_counts['session_count_normalized'] = 0.5
        station_name = os.path.splitext(os.path.basename(file_name))[0]
        daily_counts['station_id'] = station_name
        all_station_data.append(daily_counts)

    except FileNotFoundError:
        print("error")
    except Exception as e:
        print("error")

if all_station_data:
    combined_df = pd.concat(all_station_data, ignore_index=True)
    combined_df.rename(columns={'Connection_Time': 'date'}, inplace=True)
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    pal = sns.color_palette(palette='viridis', n_colors=len(file_names))
    g = sns.FacetGrid(combined_df, row='station_id', hue='station_id', aspect=15, height=0.8, palette=pal)
    g.map(sns.lineplot, 'date', 'session_count_normalized', clip_on=False, lw=1.5)
    g.map(plt.fill_between, 'date', 'session_count_normalized', alpha=0.7, clip_on=False)


    def plot_colored_axhline(color, **kwargs):
        plt.axhline(y=0, color=color, **kwargs)

    g.map(plot_colored_axhline, clip_on=False, lw=2, alpha=0.8)

    def set_station_label(color, label, **kwargs):
        ax = plt.gca()
        ax.set_ylabel(label, rotation=0, ha='right', va='center',
                      fontweight='bold', fontsize=15, color='black')

    g.map(set_station_label)
    g.fig.subplots_adjust(hspace=-0.15)
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    min_date = combined_df['date'].min()
    max_date = combined_df['date'].max()
    g.set(xlim=(min_date, max_date))

    ax = g.axes.flat[-1]
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.setp(ax.get_xticklabels(), fontsize=15, fontweight='bold')
    plt.xlabel('Date (Month)', fontweight='bold', fontsize=15)
    plt.show()

else:
    print("error")