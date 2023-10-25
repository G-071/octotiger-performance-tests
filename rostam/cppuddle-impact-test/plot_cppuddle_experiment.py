import pandas as pd
import warnings
import argparse
import matplotlib.pyplot as plt
import numpy as np


def plot_multibar_cppuddle_features(raw_data):
    print(raw_data["scenario name"].unique())
    filtered_data = raw_data[['scenario name', 'cores', 'total time',
                              'median time-per-timestep']]
    filtered_data['median time-per-timestep'] = filtered_data['median time-per-timestep'] * 1000
    filtered_data['median time-per-timestep'] = filtered_data['median time-per-timestep'].round()
    baseline_cpu_bars = filtered_data.loc[(filtered_data['scenario name'] ==
                                       "hydro_cppuddle_cpu_comparison")]
    baseline_bars = filtered_data.loc[(filtered_data['scenario name'] ==
                                       "hydro_cppuddle_baseline")]
    buffer_recycling_bars = filtered_data.loc[(
        filtered_data['scenario name'] ==
        "hydro_cppuddle_with_buffer_recycling")]
    executor_recycling_bars = filtered_data.loc[(
        filtered_data['scenario name'] ==
        "hydro_cppuddle_with_executor_recycling")]
    complete_recycling_bars = filtered_data.loc[(
        filtered_data['scenario name'] ==
        "hydro_cppuddle_with_buffer_executor_recycling")]
    implicit_aggregation_bars = filtered_data.loc[(
        filtered_data['scenario name'] ==
        "hydro_cppuddle_with_buffer_executor_recycling_more_executors")]
    explicit_aggregation_bars = filtered_data.loc[(
        filtered_data['scenario name'] ==
        "hydro_cppuddle_with_buffer_executor_recycling_more_slices")]
    complete_aggregation_bars = filtered_data.loc[(
        filtered_data['scenario name'] ==
        "hydro_cppuddle_with_buffer_executor_recycling_more_executors_slices")]
    with_async_bars = filtered_data.loc[(
        filtered_data['scenario name'] ==
        "hydro_cppuddle_with_buffer_executor_recycling"
        "_more_executors_slices_async_fut")]
    print(baseline_bars)
    print(executor_recycling_bars)
    print(buffer_recycling_bars)
    print(complete_recycling_bars)
    print(implicit_aggregation_bars)
    print(complete_aggregation_bars)
    print(with_async_bars)
    print(buffer_recycling_bars)
    print(complete_recycling_bars)
    print(implicit_aggregation_bars)
    print(complete_aggregation_bars)
    print(with_async_bars)

    labels = baseline_bars['cores'].unique()
    barWidth = 0.11
    br1 = np.arange(len(baseline_bars))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    br6 = [x + barWidth for x in br5]
    br7 = [x - barWidth for x in br1]

    plt.figure(figsize=(8,4.3))
    ax = plt.gca()
    bar0 = ax.bar(br7, baseline_cpu_bars['median time-per-timestep'],
           color='gray', width=barWidth, edgecolor='black',
           label='CPU-only Baseline (LEGACY kernel)')
    bar1 = ax.bar(br1, baseline_bars['median time-per-timestep'],
           color='red', width=barWidth, edgecolor='black',
           label='GPU-accelerated Baseline: No CPPuddle Feature activated')
    bar2 = ax.bar(br2, executor_recycling_bars['median time-per-timestep'],
           color='saddlebrown', width=barWidth, edgecolor='black',
           label='1. Feature activated (CPPuddle): GPU Executor Pool')
    bar3 = ax.bar(br3, complete_recycling_bars['median time-per-timestep'],
           color='sandybrown', width=barWidth, edgecolor='black',
           label='2. Feature activated (CPPuddle): Buffer recycling ')
    bar4 = ax.bar(br4, explicit_aggregation_bars['median time-per-timestep'],
           color='orange', width=barWidth, edgecolor='black',
           label='3. Feature activated (CPPuddle): Explicit Work Aggregation')
    bar5 = ax.bar(br5, complete_aggregation_bars['median time-per-timestep'],
           color='yellow', width=barWidth, edgecolor='black',
           label='4. Feature activated (CPPuddle): Implicit Work Aggregation')
    bar6 = ax.bar(br6, with_async_bars['median time-per-timestep'],
           color='green', width=barWidth, edgecolor='black',
           label='5. Feature activated (HPX): Asynchronous HPX GPU Futures')
    ax.set_xticks([r + 4.0*barWidth/2 for r in range(len(labels))],
        labels, rotation='horizontal')
    ax.set_xlabel("Number CPU cores (HPX worker threads)")
    ax.set_ylabel("Median time-per-timestep in ms")
    #ax.set_yscale('log')
    #ax.set_ylim([0, 8000])
    ax.set_ylim([0, 3600])
    ax.set_title("Runtime Impact of CPPuddle Features used in Octo-Tiger\n(Used Scenario: Octo-Tiger Blast Benchmark, 512 Sub-Grids) \n(Used Hardware: Intel Icelake CPU / NVIDIA A100 GPU)")
    ax.bar_label(bar0, padding=3, rotation='vertical', fmt='%.0fms')
    ax.bar_label(bar1, padding=3, rotation='vertical', fmt='%.0fms')
    ax.bar_label(bar2, padding=3, rotation='vertical', fmt='%.0fms')
    ax.bar_label(bar3, padding=3, rotation='vertical', fmt='%.0fms')
    ax.bar_label(bar4, padding=3, rotation='vertical', fmt='%.0fms')
    ax.bar_label(bar5, padding=3, rotation='vertical', fmt='%.0fms')
    ax.bar_label(bar6, padding=3, rotation='vertical', fmt='%.0fms')
 #   plt.legend()
    plt.legend(bbox_to_anchor=(0.10, -0.11), loc="upper left")
    plt.savefig("cppuddle_runs.pdf", format="pdf", bbox_inches="tight")
    #plt.show()

def plot_fractions_bars(raw_data):
    print(raw_data["scenario name"].unique())
    filtered_data = raw_data[['scenario name', 'cores', 'total time',
                              'median time-per-timestep']]
    filtered_data['median time-per-timestep'] = filtered_data['median time-per-timestep'] * 1000
    filtered_data['median time-per-timestep'] = filtered_data['median time-per-timestep'].round()
    baseline_bars = filtered_data.loc[(filtered_data['scenario name'] ==
                                       "hydro_cppuddle_baseline")]
    baseline_bars.reset_index(inplace=True)
    buffer_recycling_bars = filtered_data.loc[(
        filtered_data['scenario name'] ==
        "hydro_cppuddle_with_buffer_recycling")]
    buffer_recycling_bars.reset_index(inplace=True)
    executor_recycling_bars = filtered_data.loc[(
        filtered_data['scenario name'] ==
        "hydro_cppuddle_with_executor_recycling")]
    executor_recycling_bars.reset_index(inplace=True)
    complete_recycling_bars = filtered_data.loc[(
        filtered_data['scenario name'] ==
        "hydro_cppuddle_with_buffer_executor_recycling")]
    complete_recycling_bars.reset_index(inplace=True)
    implicit_aggregation_bars = filtered_data.loc[(
        filtered_data['scenario name'] ==
        "hydro_cppuddle_with_buffer_executor_recycling_more_executors")]
    implicit_aggregation_bars.reset_index(inplace=True)
    complete_aggregation_bars = filtered_data.loc[(
        filtered_data['scenario name'] ==
        "hydro_cppuddle_with_buffer_executor_recycling_more_executors_slices")]
    complete_aggregation_bars.reset_index(inplace=True)
    with_async_bars = filtered_data.loc[(
        filtered_data['scenario name'] ==
        "hydro_cppuddle_with_buffer_executor_recycling"
        "_more_executors_slices_async_fut")]
    with_async_bars.reset_index(inplace=True)

    baseline_bars.reset_index(inplace=True)
    complete_recycling_bars.reset_index(inplace=True)
    baseline_bars['overheads'] = baseline_bars['median time-per-timestep'] - complete_recycling_bars['median time-per-timestep']
    baseline_bars['aggregation'] = complete_recycling_bars['median time-per-timestep'] - with_async_bars['median time-per-timestep']
    baseline_bars['async'] = with_async_bars['median time-per-timestep']
    print(baseline_bars['async'])

    plt.figure(figsize=(10,4))
    ax = plt.gca()
    left = np.zeros(len(baseline_bars))
    br1 = np.arange(len(baseline_bars))
    bar0 = ax.barh(br1, baseline_bars['median time-per-timestep'],
           color='black', edgecolor='black', left=left,
           label='basic_runtime')
    bar1 = ax.barh(br1, baseline_bars['async'],
           color='green', edgecolor='black', left=left,
           label='basic_runtime')
    left += baseline_bars['async']
    bar2 = ax.barh(br1, baseline_bars['aggregation'],
           color='yellow', edgecolor='black', left=left,
           label='basic_runtime')
    left += baseline_bars['aggregation']
    bar3 = ax.barh(br1, baseline_bars['overheads'],
           color='red', edgecolor='black', left=left,
           label='basic_runtime')
    left += baseline_bars['async']
    ax.bar_label(bar0, padding=3, rotation='horizontal', fmt='Overall:\n%.0fms')
    ax.bar_label(bar1, label_type='center', rotation='vertical', fmt='%.0fms')
    ax.bar_label(bar2, label_type='center', fmt='Aggregation/\nInterleaving\n%.0fms')
    ax.bar_label(bar3, label_type='center', fmt='Overhead:\n%.0fms')
        
    labels = baseline_bars['cores'].unique()
    ax.set_yticks([r for r in range(len(labels))],
        labels, rotation='horizontal')
    ax.set_xlim([0, 3300])
    ax.set_ylabel("Number HPX worker threads")
    ax.set_xlabel("Median time-per-timestep in ms")
    ax.set_title("Costs of Overhead and Starvation\n(Used Scenario: Octo-Tiger Blast Benchmark, 512 Sub-Grids) \n(Used Hardware: Intel Icelake CPU / NVIDIA A100 GPU)")
    plt.savefig("cppuddle_fractions.pdf", format="pdf", bbox_inches="tight")
    #plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Process/Plot work aggregation runtime data')
    parser.add_argument('--filename', dest='filename',
                        action='store', required=True,
                        help='Filename of the runtime data')
    args = parser.parse_args()
    print(args.filename)

    raw_data_colnames = [
        'scenario name',
        'max level',
        'cores',
        'slices',
        'executors',
        'total time',
        'computation time',
        'number creation',
        'number allocations',
        'number deallocations',
        'number hydro launches',
        'number aggregated hydro launches',
        'number cuda hydro launches',
        'number cuda aggregated hydro launches',
        'min time-per-timestep',
        'max time-per-timestep',
        'median time-per-timestep',
        'average time-per-timestep',
        'list of times per timestep',
        ]
    print("Reading " + str(args.filename) + " ...")
    raw_data = pd.read_csv(
        args.filename,
        comment='#',
        names=raw_data_colnames,
        header=0,
        on_bad_lines='warn')
    # squelch matplotlib warnings...
    warnings.filterwarnings("ignore")
    print(raw_data)
    plot_multibar_cppuddle_features(raw_data)
    plot_fractions_bars(raw_data)
