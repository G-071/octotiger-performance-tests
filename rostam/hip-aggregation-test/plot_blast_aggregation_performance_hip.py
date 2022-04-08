import pandas as pd
import matplotlib.pyplot as plt


def plot_cpu_only_node_level_scaling(raw_data, result_filename):
    # Get host only data
    host_only_run = raw_data.loc[raw_data['Executors'] == 0]
    # Check invariants...
    if not (host_only_run['Max Aggregation Slices'] == 1).all():
        print("Error! Data format is wrong! Host-only run contains Max Slices > 1")
        exit(1)
    if not (host_only_run['Reconstruct Kernel Launches'] == 0).all():
        print("Error! Data format is wrong! Host-only run contains Reconstruct Kernel launches")
        exit(1)
    if not (host_only_run['Flux Kernel Launches'] == 0).all():
        print("Error! Data format is wrong! Host-only run contains Flux Kernel launches")
        exit(1)
    if not (host_only_run['Discs1 Kernel Launches'] == 0).all():
        print("Error! Data format is wrong! Host-only run contains Discs1 Kernel launches")
        exit(1)
    if not (host_only_run['Discs2 Kernel Launches'] == 0).all():
        print("Error! Data format is wrong! Host-only run contains Discs2 Kernel launches")
        exit(1)
    if not (host_only_run['Pre_Recon Kernel Launches'] == 0).all():
        print("Error! Data format is wrong! Host-only run contains Pre_Recon Kernel launches")
        exit(1)
    if not (host_only_run['Profiling Computation Time'] == 0.0).all():
        print("Error! Data format is wrong! Host-only run contains profiling time")
        exit(1)
    if not (host_only_run['Profiling Total Time'] == 0.0).all():
        print("Error! Data format is wrong! Host-only run contains profiling time")
        exit(1)
    # Create copy without the unnecessary columns
    host_only_run = host_only_run[[
        'Cores', 'Computation Time', 'Total Time']].copy()
    print(host_only_run)
    ax = plt.gca()
    ax.plot(
        host_only_run["Cores"],
        host_only_run["Total Time"],
        '-o',
        c='blue',
        alpha=0.95,
        markeredgecolor='none')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.title("Hydro-only: CPU-only Node-Level Scaling")
    plt.xlim(host_only_run["Cores"].min() - 0.1, host_only_run["Cores"].max() + 10)
    plt.xlabel("Number of Cores")
    plt.xticks(host_only_run["Cores"], host_only_run["Cores"])
    plt.ylim([0, host_only_run["Total Time"].max() + 10])
    plt.ylabel("Runtime in seconds")
    plt.yticks(host_only_run["Total Time"], host_only_run["Total Time"])
    plt.grid(True)
    plt.savefig(result_filename, format="pdf", bbox_inches="tight")
    plt.clf()
    return


def plot_gpu_only_node_level_scaling(raw_data, result_filename, with_cpu_only_plot=False):
    # Get host only data
    host_only_run = raw_data.loc[raw_data['Executors'] == 0]
    host_only_run = host_only_run[[
        'Cores', 'Computation Time', 'Total Time']].copy()
    # Get data for plain work aggregation via 128 executors
    non_aggregated_run = raw_data.loc[(raw_data['Executors'] == 128) & (
        raw_data['Max Aggregation Slices'] == 1)]
    non_aggregated_run = non_aggregated_run[[
        'Cores', 'Computation Time', 'Total Time']].copy()
    # Get data for explicit work aggregation with slices
    slice_aggregated_run = raw_data.loc[(raw_data['Executors'] == 1) & (
        raw_data['Max Aggregation Slices'] == 128)]
    slice_aggregated_run = slice_aggregated_run[[
        'Cores', 'Computation Time', 'Total Time']].copy()
    # Get data for combined work aggregation with 8 executors and 16 slices
    fully_aggregated_run = raw_data.loc[(raw_data['Executors'] == 8) & (
        raw_data['Max Aggregation Slices'] == 16)]
    fully_aggregated_run = fully_aggregated_run[[
        'Cores', 'Computation Time', 'Total Time']].copy()
    # Get data for combined work aggregation with 8 executors and 128 slices
    fully_aggregated_run2 = raw_data.loc[(raw_data['Executors'] == 8) & (
        raw_data['Max Aggregation Slices'] == 128)]
    fully_aggregated_run2 = fully_aggregated_run2[[
        'Cores', 'Computation Time', 'Total Time']].copy()
    print(non_aggregated_run)

    # Start plot...
    ax = plt.gca()
    if with_cpu_only_plot is True:
        ax.plot(
            host_only_run["Cores"],
            host_only_run["Total Time"],
            '-o',
            c='black',
            alpha=0.65,
            markeredgecolor='none',
            label='CPU-only')
    ax.plot(
        non_aggregated_run["Cores"],
        non_aggregated_run["Total Time"],
        '-o',
        c='red',
        alpha=0.95,
        markeredgecolor='none',
        label='Using 128 executors, 1 kernels per launch')
    ax.plot(
        slice_aggregated_run["Cores"],
        slice_aggregated_run["Total Time"],
        '-o',
        c='green',
        alpha=0.95,
        markeredgecolor='none',
        label='Using 1 executors, Up to 128 kernels aggregated per launch')
    ax.plot(
        fully_aggregated_run2["Cores"],
        fully_aggregated_run2["Total Time"],
        '-o',
        c='lightgreen',
        alpha=0.95,
        markeredgecolor='none',
        label='Using 8 executors, Up to 128 kernels aggregated per launch')
    ax.plot(
        fully_aggregated_run["Cores"],
        fully_aggregated_run["Total Time"],
        '-o',
        c='teal',
        alpha=0.95,
        markeredgecolor='none',
        label='Using 8 executors, Up to 16 kernels aggregated per launch')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.title("Hydro-only: GPU infrastructure scaling")
    plt.xlim(non_aggregated_run["Cores"].min() - 0.1,
             non_aggregated_run["Cores"].max() + 10)
    plt.xlabel("Number of Cores")
    plt.xticks(non_aggregated_run["Cores"], non_aggregated_run["Cores"])
    plt.ylabel("Runtime in seconds")
    if with_cpu_only_plot is True:
        plt.yticks([host_only_run["Total Time"].max(),
                    host_only_run["Total Time"].min(),
                    non_aggregated_run["Total Time"].max(),
                    non_aggregated_run["Total Time"].min(),
                    slice_aggregated_run["Total Time"].min(),
                    fully_aggregated_run2["Total Time"].min(),
                    fully_aggregated_run["Total Time"].min(),
                    fully_aggregated_run["Total Time"].max()],
                   [host_only_run["Total Time"].max(),
                    host_only_run["Total Time"].min(),
                    non_aggregated_run["Total Time"].max(),
                    non_aggregated_run["Total Time"].min(),
                    slice_aggregated_run["Total Time"].min(),
                    fully_aggregated_run2["Total Time"].min(),
                    fully_aggregated_run["Total Time"].min(),
                    fully_aggregated_run["Total Time"].max()])
    else:
        plt.yticks([non_aggregated_run["Total Time"].max(),
                    non_aggregated_run["Total Time"].min(),
                    slice_aggregated_run["Total Time"].min(),
                    fully_aggregated_run2["Total Time"].min(),
                    fully_aggregated_run["Total Time"].min(),
                    fully_aggregated_run["Total Time"].max()],
                   [non_aggregated_run["Total Time"].max(),
                       non_aggregated_run["Total Time"].min(),
                       slice_aggregated_run["Total Time"].min(),
                       fully_aggregated_run2["Total Time"].min(),
                       fully_aggregated_run["Total Time"].min(),
                       fully_aggregated_run["Total Time"].max()])
    ax.axhline(
        non_aggregated_run["Total Time"].max(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        fully_aggregated_run["Total Time"].max(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        fully_aggregated_run["Total Time"].min(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        fully_aggregated_run2["Total Time"].min(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        slice_aggregated_run["Total Time"].min(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        non_aggregated_run["Total Time"].min(),
        linestyle='--', alpha=0.25,
        color='k')
    if with_cpu_only_plot is True:
        ax.axhline(
            host_only_run["Total Time"].min(),
            linestyle='--', alpha=0.25,
            color='k')
        ax.axhline(
            host_only_run["Total Time"].max(),
            linestyle='--', alpha=0.25,
            color='k')
    plt.legend()
    plt.savefig(result_filename, format="pdf", bbox_inches="tight")
    plt.clf()
    return


def plot_kernel_aggregation_performance(raw_data, kernelname):
    # get performance for starved gpu (ie 1 executor)
    kernel_starved = raw_data.loc[(
        raw_data['Executors'] == 1) & (raw_data['Cores'] == 64)]
    kernel_starved = kernel_starved[[
        'Max Aggregation Slices', kernelname + ' Kernel Launches', kernelname + ' Kernel Avg Time']]
    kernel_starved['Avg Kernel Aggregation'] = kernel_starved.apply(
        lambda row: kernel_starved[kernelname + ' Kernel Launches'].values[0] /
        row[kernelname + ' Kernel Launches'],
        axis=1)
    kernel_starved[kernelname + ' Avg Subgrid Runtime'] = kernel_starved.apply(
        lambda row: row[kernelname + ' Kernel Avg Time'] / row['Avg Kernel Aggregation'], axis=1)
    kernel_starved['Aggregation Speedup'] = kernel_starved.apply(
        lambda row: kernel_starved[kernelname + ' Avg Subgrid Runtime'].values[0] /
        row[kernelname + ' Avg Subgrid Runtime'],
        axis=1)
    print(kernel_starved)

    # get performance for starved gpu (ie 64 executor)
    kernel_non_starved = raw_data.loc[(
        raw_data['Executors'] == 64) & (raw_data['Cores'] == 64)]
    kernel_non_starved = kernel_non_starved[[
        'Max Aggregation Slices', kernelname + ' Kernel Launches', kernelname + ' Kernel Avg Time']]
    kernel_non_starved['Avg Kernel Aggregation'] = kernel_non_starved.apply(
        lambda row: kernel_non_starved[kernelname + ' Kernel Launches'].values[0] /
        row[kernelname + ' Kernel Launches'],
        axis=1)
    kernel_non_starved[kernelname + ' Avg Subgrid Runtime'] = kernel_non_starved.apply(
        lambda row: row[kernelname + ' Kernel Avg Time'] / row['Avg Kernel Aggregation'], axis=1)
    kernel_non_starved['Aggregation Speedup'] = kernel_non_starved.apply(
        lambda row: kernel_non_starved[kernelname + ' Avg Subgrid Runtime'].values[0] /
        row[kernelname + ' Avg Subgrid Runtime'],
        axis=1)
    print(kernel_non_starved)

    ax = plt.gca()
    plot1 = ax.plot(
        kernel_starved["Max Aggregation Slices"],
        kernel_starved["Aggregation Speedup"],
        '-o',
        c='red',
        alpha=0.95,
        markeredgecolor='none',
        label='Avg speedup [Starved GPU]')
    plot2 = ax.plot(
        kernel_non_starved["Max Aggregation Slices"],
        kernel_non_starved["Aggregation Speedup"],
        '-o',
        c='orange',
        alpha=0.95,
        markeredgecolor='none',
        label='Avg speedup [Busy GPU]')
    ax.set_xscale('log')
    ax.set_title(
        "Hydro-only on AMD MI100: " + kernelname + " Kernel Aggregation Sub-Grid Speedup")
    ax.set_xlim(kernel_starved["Max Aggregation Slices"].min(
    ) - 0.1, kernel_starved["Max Aggregation Slices"].max() + 10)
    ax.set_xlabel(
        "Maximum allowed number of aggregated Subgrids per " + kernelname + " kernel launch")
    ax.set_xticks(
        ticks=kernel_starved["Max Aggregation Slices"],
        labels=kernel_starved["Max Aggregation Slices"])
    ax.set_ylim([0, kernel_starved["Aggregation Speedup"].max() + 1])
    ax.set_ylabel("Avg Speedup of the " + kernelname + " Kernel per Sub-grid")
    ax.set_yticks(ticks=[1,
                         2,
                         4,
                         8,
                         16,
                         kernel_starved["Aggregation Speedup"].max(),
                         kernel_non_starved["Aggregation Speedup"].max()],
                  labels=[1,
                          2,
                          4,
                          8,
                          16,
                          "" + str(round(kernel_starved["Aggregation Speedup"].max(),
                                         1)),
                          "" + str(round(kernel_non_starved["Aggregation Speedup"].max(),
                                         1))])
    ax.axhline(
        kernel_starved["Aggregation Speedup"].max(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        kernel_non_starved["Aggregation Speedup"].max(),
        linestyle='--', alpha=0.25,
        color='k')
    ax2 = ax.twinx()
    plot3 = ax2.plot(
        kernel_starved["Max Aggregation Slices"],
        kernel_starved["Avg Kernel Aggregation"],
        '--',
        c='red',
        alpha=0.35,
        markeredgecolor='none',
        label='Avg aggregation rate [Starved GPU]')
    plot4 = ax2.plot(
        kernel_non_starved["Max Aggregation Slices"],
        kernel_non_starved["Avg Kernel Aggregation"],
        '--',
        c='orange',
        alpha=0.35,
        markeredgecolor='none',
        label='Avg aggregation rate [Busy GPU]')
    ax2.set_ylabel("Average Aggregation Rate (Subgrids per Launch)")
    all_plots = plot1 + plot2 + plot3 + plot4
    labels = [l.get_label() for l in all_plots]
    ax.legend(all_plots, labels, loc=0)
    plt.savefig(kernelname + '-Speedup-Per-Subgrid.pdf',
                format="pdf", bbox_inches="tight")
    plt.clf()


    ax = plt.gca()
    plot1 = ax.plot(
        kernel_starved["Max Aggregation Slices"],
        kernel_starved[kernelname + " Avg Subgrid Runtime"],
        '-o',
        c='red',
        alpha=0.95,
        markeredgecolor='none',
        label='Avg runtime [Starved GPU]')
    plot2 = ax.plot(
        kernel_non_starved["Max Aggregation Slices"],
        kernel_non_starved[kernelname + " Avg Subgrid Runtime"],
        '-o',
        c='orange',
        alpha=0.95,
        markeredgecolor='none',
        label='Avg runtime [Busy GPU]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(
        "Hydro-only on AMD MI100: " + kernelname + " Kernel Avg Subgrid Runtime")
    ax.set_xlim(kernel_starved["Max Aggregation Slices"].min(
    ) - 0.1, kernel_starved["Max Aggregation Slices"].max() + 10)
    ax.set_xlabel(
        "Maximum allowed number of aggregated Subgrids per " + kernelname + " kernel launch")
    ax.set_xticks(
        ticks=kernel_starved["Max Aggregation Slices"],
        labels=kernel_starved["Max Aggregation Slices"])
    ax.set_ylim(
        [0, kernel_non_starved[kernelname + " Avg Subgrid Runtime"].max() +
            kernel_non_starved[kernelname + " Avg Subgrid Runtime"].max()/2])
    ax.set_ylabel("Avg runtime of the " + kernelname + " Kernel per Sub-grid")
    ax.set_yticks(ticks=[kernel_starved[kernelname + " Avg Subgrid Runtime"].max(),
                         kernel_non_starved[kernelname + " Avg Subgrid Runtime"].max(),
                         kernel_starved[kernelname + " Avg Subgrid Runtime"].min(),
                         kernel_non_starved[kernelname + " Avg Subgrid Runtime"].min()],
                  labels=[str(round(kernel_starved[kernelname + " Avg Subgrid Runtime"].max() / 1000,
                                    3)) + str('\u03bcs'),
                          str(round(kernel_non_starved[kernelname + " Avg Subgrid Runtime"].max() / 1000,
                                    3)) + str('\u03bcs'),
                          str(round(kernel_starved[kernelname + " Avg Subgrid Runtime"].min() / 1000,
                                    3)) + str('\u03bcs'),
                          str(round(kernel_non_starved[kernelname + " Avg Subgrid Runtime"].min() / 1000,
                                    3)) + str('\u03bcs')])
    ax.axhline(
        kernel_starved[kernelname + " Avg Subgrid Runtime"].max(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        kernel_non_starved[kernelname + " Avg Subgrid Runtime"].max(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        kernel_starved[kernelname + " Avg Subgrid Runtime"].min(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        kernel_non_starved[kernelname + " Avg Subgrid Runtime"].min(),
        linestyle='--', alpha=0.25,
        color='k')
    ax2 = ax.twinx()
    plot3 = ax2.plot(
        kernel_starved["Max Aggregation Slices"],
        kernel_starved["Avg Kernel Aggregation"],
        '--',
        c='red',
        alpha=0.35,
        markeredgecolor='none',
        label='Avg aggregation rate [Starved GPU]')
    plot4 = ax2.plot(
        kernel_non_starved["Max Aggregation Slices"],
        kernel_non_starved["Avg Kernel Aggregation"],
        '--',
        c='orange',
        alpha=0.35,
        markeredgecolor='none',
        label='Avg aggregation rate [Busy GPU]')
    ax2.set_ylabel("Average Aggregation Rate (Subgrids per Launch)")
    all_plots = plot1 + plot2 + plot3 + plot4
    labels = [l.get_label() for l in all_plots]
    ax.legend(all_plots, labels, loc=0)
    plt.savefig(kernelname + '-Runtime-Per-Subgrid.pdf',
                format="pdf", bbox_inches="tight")
    plt.clf()

    ax = plt.gca()
    plot1 = ax.plot(
        kernel_starved["Max Aggregation Slices"],
        kernel_starved[kernelname + " Kernel Avg Time"],
        '-o',
        c='red',
        alpha=0.95,
        markeredgecolor='none',
        label='Avg runtime [Starved GPU]')
    plot2 = ax.plot(
        kernel_non_starved["Max Aggregation Slices"],
        kernel_non_starved[kernelname + " Kernel Avg Time"],
        '-o',
        c='orange',
        alpha=0.95,
        markeredgecolor='none',
        label='Avg runtime [Busy GPU]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(
        "Hydro-only on AMD MI100: Avg Aggregated " + kernelname + " Kernel Runtime")
    ax.set_xlim(kernel_starved["Max Aggregation Slices"].min(
    ) - 0.1, kernel_starved["Max Aggregation Slices"].max() + 10)
    ax.set_xlabel(
        "Maximum allowed number of aggregated Subgrids per " + kernelname + " kernel launch")
    ax.set_xticks(
        ticks=kernel_starved["Max Aggregation Slices"],
        labels=kernel_starved["Max Aggregation Slices"])
    ax.set_ylim([kernel_starved[kernelname + " Kernel Avg Time"].min() -
                 kernel_starved[kernelname + " Kernel Avg Time"].min()/10,
                 kernel_non_starved[kernelname + " Kernel Avg Time"].max() +
                 kernel_non_starved[kernelname + " Kernel Avg Time"].max()/2])
    ax.set_ylabel("Avg runtime of the " + kernelname + " Kernel per kernel launch")
    ax.set_yticks(ticks=[kernel_starved[kernelname + " Kernel Avg Time"].max(),
                         kernel_non_starved[kernelname + " Kernel Avg Time"].max(),
                         kernel_starved[kernelname + " Kernel Avg Time"].min(),
                         kernel_non_starved[kernelname + " Kernel Avg Time"].min()],
                  labels=[str(round(kernel_starved[kernelname + " Kernel Avg Time"].max() / 1000,
                                    3)) + str('\u03bcs'),
                          str(round(kernel_non_starved[kernelname + " Kernel Avg Time"].max() / 1000,
                                    3)) + str('\u03bcs'),
                          str(round(kernel_starved[kernelname + " Kernel Avg Time"].min() / 1000,
                                    3)) + str('\u03bcs'),
                          str(round(kernel_non_starved[kernelname + " Kernel Avg Time"].min() / 1000,
                                    3)) + str('\u03bcs')],
                  minor=False)
    ax.set_yticks(ticks=[], labels=[], minor=True)  # Reset minor ticks
    ax.axhline(
        kernel_starved[kernelname + " Kernel Avg Time"].max(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(kernel_non_starved[kernelname + " Kernel Avg Time"].max(),
               linestyle='--', alpha=0.25,
               color='k')
    ax.axhline(
        kernel_starved[kernelname + " Kernel Avg Time"].min(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        kernel_non_starved[kernelname + " Kernel Avg Time"].min(),
        linestyle='--', alpha=0.25,
        color='k')
    ax2 = ax.twinx()
    plot3 = ax2.plot(
        kernel_starved["Max Aggregation Slices"],
        kernel_starved["Avg Kernel Aggregation"],
        '--',
        c='red',
        alpha=0.35,
        markeredgecolor='none',
        label='Avg aggregation rate [Starved GPU]')
    plot4 = ax2.plot(
        kernel_non_starved["Max Aggregation Slices"],
        kernel_non_starved["Avg Kernel Aggregation"],
        '--',
        c='orange',
        alpha=0.35,
        markeredgecolor='none',
        label='Avg aggregation rate [Busy GPU]')
    ax2.set_ylabel("Average Aggregation Rate (Subgrids per Launch)")
    all_plots = plot1 + plot2 + plot3 + plot4
    labels = [l.get_label() for l in all_plots]
    ax.legend(all_plots, labels, loc=0)
    plt.savefig(kernelname + '-Runtime-Per-Launch.pdf',
                format="pdf", bbox_inches="tight")
    plt.clf()
    return


default_filepath = "sample-datasets/"
default_filename = \
    "blast_aggregation_test_hip_cpuamr_3_15_2022-04-01_10:09:14_kamand0.rostam.cct.lsu.edu_LOG.txt"
raw_data_colnames = [
    'Cores',
    'Executors',
    'Max Aggregation Slices',
    'Computation Time',
    'Total Time',
    'Reconstruct Kernel Launches',
    'Reconstruct Kernel Avg Time',
    'Flux Kernel Launches',
    'Flux Kernel Avg Time',
    'Discs1 Kernel Launches',
    'Discs1 Kernel Avg Time',
    'Discs2 Kernel Launches',
    'Discs2 Kernel Avg Time',
    'Pre_Recon Kernel Launches',
    'Pre_Recon Kernel Avg Time',
    'Profiling Computation Time',
    'Profiling Total Time']
raw_data = pd.read_csv(
    default_filepath + default_filename,
    comment='#',
    names=raw_data_colnames,
    header=None)
# Check raw data invariants
print(raw_data)
if not (raw_data['Cores'] >= 1).all():
    print("Error! Data format is wrong! Data entry with less than 1 core found")
    exit(1)
if not (raw_data['Cores'] <= 128).all():
    print("Error! Data format is wrong! Data entry with more than 128 cores found")
    exit(1)
if not (raw_data['Executors'] >= 0).all():
    print("Error! Data format is wrong! Data entry with less than 0 Executors found")
    exit(1)
if not (raw_data['Executors'] <= 128).all():
    print("Error! Data format is wrong! Data entry with more than 128 Executors found")
    exit(1)
if not (raw_data['Max Aggregation Slices'] >= 1).all():
    print("Error! Data format is wrong! Data entry with less than 1 Aggregation Slice found")
    exit(1)
if not (raw_data['Max Aggregation Slices'] <= 128).all():
    print("Error! Data format is wrong! Data entry with more than 128 Slices found")
    exit(1)
if (raw_data['Computation Time'] < 0.0).all():
    print("Error! Data format is wrong! Negative execution time found!")
    exit(1)
if (raw_data['Total Time'] < 0.0).all():
    print("Error! Data format is wrong! Negative execution time found!")
    exit(1)
if (raw_data['Profiling Computation Time'] < 0.0).all():
    print("Error! Data format is wrong! Negative execution time found!")
    exit(1)
if (raw_data['Profiling Total Time'] < 0.0).all():
    print("Error! Data format is wrong! Negative execution time found!")
    exit(1)
if (raw_data['Reconstruct Kernel Avg Time'] < 0.0).all():
    print("Error! Data format is wrong! Negative reconstruct execution time found!")
    exit(1)
if (raw_data['Flux Kernel Avg Time'] < 0.0).all():
    print("Error! Data format is wrong! Negative flux execution time found!")
    exit(1)
if (raw_data['Discs1 Kernel Avg Time'] < 0.0).all():
    print("Error! Data format is wrong! Negative discs1 execution time found!")
    exit(1)
if (raw_data['Discs2 Kernel Avg Time'] < 0.0).all():
    print("Error! Data format is wrong! Negative discs2 execution time found!")
    exit(1)
if (raw_data['Pre_Recon Kernel Avg Time'] < 0.0).all():
    print("Error! Data format is wrong! Negative Pre_Recon execution time found!")
    exit(1)

plot_cpu_only_node_level_scaling(raw_data, 'cpu-only-nodelevel-scaling.pdf')
plot_gpu_only_node_level_scaling(raw_data, 'gpu-only-nodelevel-scaling.pdf')
plot_gpu_only_node_level_scaling(raw_data, 'cpu-gpu-nodelevel-scaling.pdf', True)
plot_kernel_aggregation_performance(raw_data, 'Reconstruct')
plot_kernel_aggregation_performance(raw_data, 'Flux')
plot_kernel_aggregation_performance(raw_data, 'Discs1')
plot_kernel_aggregation_performance(raw_data, 'Discs2')
plot_kernel_aggregation_performance(raw_data, 'Pre_Recon')
