import pandas as pd
import matplotlib.pyplot as plt
import warnings


def plot_cpu_only_node_level_scaling(raw_data, result_filename):
    # Get host only data
    host_only_run = raw_data.loc[raw_data['Executors'] == 0]
    # Create copy without the unnecessary columns
    host_only_run = host_only_run[[
        'Cores', 'Computation Time', 'Total Time']].copy()
    ax = plt.gca()
    ax.plot(
        host_only_run["Cores"],
        host_only_run["Computation Time"],
        '-o',
        c='blue',
        alpha=0.95,
        markeredgecolor='none')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.title("Hydro-only: CPU-only Node-Level Scaling")
    plt.xlim(host_only_run["Cores"].min() - 0.1,
             host_only_run["Cores"].max() + 10)
    plt.xlabel("Number of Cores")
    plt.xticks(host_only_run["Cores"], host_only_run["Cores"])
    plt.ylim([0, host_only_run["Computation Time"].max() + 10])
    plt.ylabel("Runtime in seconds")
    plt.yticks(host_only_run["Computation Time"],
               host_only_run["Computation Time"])
    plt.grid(True)
    plt.savefig(result_filename, format="pdf", bbox_inches="tight")
    plt.clf()
    return


def plot_gpu_only_node_level_scaling(
        raw_data,
        result_filename, gpu_name,
        with_cpu_only_plot=False):

    max_executors = raw_data['Executors'].max()
    max_slices = raw_data['Max Aggregation Slices'].max()
    assert max_slices >= 16
    assert max_executors >= 8
    # Get host only data
    host_only_run = raw_data.loc[raw_data['Executors'] == 0]
    host_only_run = host_only_run[[
        'Cores', 'Computation Time', 'Total Time']].copy()
    # Get data for plain work aggregation via max executors
    non_aggregated_run = raw_data.loc[(raw_data['Executors'] == max_executors) &
                                      (raw_data['Max Aggregation Slices'] == 1)]
    non_aggregated_run = non_aggregated_run[[
        'Cores', 'Computation Time', 'Total Time']].copy()
    # Get data for explicit work aggregation with slices
    slice_aggregated_run = raw_data.loc[(raw_data['Executors'] == 1) & (
        raw_data['Max Aggregation Slices'] == max_slices)]
    slice_aggregated_run = slice_aggregated_run[[
        'Cores', 'Computation Time', 'Total Time']].copy()
    # Get data for combined work aggregation with 8 executors and 16 slices
    fully_aggregated_run = raw_data.loc[(raw_data['Executors'] == 8) & (
        raw_data['Max Aggregation Slices'] == 16)]
    fully_aggregated_run = fully_aggregated_run[[
        'Cores', 'Computation Time', 'Total Time']].copy()
    # Get data for combined work aggregation with 8 executors and max slices
    fully_aggregated_run2 = raw_data.loc[(raw_data['Executors'] == 8) & (
        raw_data['Max Aggregation Slices'] == max_slices)]
    fully_aggregated_run2 = fully_aggregated_run2[[
        'Cores', 'Computation Time', 'Total Time']].copy()

    # Start plot...
    ax = plt.gca()
    if with_cpu_only_plot is True:
        ax.plot(
            host_only_run["Cores"],
            host_only_run["Computation Time"],
            '-o',
            c='black',
            alpha=0.65,
            markeredgecolor='none',
            label='CPU-only')
    ax.plot(
        non_aggregated_run["Cores"],
        non_aggregated_run["Computation Time"],
        '-o',
        c='red',
        alpha=0.95,
        markeredgecolor='none',
        label='Using ' + str(max_executors) +
        ' executors, 1 kernels per launch')
    ax.plot(
        slice_aggregated_run["Cores"],
        slice_aggregated_run["Computation Time"],
        '-o',
        c='green',
        alpha=0.95,
        markeredgecolor='none',
        label='Using 1 executors, Up to ' + str(max_slices) +
        ' kernels aggregated per launch')
    ax.plot(
        fully_aggregated_run2["Cores"],
        fully_aggregated_run2["Computation Time"],
        '-o',
        c='lightgreen',
        alpha=0.95,
        markeredgecolor='none',
        label='Using 8 executors, Up to ' + str(max_slices) +
        ' kernels aggregated per launch')
    ax.plot(
        fully_aggregated_run["Cores"],
        fully_aggregated_run["Computation Time"],
        '-o',
        c='teal',
        alpha=0.95,
        markeredgecolor='none',
        label='Using 8 executors, Up to 16 kernels aggregated per launch')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.title("Hydro-only: " + str(gpu_name) + " GPU API scaling over Cores")
    plt.xlim(non_aggregated_run["Cores"].min() - 0.1,
             non_aggregated_run["Cores"].max() + 10)
    plt.xlabel("Number of Cores")
    plt.xticks(non_aggregated_run["Cores"], non_aggregated_run["Cores"])
    plt.ylabel("Runtime in seconds")
    if with_cpu_only_plot is True:
        plt.yticks([host_only_run["Computation Time"].max(),
                    host_only_run["Computation Time"].min(),
                    non_aggregated_run["Computation Time"].max(),
                    non_aggregated_run["Computation Time"].min(),
                    slice_aggregated_run["Computation Time"].min(),
                    fully_aggregated_run2["Computation Time"].min(),
                    fully_aggregated_run["Computation Time"].min(),
                    fully_aggregated_run["Computation Time"].max()],
                   [host_only_run["Computation Time"].max(),
                    host_only_run["Computation Time"].min(),
                    non_aggregated_run["Computation Time"].max(),
                    non_aggregated_run["Computation Time"].min(),
                    slice_aggregated_run["Computation Time"].min(),
                    fully_aggregated_run2["Computation Time"].min(),
                    fully_aggregated_run["Computation Time"].min(),
                    fully_aggregated_run["Computation Time"].max()])
    else:
        plt.yticks([non_aggregated_run["Computation Time"].max(),
                    non_aggregated_run["Computation Time"].min(),
                    slice_aggregated_run["Computation Time"].min(),
                    fully_aggregated_run2["Computation Time"].min(),
                    fully_aggregated_run["Computation Time"].min(),
                    fully_aggregated_run["Computation Time"].max()],
                   [non_aggregated_run["Computation Time"].max(),
                       non_aggregated_run["Computation Time"].min(),
                       slice_aggregated_run["Computation Time"].min(),
                       fully_aggregated_run2["Computation Time"].min(),
                       fully_aggregated_run["Computation Time"].min(),
                       fully_aggregated_run["Computation Time"].max()])
    ax.axhline(
        non_aggregated_run["Computation Time"].max(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        fully_aggregated_run["Computation Time"].max(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        fully_aggregated_run["Computation Time"].min(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        fully_aggregated_run2["Computation Time"].min(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        slice_aggregated_run["Computation Time"].min(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        non_aggregated_run["Computation Time"].min(),
        linestyle='--', alpha=0.25,
        color='k')
    if with_cpu_only_plot is True:
        ax.axhline(
            host_only_run["Computation Time"].min(),
            linestyle='--', alpha=0.25,
            color='k')
        ax.axhline(
            host_only_run["Computation Time"].max(),
            linestyle='--', alpha=0.25,
            color='k')
    plt.legend()
    plt.savefig(result_filename, format="pdf", bbox_inches="tight")
    plt.clf()
    return


def plot_scaling_over_slices(raw_data, result_filename, gpu_name):

    max_cores = raw_data['Cores'].max()
    max_number_executors = raw_data['Executors'].max()
    # max_slices = raw_data['Max Aggregation Slices'].max()

    assert max_number_executors >= 8
    single_executor = raw_data.loc[(
        raw_data['Executors'] == 1) & (raw_data['Cores'] == max_cores)]
    max_executors = raw_data.loc[(
        raw_data['Executors'] == max_number_executors) &
        (raw_data['Cores'] == max_cores)]
    balanced_executors = raw_data.loc[(
        raw_data['Executors'] == 8) &
        (raw_data['Cores'] == max_cores)]

    ax = plt.gca()
    ax.plot(
        single_executor["Max Aggregation Slices"],
        single_executor["Computation Time"],
        '-o',
        c='red',
        alpha=0.95,
        markeredgecolor='none',
        label='Runtime [1 Executor]')
    ax.plot(
        max_executors["Max Aggregation Slices"],
        max_executors["Computation Time"],
        '-o',
        c='orange',
        alpha=0.95,
        markeredgecolor='none',
        label='Runtime  [' + str(max_number_executors) + ' Executors]')
    ax.plot(
        balanced_executors["Max Aggregation Slices"],
        balanced_executors["Computation Time"],
        '-o',
        c='green',
        alpha=0.95,
        markeredgecolor='none',
        label='Runtime  [' + str(8) + ' Executors]')
    ax.set_yscale('log')
    ax.set_yticks(
        ticks=[single_executor["Computation Time"].min(),
               single_executor["Computation Time"].max() - 1,
               max_executors["Computation Time"].min(),
               max_executors["Computation Time"].max(),
               balanced_executors["Computation Time"].min() - 0.05,
               balanced_executors["Computation Time"].max(),
               ],
        labels=[single_executor["Computation Time"].min(),
                single_executor["Computation Time"].max(),
                max_executors["Computation Time"].min(),
                max_executors["Computation Time"].max(),
                balanced_executors["Computation Time"].min(),
                balanced_executors["Computation Time"].max(),
                ], minor=False)
    ax.set_yticks(ticks=[], labels=[], minor=True)  # Reset minor ticks
    ax.set_xscale('log')
    ax.set_xticks(
        ticks=single_executor["Max Aggregation Slices"],
        labels=single_executor["Max Aggregation Slices"])
    ax.set_xlabel(
        "Maximum allowed number of aggregated Subgrids per kernel launch ")
    ax.set_ylabel(
        "Runtime in s")
    ax.set_title(
        "Hydro-only on " + str(gpu_name) + ": " +
        " Runtime over maximum aggregation slices with " +
        str(max_cores) + " cores")

    ax.axhline(
        single_executor["Computation Time"].max(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        single_executor["Computation Time"].min(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        max_executors["Computation Time"].max(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        max_executors["Computation Time"].min(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        balanced_executors["Computation Time"].max(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        balanced_executors["Computation Time"].min(),
        linestyle='--', alpha=0.25,
        color='k')

    ax.legend()
    plt.savefig(result_filename,
                format="pdf", bbox_inches="tight")
    plt.clf()
    return


def plot_scaling_over_executors(raw_data, result_filename, gpu_name):

    max_cores = raw_data['Cores'].max()
    max_number_slices = raw_data['Max Aggregation Slices'].max()

    assert max_number_slices >= 16
    single_slice = raw_data.loc[(
        raw_data['Max Aggregation Slices'] == 1) &
        (raw_data['Cores'] == max_cores) & (raw_data['Executors'] >= 1)]
    max_slices = raw_data.loc[(
        raw_data['Max Aggregation Slices'] == max_number_slices) &
        (raw_data['Cores'] == max_cores)]
    balanced_slices = raw_data.loc[(
        raw_data['Max Aggregation Slices'] == 16) &
        (raw_data['Cores'] == max_cores)]

    ax = plt.gca()
    ax.plot(
        single_slice["Executors"],
        single_slice["Computation Time"],
        '-o',
        c='red',
        alpha=0.95,
        markeredgecolor='none',
        label='Runtime [1 Slice]')
    ax.plot(
        max_slices["Executors"],
        max_slices["Computation Time"],
        '-o',
        c='orange',
        alpha=0.95,
        markeredgecolor='none',
        label='Runtime  [' + str(max_number_slices) + ' Slices]')
    ax.plot(
        balanced_slices["Executors"],
        balanced_slices["Computation Time"],
        '-o',
        c='green',
        alpha=0.95,
        markeredgecolor='none',
        label='Runtime  [' + str(16) + ' Slices]')
    ax.set_yscale('log')
    ax.set_yticks(
        ticks=[single_slice["Computation Time"].min(),
               single_slice["Computation Time"].max() - 1,
               max_slices["Computation Time"].min(),
               max_slices["Computation Time"].max(),
               balanced_slices["Computation Time"].min() - 0.05,
               balanced_slices["Computation Time"].max(),
               ],
        labels=[single_slice["Computation Time"].min(),
                single_slice["Computation Time"].max(),
                max_slices["Computation Time"].min(),
                max_slices["Computation Time"].max(),
                balanced_slices["Computation Time"].min(),
                balanced_slices["Computation Time"].max(),
                ], minor=False)
    ax.set_yticks(ticks=[], labels=[], minor=True)  # Reset minor ticks
    ax.set_xscale('log')
    ax.set_xticks(
        ticks=single_slice["Executors"],
        labels=single_slice["Executors"])
    ax.set_xlabel(
        "GPU Executors")
    ax.set_ylabel(
        "Runtime in s")
    ax.set_title(
        "Hydro-only on " + str(gpu_name) + ": " +
        " Runtime over executors with " +
        str(max_cores) + " cores")

    ax.axhline(
        single_slice["Computation Time"].max(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        single_slice["Computation Time"].min(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        max_slices["Computation Time"].max(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        max_slices["Computation Time"].min(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        balanced_slices["Computation Time"].max(),
        linestyle='--', alpha=0.25,
        color='k')
    ax.axhline(
        balanced_slices["Computation Time"].min(),
        linestyle='--', alpha=0.25,
        color='k')

    ax.legend()
    plt.savefig(result_filename,
                format="pdf", bbox_inches="tight")
    plt.clf()
    return


def find_best_runs(raw_data):
    best_compute_time_index = \
        raw_data['Computation Time'].idxmin()

    raw_implicit_only = raw_data.loc[(
        raw_data['Max Aggregation Slices'] == 1)]
    best_implicit_only_compute_time_index = \
        raw_implicit_only['Computation Time'].idxmin()

    raw_explicit_only = raw_data.loc[(
        raw_data['Executors'] == 1)]
    best_explicit_only_compute_time_index = \
        raw_explicit_only['Computation Time'].idxmin()

    raw_host_only = raw_data.loc[(
        raw_data['Executors'] == 0)]
    best_host_only_compute_time_index = \
        raw_host_only['Computation Time'].idxmin()

    print()
    print("Best cpu-only Computation Time configuration: ")
    print(raw_data.loc[best_host_only_compute_time_index, :])
    print()

    print()
    print("Best implicit aggregation Computation Time configuration: ")
    print(raw_data.loc[best_implicit_only_compute_time_index, :])
    print()
    print("Best explicit aggregation Computation Time configuration: ")
    print(raw_data.loc[best_explicit_only_compute_time_index, :])
    print()
    print("Best Overall Computation Time configuration: ")
    print(raw_data.loc[best_compute_time_index, :])
    print()
    print("Speedup best run over best cpu-only run: " + str(
          raw_data.loc[best_host_only_compute_time_index, "Computation Time"] /
          raw_data.loc[best_compute_time_index, "Computation Time"]))
    print()
    print("Combined Work Aggregation Speedup (over implicit-only): " + str(
          raw_data.loc[best_implicit_only_compute_time_index, "Computation Time"] /
          raw_data.loc[best_compute_time_index, "Computation Time"]))
    print("Combined Work Aggregation Speedup (over explicit-only): " + str(
          raw_data.loc[best_explicit_only_compute_time_index, "Computation Time"] /
          raw_data.loc[best_compute_time_index, "Computation Time"]))
    print()
    print("Best Compute Time: " +
            str(raw_data.loc[best_compute_time_index, "Computation Time"]) + "s")

    return


def plot_kernel_aggregation_performance(raw_data, kernelname, gpu_name):
    # get performance for starved gpu (ie 1 executor)
    max_cores = raw_data['Cores'].max()
    max_executors = raw_data['Executors'].max()
    kernel_starved = raw_data.loc[(
        raw_data['Executors'] == 1) & (raw_data['Cores'] == max_cores)]
    kernel_starved = kernel_starved[['Max Aggregation Slices',
                                     kernelname + ' Kernel Launches',
                                     kernelname + ' Kernel Avg Time']]
    kernel_starved['Avg Kernel Aggregation'] = kernel_starved.apply(
        lambda row: kernel_starved[kernelname + ' Kernel Launches'].values[0] /
        row[kernelname + ' Kernel Launches'],
        axis=1)
    kernel_starved[kernelname + ' Avg Subgrid Runtime'] = kernel_starved.apply(
        lambda row: row[kernelname +
                        ' Kernel Avg Time'] / row['Avg Kernel Aggregation'],
        axis=1)
    kernel_starved['Aggregation Speedup'] = kernel_starved.apply(
        lambda row: kernel_starved[kernelname +
                                   ' Avg Subgrid Runtime'].values[0] /
        row[kernelname + ' Avg Subgrid Runtime'],
        axis=1)

    # get performance for busy gpu (ie max executor)
    kernel_non_starved = raw_data.loc[(
        raw_data['Executors'] == max_executors) &
        (raw_data['Cores'] == max_cores)]
    kernel_non_starved = kernel_non_starved[[
        'Max Aggregation Slices', kernelname + ' Kernel Launches', kernelname +
        ' Kernel Avg Time']]
    kernel_non_starved['Avg Kernel Aggregation'] = kernel_non_starved.apply(
        lambda row: kernel_non_starved[kernelname +
                                       ' Kernel Launches'].values[0] /
        row[kernelname + ' Kernel Launches'],
        axis=1)
    kernel_non_starved[kernelname +
                       ' Avg Subgrid Runtime'] = kernel_non_starved.apply(
        lambda row: row[kernelname + ' Kernel Avg Time'] /
        row['Avg Kernel Aggregation'], axis=1)
    kernel_non_starved['Aggregation Speedup'] = kernel_non_starved.apply(
        lambda row: kernel_non_starved[kernelname +
                                       ' Avg Subgrid Runtime'].values[0] /
        row[kernelname + ' Avg Subgrid Runtime'],
        axis=1)

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
        "Hydro-only on " + str(gpu_name) + ": " +
        kernelname +
        " Kernel Aggregation Sub-Grid Speedup")
    ax.set_xlim(kernel_starved["Max Aggregation Slices"].min(
    ) - 0.1, kernel_starved["Max Aggregation Slices"].max() + 10)
    ax.set_xlabel(
        "Maximum allowed number of aggregated Subgrids per " +
        kernelname +
        " kernel launch")
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
                          "" + str(round(kernel_starved["Aggregation Speedup"].
                                   max(),
                                         1)),
                          "" + str(round(
                              kernel_non_starved["Aggregation Speedup"].max(), 1
                              ))])
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
        "Hydro-only on " + str(gpu_name) + ": " +
        kernelname +
        " Kernel Avg Subgrid Runtime")
    ax.set_xlim(kernel_starved["Max Aggregation Slices"].min(
    ) - 0.1, kernel_starved["Max Aggregation Slices"].max() + 10)
    ax.set_xlabel(
        "Maximum allowed number of aggregated Subgrids per " +
        kernelname +
        " kernel launch")
    ax.set_xticks(
        ticks=kernel_starved["Max Aggregation Slices"],
        labels=kernel_starved["Max Aggregation Slices"])
    ax.set_ylim(
        [0, kernel_non_starved[kernelname + " Avg Subgrid Runtime"].max() +
            kernel_non_starved[kernelname + " Avg Subgrid Runtime"].max() / 2])
    ax.set_ylabel("Avg runtime of the " + kernelname + " Kernel per Sub-grid")
    ax.set_yticks(ticks=[kernel_starved[kernelname +
                                        " Avg Subgrid Runtime"].max(),
                         kernel_non_starved[kernelname +
                                            " Avg Subgrid Runtime"].max(),
                         kernel_starved[kernelname +
                                        " Avg Subgrid Runtime"].min(),
                         kernel_non_starved[kernelname +
                                            " Avg Subgrid Runtime"].min()],
                  labels=[str(round(kernel_starved[kernelname +
                                                   " Avg Subgrid Runtime"].max()
                                    / 1000,
                                    3)) + str('\u03bcs'),
                          str(round(kernel_non_starved[kernelname +
                              " Avg Subgrid Runtime"].max() / 1000,
                                    3)) + str('\u03bcs'),
                          str(round(kernel_starved[kernelname +
                              " Avg Subgrid Runtime"].min() / 1000,
                                    3)) + str('\u03bcs'),
                          str(round(kernel_non_starved[kernelname +
                              " Avg Subgrid Runtime"].min() / 1000,
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
        "Hydro-only on " + str(gpu_name) + ": Avg Aggregated " +
        kernelname +
        " Kernel Runtime")
    ax.set_xlim(kernel_starved["Max Aggregation Slices"].min(
    ) - 0.1, kernel_starved["Max Aggregation Slices"].max() + 10)
    ax.set_xlabel(
        "Maximum allowed number of aggregated Subgrids per " +
        kernelname +
        " kernel launch")
    ax.set_xticks(
        ticks=kernel_starved["Max Aggregation Slices"],
        labels=kernel_starved["Max Aggregation Slices"])
    ax.set_ylim([kernel_starved[kernelname + " Kernel Avg Time"].min() -
                 kernel_starved[kernelname + " Kernel Avg Time"].min() / 10,
                 kernel_non_starved[kernelname + " Kernel Avg Time"].max() +
                 kernel_non_starved[kernelname + " Kernel Avg Time"].max() / 2])
    ax.set_ylabel(
        "Avg runtime of the " +
        kernelname +
        " Kernel per kernel launch")
    ax.set_yticks(ticks=[kernel_starved[kernelname + " Kernel Avg Time"].max(),
                         kernel_non_starved[kernelname + " Kernel Avg Time"].
                         max(),
                         kernel_starved[kernelname + " Kernel Avg Time"].
                         min(),
                         kernel_non_starved[kernelname + " Kernel Avg Time"].
                         min()],
                  labels=[str(round(kernel_starved[kernelname +
                                                   " Kernel Avg Time"].max()
                                    / 1000,
                                    3)) + str('\u03bcs'),
                          str(round(kernel_non_starved[kernelname +
                              " Kernel Avg Time"].max() / 1000,
                                    3)) + str('\u03bcs'),
                          str(round(kernel_starved[kernelname +
                              " Kernel Avg Time"].min() / 1000,
                                    3)) + str('\u03bcs'),
                          str(round(kernel_non_starved[kernelname +
                              " Kernel Avg Time"].min() / 1000,
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


def check_aggregation_dataset_invariants(raw_data):
    # raw data invariants:
    if not (raw_data['Cores'] >= 1).all():
        print("Error! Data format is wrong!")
        print("Data entry with less than 1 core found")
        exit(1)
    if not (raw_data['Cores'] <= 128).all():
        print("Error! Data format is wrong!")
        print("Data entry with more than 128 cores found")
        exit(1)
    if not (raw_data['Executors'] >= 0).all():
        print("Error! Data format is wrong!")
        print("Data entry with less than 0 Executors found")
        exit(1)
    if not (raw_data['Executors'] <= 128).all():
        print("Error! Data format is wrong!")
        print("Data entry with more than 128 Executors found")
        exit(1)
    if not (raw_data['Max Aggregation Slices'] >= 1).all():
        print("Error! Data format is wrong!")
        print("Data entry with less than 1 Aggregation Slice found")
        exit(1)
    if not (raw_data['Max Aggregation Slices'] <= 128).all():
        print("Error! Data format is wrong!")
        print("Data entry with more than 128 Slices found")
        exit(1)
    if (raw_data['Computation Time'] < 0.0).all():
        print("Error! Data format is wrong!")
        print("Negative execution time found!")
        exit(1)
    if (raw_data['Total Time'] < 0.0).all():
        print("Error! Data format is wrong!")
        print("Negative execution time found!")
        exit(1)
    if (raw_data['Profiling Computation Time'] < 0.0).all():
        print("Error! Data format is wrong!")
        print("Negative execution time found!")
        exit(1)
    if (raw_data['Profiling Total Time'] < 0.0).all():
        print("Error! Data format is wrong!")
        print("Negative execution time found!")
        exit(1)
    if (raw_data['Reconstruct Kernel Avg Time'] < 0.0).all():
        print("Error! Data format is wrong!")
        print("Negative reconstruct execution time found!")
        exit(1)
    if (raw_data['Flux Kernel Avg Time'] < 0.0).all():
        print("Error! Data format is wrong!")
        print("Negative flux execution time found!")
        exit(1)
    if (raw_data['Discs1 Kernel Avg Time'] < 0.0).all():
        print("Error! Data format is wrong!")
        print("Negative discs1 execution time found!")
        exit(1)
    if (raw_data['Discs2 Kernel Avg Time'] < 0.0).all():
        print("Error! Data format is wrong!")
        print("Negative discs2 execution time found!")
        exit(1)
    if (raw_data['Pre_Recon Kernel Avg Time'] < 0.0).all():
        print("Error! Data format is wrong!")
        print("Negative Pre_Recon execution time found!")
        exit(1)
    # Host only invariants
    host_only_run = raw_data.loc[raw_data['Executors'] == 0]
    # Check invariants...
    if not (host_only_run['Max Aggregation Slices'] == 1).all():
        print("Error! Data format is wrong!")
        print("Host-only run contains Max Slices > 1")
        exit(1)
    if not (host_only_run['Reconstruct Kernel Launches'] == 0).all():
        print("Error! Data format is wrong!")
        print("Host-only run contains Reconstruct Kernel launches")
        exit(1)
    if not (host_only_run['Flux Kernel Launches'] == 0).all():
        print("Error! Data format is wrong!")
        print("Host-only run contains Flux Kernel launches")
        exit(1)
    if not (host_only_run['Discs1 Kernel Launches'] == 0).all():
        print("Error! Data format is wrong!")
        print("Host-only run contains Discs1 Kernel launches")
        exit(1)
    if not (host_only_run['Discs2 Kernel Launches'] == 0).all():
        print("Error! Data format is wrong!")
        print("Host-only run contains Discs2 Kernel launches")
        exit(1)
    if not (host_only_run['Pre_Recon Kernel Launches'] == 0).all():
        print("Error! Data format is wrong!")
        print("Host-only run contains Pre_Recon Kernel launches")
        exit(1)
    if not (host_only_run['Profiling Computation Time'] == 0.0).all():
        print("Error! Data format is wrong!")
        print("Host-only run contains profiling time")
        exit(1)
    if not (host_only_run['Profiling Total Time'] == 0.0).all():
        print("Error! Data format is wrong!")
        print("Host-only run contains profiling time")
        exit(1)


if __name__ == "__main__":
    default_filepath = "sample-datasets/"
    default_filename_hip = \
        "blast_aggregation_test_hip_cpuamr" + \
        "_3_15_2022-04-01_10:09:14_kamand0.rostam.cct.lsu.edu_LOG.txt"
    default_filename_cuda = \
        "blast_aggregation_test_cuda_cpuamr" + \
        "_3_15_2022-03-30_19:44:10_toranj0.rostam.cct.lsu.edu_LOG.txt"
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
    print("Reading ...")
    raw_data = pd.read_csv(
        default_filepath + default_filename_cuda,
        comment='#',
        names=raw_data_colnames,
        header=None)
    # Check raw data invariants
    print("Check dataset correctness...")
    check_aggregation_dataset_invariants(raw_data)

    # squelch matplotlib warnings...
    warnings.filterwarnings("ignore")

    # plot overviews
    # gpu_name = "AMD MI100"
    gpu_name = "NVIDIA A100"
    print("Plot cpu-only scaling over cores...")
    plot_cpu_only_node_level_scaling(raw_data, 'cpu-only-nodelevel-scaling.pdf')
    print("Plot gpu-only scaling over cores...")
    plot_gpu_only_node_level_scaling(raw_data, 'gpu-only-nodelevel-scaling.pdf',
                                     gpu_name)
    print("Plot gpu-only and gpu-only scaling over cores...")
    plot_gpu_only_node_level_scaling(
        raw_data, 'cpu-gpu-nodelevel-scaling.pdf', gpu_name, True)
    print("Plot runtime over aggregation slices...")
    plot_scaling_over_slices(raw_data, 'slices_scaling.pdf', gpu_name)
    print("Plot runtime over GPU executors...")
    plot_scaling_over_executors(raw_data, 'executors_scaling.pdf', gpu_name)

    # plot kernel performance
    print("Plot aggregation speedup for Reconstruct kernel...")
    plot_kernel_aggregation_performance(raw_data, 'Reconstruct', gpu_name)
    print("Plot aggregation speedup for Flux kernel...")
    plot_kernel_aggregation_performance(raw_data, 'Flux', gpu_name)
    print("Plot aggregation speedup for Discs1 kernel...")
    plot_kernel_aggregation_performance(raw_data, 'Discs1', gpu_name)
    print("Plot aggregation speedup for Discs2 kernel...")
    plot_kernel_aggregation_performance(raw_data, 'Discs2', gpu_name)
    print("Plot aggregation speedup for Pre_Recon kernel...")
    plot_kernel_aggregation_performance(raw_data, 'Pre_Recon', gpu_name)
    print("Find best runs...")
    find_best_runs(raw_data)
    exit(0)
