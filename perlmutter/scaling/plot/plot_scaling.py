import pandas as pd
import warnings
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Process/Plot work aggregation runtime data')
    parser.add_argument('--filename', dest='filename',
                        action='store', required=True,
                        help='Filename of the runtime data')
    parser.add_argument('--number_leaf_subgrids', dest='number_leaf_subgrids',
                        action='store', required=True,
                        help='How many leaf subgrids were used?')
    parser.add_argument('--scenario_name', dest='scenario_name',
                        action='store', required=True,
                        help='Name of the scenario')
    parser.add_argument('--output_filename_scaling',
                        dest='output_filename_scaling',
                        action='store', required=True,
                        help='filename of the parallel efficiency plot')
    parser.add_argument('--output_filename_cppuddle',
                        dest='output_filename_cppuddle',
                        action='store', required=True,
                        help='filename of the cppuddle efficiency plot')
    args = parser.parse_args()
    print(args.filename)

    raw_data_colnames = [
        'future_type',
        'slices',
        'executors',
        'nodes',
        'processcount',
        'total time',
        'computation time',
        'number creation',
        'number allocations',
        'number deallocations',
        'number hydro launches',
        'number aggregated hydro launches',
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
    basic_scaling = raw_data[['processcount', 'total time',
                              'number hydro launches',
                              'number aggregated hydro launches',
                              'median time-per-timestep',
                              'number creation',
                              'number allocations',
                              ]]

    # Convert to ms
    basic_scaling["median time-per-timestep"] = \
        basic_scaling["median time-per-timestep"] * 1000
    basic_scaling["median time-per-timestep"] = \
        basic_scaling["median time-per-timestep"].round(0).astype(int)
    # Get speedup and parallel efficiency
    baseline_processes = basic_scaling['processcount'].min()
    localities_string = " HPX Localities"
    if baseline_processes == 1:
        localities_string = " HPX Locality"
    base_value_timestep = basic_scaling.loc[basic_scaling['processcount'] ==
                                            baseline_processes].iloc[0]['median time-per-timestep']
    basic_scaling["timestep speedup"] = (base_value_timestep /
                                         basic_scaling["median time-per-timestep"]).round(2)
    basic_scaling["timestep efficiency"] = (basic_scaling["timestep speedup"] /
                                            basic_scaling["processcount"] *
                                            baseline_processes * 100).round(2)
    # Convert string with times-per-timestep into array for boxplotting
    times_per_timestep = raw_data[['processcount', 'list of times per timestep']]
    list_of_times = []
    number_of_timesteps = 0
    for times in times_per_timestep["list of times per timestep"]:
        times = times.replace('"', '')
        times_split = times.split()
        float_list = []
        for i in times_split:
            float_list.append(float(i) * 1000)
        number_of_timesteps = len(float_list)
        list_of_times.append(float_list)
    times_per_timestep['box'] = list_of_times
    # Calculate aggregation rate
    basic_scaling["avg aggregation rate"] = (basic_scaling["number hydro launches"] /
                                             basic_scaling["number aggregated hydro launches"])
    basic_scaling["aggregation efficiency"] = (basic_scaling["avg aggregation rate"] / 8.0 * 100.0)

    basic_scaling["recycle efficiency"] = ((basic_scaling["number allocations"] - basic_scaling["number creation"]) /
                                             basic_scaling["number allocations"] * 100.0)
            

    print(basic_scaling)
    print(times_per_timestep)
    # ax = plt.gca()
    # ax.plot(
    #     basic_scaling["processcount"],
    #     basic_scaling["total time"],
    #     '-o',
    #     c='red',
    #     alpha=0.95,
    #     markeredgecolor='none')
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # plt.xticks(basic_scaling["processcount"], basic_scaling["processcount"])
    # plt.yticks(basic_scaling["total time"], basic_scaling["total time"])
    # plt.xlabel("Number of HPX localities (16 Cores, 1 A100 per Locality)")
    # plt.savefig("test.pdf", format="pdf", bbox_inches="tight")
    # plt.show()
    # plt.clf()
    plt.rcParams.update({'font.size': 7})
    # plt.figure().set_figwidth(8.6)
    # plt.figure().set_figwidth(9.6)
    plt.figure(0)
    ax = plt.gca()
    flierprops = dict(marker='o', markerfacecolor='orange', markersize=2,
                      linestyle='none')
    ax.boxplot(
        times_per_timestep.box.values.tolist(),  flierprops=flierprops)

    ax.set_yscale('log')
    # ax.set_xscale('log')
    label_list = []
    for label in basic_scaling["processcount"]:
        number_localities = int(label)
        number_cores = number_localities * 16
        number_gpus = number_localities * 1
        label = str(number_cores) + " Cores\n" + str(number_gpus) + " A100"
        label_list.append(label)
    ylabel_list = []
    for label in basic_scaling["median time-per-timestep"]:
        label = str(label) + " ms"
        ylabel_list.append(label)
    plt.yticks(basic_scaling["median time-per-timestep"], ylabel_list)
    ax.set_yticks(ticks=[], labels=[], minor=True)  # Reset minor ticks
    plt.xticks([i + 1 for i, _ in enumerate(label_list)], label_list)
    plt.xticks(rotation=0)
    ax.tick_params(axis='x', which='major', labelsize=6)
    plt.xlabel("Utilized Hardware (16 Cores, 1 A100 per HPX Locality)")
    plt.ylabel("Time-per-timestep in ms")
    plt.title("Perlmutter Scaling: " + str(args.scenario_name) +
              " Scenario with " + str(args.number_leaf_subgrids) +
              " Leaf Sub-Grids over " + str(number_of_timesteps) +
              " Timesteps")
    plot1 = ax.plot(
        [i + 1 for i, _ in enumerate(label_list)],
        basic_scaling["median time-per-timestep"],
        '-',
        c='orange',
        alpha=0.20,
        markeredgecolor='none', label='Median time-per-timestep')
    ax2 = ax.twinx()
    plot2 = ax2.plot(
        [i + 1 for i, _ in enumerate(label_list)],
        basic_scaling["timestep efficiency"],
        '--',
        c='gray',
        alpha=0.35,
        markeredgecolor='none',
        label=('Parallel Efficiency w.r.t. ' + str(baseline_processes) +
               localities_string))
    ax2.set_ylim([0, 100])
    plt.ylabel('Parallel Efficiency w.r.t. ' + str(baseline_processes) +
               localities_string)
    ylabel_list2 = []
    last_efficiency = 200
    for label in basic_scaling["timestep efficiency"]:
        tmp = label
        if last_efficiency - label < 5:
            label = ""
        else:
            label = str(label) + "%"
        last_efficiency = tmp
        ylabel_list2.append(label)
    plt.yticks(basic_scaling["timestep efficiency"], ylabel_list2)

    all_plots = plot1 + plot2
    ax.legend(all_plots, ["Median time-per-timestep",
                          "Parallel Efficiency w.r.t " +
                          str(baseline_processes) + localities_string],
              loc=0, prop={'size': 8})
    # ax2.legend(loc=0)

    plt.savefig(args.output_filename_scaling, format="pdf", bbox_inches="tight")
    #plt.show()
    plt.clf




    plt.figure(1)
    ax = plt.gca()
    flierprops = dict(marker='o', markerfacecolor='orange', markersize=2,
                      linestyle='none')
    ax.boxplot(
        times_per_timestep.box.values.tolist(),  flierprops=flierprops)

    ax.set_yscale('log')
    # ax.set_xscale('log')
    label_list = []
    for label in basic_scaling["processcount"]:
        number_localities = int(label)
        number_cores = number_localities * 16
        number_gpus = number_localities * 1
        label = str(number_cores) + " Cores\n" + str(number_gpus) + " A100"
        label_list.append(label)
    ylabel_list = []
    for label in basic_scaling["median time-per-timestep"]:
        label = str(label) + " ms"
        ylabel_list.append(label)
    plt.yticks(basic_scaling["median time-per-timestep"], ylabel_list)
    ax.set_yticks(ticks=[], labels=[], minor=True)  # Reset minor ticks
    plt.xticks([i + 1 for i, _ in enumerate(label_list)], label_list)
    plt.xticks(rotation=0)
    ax.tick_params(axis='x', which='major', labelsize=6)
    plt.xlabel("Utilized Hardware (16 Cores, 1 A100 per HPX Locality)")
    plt.ylabel("Time-per-timestep in ms")
    plt.title("Perlmutter CPPuddle Efficiencies: " + str(args.scenario_name) +
              " Scenario with " + str(args.number_leaf_subgrids) +
              " Leaf Sub-Grids over " + str(number_of_timesteps) +
              " Timesteps")
    plot1 = ax.plot(
        [i + 1 for i, _ in enumerate(label_list)],
        basic_scaling["median time-per-timestep"],
        '-',
        c='orange',
        alpha=0.20,
        markeredgecolor='none', label='Median time-per-timestep')
    ax2 = ax.twinx()
    plot2 = ax2.plot(
        [i + 1 for i, _ in enumerate(label_list)],
        basic_scaling["aggregation efficiency"],
        'o--',
        c='green',
        alpha=0.55, markersize=4,
        markeredgecolor='none')
    plot3 = ax2.plot(
        [i + 1 for i, _ in enumerate(label_list)],
        basic_scaling["recycle efficiency"],
        'o:',
        c='gray',
        alpha=0.55, markersize=4,
        markeredgecolor='none')
    ax2.set_ylim([0, 101])
    plt.ylabel('Efficiency in %')
    ylabel_list2 = []
    last_efficiency = 200
    for label in basic_scaling["aggregation efficiency"]:
        tmp = label
        if last_efficiency - label < 5:
            label = ""
        else:
            label = str(label) + "%"
        last_efficiency = tmp
        ylabel_list2.append(label)
    #plt.yticks(basic_scaling["timestep efficiency"], ylabel_list2)

    all_plots = plot1 + plot2 + plot3
    ax.legend(all_plots, ["Octo-Tiger: Median time-per-timestep",
                          "CPPuddle: Kernel Aggregation Efficiency in % ",
                          "CPPuddle: Buffer Recycle Efficiency in % "],
              loc=3, prop={'size': 8})
    # ax2.legend(loc=0)

    plt.savefig(args.output_filename_cppuddle, format="pdf", bbox_inches="tight")
    #plt.show()
    plt.clf
