import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings
import argparse

def plot_hydro_speedup(raw_data_icelake, raw_data_epyc, raw_data_arm):
    raw_data_icelake = raw_data_icelake.round(2)
    raw_data_epyc = raw_data_epyc.round(2)
    raw_data_arm = raw_data_arm.round(2)
    print(raw_data_icelake, raw_data_epyc, raw_data_arm)
    raw_data_icelake.replace("Octo-Tiger Compute time", "Compute Phase", inplace=True)
    raw_data_icelake.replace("Octo-Tiger Total time", "Entire Application", inplace=True)
    raw_data_icelake.replace("kernel hydro_reconstruct kokkos", "Reconstruct", inplace=True)
    raw_data_icelake.replace("kernel hydro_flux kokkos", "Flux", inplace=True)
    raw_data_icelake.replace("kernel multipole-rho kokkos", "Multipole Rho", inplace=True)
    raw_data_icelake.replace("kernel multipole-non-rho kokkos", "Multipole", inplace=True)
    raw_data_icelake.replace("kernel multipole-root-rho kokkos", "Root Multipole Rho", inplace=True)
    raw_data_icelake.replace("kernel multipole-root-non-rho kokkos", "Root Multipole", inplace=True)
    raw_data_icelake.replace("kernel p2p kokkos", "Monopole", inplace=True)

    raw_data_epyc.replace("Octo-Tiger Compute time", "Compute Phase", inplace=True)
    raw_data_epyc.replace("Octo-Tiger Total time", "Entire Application", inplace=True)
    raw_data_epyc.replace("kernel hydro_reconstruct kokkos", "Reconstruct", inplace=True)
    raw_data_epyc.replace("kernel hydro_flux kokkos", "Flux", inplace=True)
    raw_data_epyc.replace("kernel multipole-rho kokkos", "Multipole Rho", inplace=True)
    raw_data_epyc.replace("kernel multipole-non-rho kokkos", "Multipole", inplace=True)
    raw_data_epyc.replace("kernel multipole-root-rho kokkos", "Root Multipole Rho", inplace=True)
    raw_data_epyc.replace("kernel multipole-root-non-rho kokkos", "Root Multipole", inplace=True)
    raw_data_epyc.replace("kernel p2p kokkos", "Monopole", inplace=True)

    raw_data_arm.replace("Octo-Tiger Compute time", "Compute Phase", inplace=True)
    raw_data_arm.replace("Octo-Tiger Total time", "Entire Application", inplace=True)
    raw_data_arm.replace("kernel hydro_reconstruct kokkos", "Reconstruct", inplace=True)
    raw_data_arm.replace("kernel hydro_flux kokkos", "Flux", inplace=True)
    raw_data_arm.replace("kernel multipole-rho kokkos", "Multipole Rho", inplace=True)
    raw_data_arm.replace("kernel multipole-non-rho kokkos", "Multipole", inplace=True)
    raw_data_arm.replace("kernel multipole-root-rho kokkos", "Root Multipole Rho", inplace=True)
    raw_data_arm.replace("kernel multipole-root-non-rho kokkos", "Root Multipole", inplace=True)
    raw_data_arm.replace("kernel p2p kokkos", "Monopole", inplace=True)

    matplotlib.rcParams.update({'font.size': 16})
    labels=["Intel Icelake", "AMD Epyc", "ARM64Fx"]
    #labels=["Intel Icelake", "AMD Epyc"]
    cores = 1
    bars_legacy_one_core = []
    bars_legacy_one_core.append(raw_data_icelake.loc[(raw_data_icelake['CORES'] == cores) &
        (raw_data_icelake['NAME'] == "Compute Phase") &
        (raw_data_icelake['SIMD LIBRARY'] == "LEGACY") &
        (raw_data_icelake['SIMD EXTENSION'] == "LEGACY")].iloc[0]['MEAN'])
    bars_legacy_one_core.append(raw_data_epyc.loc[(raw_data_epyc['CORES'] == cores) &
        (raw_data_epyc['NAME'] == "Compute Phase") &
        (raw_data_epyc['SIMD LIBRARY'] == "LEGACY") &
        (raw_data_epyc['SIMD EXTENSION'] == "LEGACY")].iloc[0]['MEAN'])
    bars_legacy_one_core.append(raw_data_arm.loc[(raw_data_arm['CORES'] == cores) &
        (raw_data_arm['NAME'] == "Compute Phase") &
        (raw_data_arm['SIMD LIBRARY'] == "LEGACY") &
        (raw_data_arm['SIMD EXTENSION'] == "LEGACY")].iloc[0]['MEAN'])

    bars_scalar_one_core = []
    bars_scalar_one_core.append(raw_data_icelake.loc[(raw_data_icelake['CORES'] == cores) &
        (raw_data_icelake['NAME'] == "Compute Phase") &
        (raw_data_icelake['SIMD LIBRARY'] == "KOKKOS") &
        (raw_data_icelake['SIMD EXTENSION'] == "SCALAR")].iloc[0]['MEAN'])
    bars_scalar_one_core.append(raw_data_epyc.loc[(raw_data_epyc['CORES'] == cores) &
        (raw_data_epyc['NAME'] == "Compute Phase") &
        (raw_data_epyc['SIMD LIBRARY'] == "KOKKOS") &
        (raw_data_epyc['SIMD EXTENSION'] == "SCALAR")].iloc[0]['MEAN'])
    bars_scalar_one_core.append(raw_data_arm.loc[(raw_data_arm['CORES'] == cores) &
        (raw_data_arm['NAME'] == "Compute Phase") &
        (raw_data_arm['SIMD LIBRARY'] == "STD") &
        (raw_data_arm['SIMD EXTENSION'] == "SCALAR")].iloc[0]['MEAN'])

    bars_simd_one_core = []
    bars_simd_one_core.append(raw_data_icelake.loc[(raw_data_icelake['CORES'] == cores) &
        (raw_data_icelake['NAME'] == "Compute Phase") &
        (raw_data_icelake['SIMD LIBRARY'] == "KOKKOS") &
        (raw_data_icelake['SIMD EXTENSION'] == "AVX512")].iloc[0]['MEAN'])
    bars_simd_one_core.append(raw_data_epyc.loc[(raw_data_epyc['CORES'] == cores) &
        (raw_data_epyc['NAME'] == "Compute Phase") &
        (raw_data_epyc['SIMD LIBRARY'] == "KOKKOS") &
        (raw_data_epyc['SIMD EXTENSION'] == "AVX")].iloc[0]['MEAN'])
    bars_simd_one_core.append(raw_data_arm.loc[(raw_data_arm['CORES'] == cores) &
        (raw_data_arm['NAME'] == "Compute Phase") &
        (raw_data_arm['SIMD LIBRARY'] == "STD") &
        (raw_data_arm['SIMD EXTENSION'] == "SVE")].iloc[0]['MEAN'])

    bars_legacy_all_cores = []
    bars_legacy_all_cores.append(raw_data_icelake.loc[(raw_data_icelake['CORES'] == 64) &
        (raw_data_icelake['NAME'] == "Compute Phase") &
        (raw_data_icelake['SIMD LIBRARY'] == "LEGACY") &
        (raw_data_icelake['SIMD EXTENSION'] == "LEGACY")].iloc[0]['MEAN'])
    bars_legacy_all_cores.append(raw_data_epyc.loc[(raw_data_epyc['CORES'] == 64) &
        (raw_data_epyc['NAME'] == "Compute Phase") &
        (raw_data_epyc['SIMD LIBRARY'] == "LEGACY") &
        (raw_data_epyc['SIMD EXTENSION'] == "LEGACY")].iloc[0]['MEAN'])
    bars_legacy_all_cores.append(raw_data_arm.loc[(raw_data_arm['CORES'] == 48) &
        (raw_data_arm['NAME'] == "Compute Phase") &
        (raw_data_arm['SIMD LIBRARY'] == "LEGACY") &
        (raw_data_arm['SIMD EXTENSION'] == "LEGACY")].iloc[0]['MEAN'])

    bars_scalar_all_cores = []
    bars_scalar_all_cores.append(raw_data_icelake.loc[(raw_data_icelake['CORES'] == 64) &
        (raw_data_icelake['NAME'] == "Compute Phase") &
        (raw_data_icelake['SIMD LIBRARY'] == "KOKKOS") &
        (raw_data_icelake['SIMD EXTENSION'] == "SCALAR")].iloc[0]['MEAN'])
    bars_scalar_all_cores.append(raw_data_epyc.loc[(raw_data_epyc['CORES'] == 64) &
        (raw_data_epyc['NAME'] == "Compute Phase") &
        (raw_data_epyc['SIMD LIBRARY'] == "KOKKOS") &
        (raw_data_epyc['SIMD EXTENSION'] == "SCALAR")].iloc[0]['MEAN'])
    bars_scalar_all_cores.append(raw_data_arm.loc[(raw_data_arm['CORES'] == 48) &
        (raw_data_arm['NAME'] == "Compute Phase") &
        (raw_data_arm['SIMD LIBRARY'] == "STD") &
        (raw_data_arm['SIMD EXTENSION'] == "SCALAR")].iloc[0]['MEAN'])

    bars_simd_all_cores = []
    bars_simd_all_cores.append(raw_data_icelake.loc[(raw_data_icelake['CORES'] == 64) &
        (raw_data_icelake['NAME'] == "Compute Phase") &
        (raw_data_icelake['SIMD LIBRARY'] == "KOKKOS") &
        (raw_data_icelake['SIMD EXTENSION'] == "AVX512")].iloc[0]['MEAN'])
    bars_simd_all_cores.append(raw_data_epyc.loc[(raw_data_epyc['CORES'] == 64) &
        (raw_data_epyc['NAME'] == "Compute Phase") &
        (raw_data_epyc['SIMD LIBRARY'] == "KOKKOS") &
        (raw_data_epyc['SIMD EXTENSION'] == "AVX")].iloc[0]['MEAN'])
    bars_simd_all_cores.append(raw_data_arm.loc[(raw_data_arm['CORES'] == 48) &
        (raw_data_arm['NAME'] == "Compute Phase") &
        (raw_data_arm['SIMD LIBRARY'] == "STD") &
        (raw_data_arm['SIMD EXTENSION'] == "SVE")].iloc[0]['MEAN'])

    print(bars_legacy_one_core, bars_scalar_one_core, bars_simd_one_core)
    print(bars_legacy_all_cores, bars_scalar_all_cores, bars_simd_all_cores)


    plt.clf()
    plt.cla()
    barWidth = 0.30
    br1 = np.arange(len(labels))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    #fig = plt.figure()
    ax_one_core = plt.gca()
    #ax = fig.add_axes([0,0,1,1])
    bar1=ax_one_core.bar(br1, bars_legacy_one_core, color = 'gray', width = barWidth, edgecolor ='black', label ='Legacy')
    bar2=ax_one_core.bar(br2, bars_scalar_one_core, color = 'blue', width = barWidth, edgecolor ='grey', label ='SCALAR')
    bar3=ax_one_core.bar(br3, bars_simd_one_core, color = 'red', width = barWidth, edgecolor ='grey', label = 'SIMD')

    ax_one_core.set_xticks([r + 2.5*barWidth/2 for r in range(len(labels))],
        labels, rotation='horizontal')
    ax_one_core.set_ylabel("Hydro Computation Time in s")
    ax_one_core.set_xlabel("Compute Node")
    plt.title("Hydro-Only Scenario using one CPU Core", fontsize = 16)
    plt.legend(loc=4, prop={'size': 15})
    ax_one_core.bar_label(bar1, padding=3, rotation='vertical')
    ax_one_core.bar_label(bar2, padding=3, rotation='vertical')
    ax_one_core.bar_label(bar3, padding=3, rotation='vertical')
    ax_one_core.set_ylim([0, 1250])
    plt.savefig("hydro_one_core.pdf", format="pdf", bbox_inches="tight")

    plt.clf()
    plt.cla()
    barWidth = 0.20
    br1 = np.arange(len(labels))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    #fig = plt.figure()
    ax_all_cores = plt.gca()
    #ax = fig.add_axes([0,0,1,1])
    bar1=ax_all_cores.bar(br1, bars_legacy_all_cores, color = 'gray', width = barWidth, edgecolor ='black', label ='Legacy')
    bar2=ax_all_cores.bar(br2, bars_scalar_all_cores, color = 'blue', width = barWidth, edgecolor ='grey', label ='SCALAR')
    bar3=ax_all_cores.bar(br3, bars_simd_all_cores, color = 'red', width = barWidth, edgecolor ='grey', label = 'SIMD')

    ax_all_cores.set_xticks([r + 2.5*barWidth/2 for r in range(len(labels))],
        labels, rotation='horizontal')
    ax_all_cores.set_ylabel("Hydro Computation Time in s")
    ax_all_cores.set_xlabel("Compute Node")
    ax_all_cores.bar_label(bar1, padding=3, rotation='vertical')
    ax_all_cores.bar_label(bar2, padding=3, rotation='vertical')
    ax_all_cores.bar_label(bar3, padding=3, rotation='vertical')
    ax_all_cores.set_ylim([0, 33])
    plt.title("Hydro-Only Scenario using 64 Cores (48 on ARM64FX) ", fontsize = 16)
    plt.legend(loc=4, prop={'size': 15})
    plt.savefig("hydro_all_cores.pdf", format="pdf", bbox_inches="tight")

def plot_cpu_bars(raw_data, title_string, result_filename, simd_key, cores):
    print(raw_data)
    print(raw_data["NAME"].unique())

    raw_data.replace("Octo-Tiger Compute time", "Compute Phase", inplace=True)
    raw_data.replace("Octo-Tiger Total time", "Entire Application", inplace=True)
    raw_data.replace("kernel hydro_reconstruct kokkos", "Reconstruct", inplace=True)
    raw_data.replace("kernel hydro_flux kokkos", "Flux", inplace=True)
    raw_data.replace("kernel multipole-rho kokkos", "Multipole Rho", inplace=True)
    raw_data.replace("kernel multipole-non-rho kokkos", "Multipole", inplace=True)
    raw_data.replace("kernel multipole-root-rho kokkos", "Root Multipole Rho", inplace=True)
    raw_data.replace("kernel multipole-root-non-rho kokkos", "Root Multipole", inplace=True)
    raw_data.replace("kernel p2p kokkos", "Monopole", inplace=True)

    unique_names = raw_data["NAME"].unique()
    min_cores=1
    max_cores=raw_data['CORES'].max();

    matplotlib.rcParams.update({'font.size': 16})

    
    bars_scalar = []
    bars_simd = []

    bars_scalar_kokkos = []
    bars_scalar_std = []
    bars_simd_kokkos = []
    bars_simd_std = []

    bars_scalar_kokkos_all_cores = []
    bars_scalar_std_all_cores = []
    bars_simd_kokkos_all_cores = []
    bars_simd_std_all_cores = []
    labels = []

    for name in unique_names:
        if name != "Compute Phase":
            labels.append(name)
            bars_scalar_kokkos.append(raw_data.loc[(raw_data['CORES'] == cores) &
                (raw_data['NAME'] == name) &
                (raw_data['SIMD LIBRARY'] == "KOKKOS") &
                (raw_data['SIMD EXTENSION'] == "SCALAR")].iloc[0]['MEAN'])
            if simd_key != "SVE" :
                bars_simd_kokkos.append(bars_scalar_kokkos[-1]/raw_data.loc[(raw_data['CORES'] == cores) &
                    (raw_data['NAME'] == name) &
                    (raw_data['SIMD LIBRARY'] == "KOKKOS") &
                    (raw_data['SIMD EXTENSION'] == simd_key)].iloc[0]['MEAN'])
            
            bars_scalar_std.append(bars_scalar_kokkos[-1]/raw_data.loc[(raw_data['CORES'] == cores) &
                (raw_data['NAME'] == name) &
                (raw_data['SIMD LIBRARY'] == "STD") &
                (raw_data['SIMD EXTENSION'] == "SCALAR")].iloc[0]['MEAN'])
            bars_simd_std.append(bars_scalar_kokkos[-1]/raw_data.loc[(raw_data['CORES'] == cores) &
                (raw_data['NAME'] == name) &
                (raw_data['SIMD LIBRARY'] == "STD") &
                (raw_data['SIMD EXTENSION'] == simd_key)].iloc[0]['MEAN'])
            bars_scalar_kokkos[-1] = 1


    barWidth = 0.20
    br1 = np.arange(len(labels))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    #fig = plt.figure()
    ax = plt.gca()
    #ax = fig.add_axes([0,0,1,1])
    ax.bar(br1, bars_scalar_kokkos, color = 'blue', width = barWidth, edgecolor ='grey', label ='KOKKOS SCALAR [Baseline] ')
    ax.bar(br2, bars_scalar_std, color = 'cyan', width = barWidth, edgecolor ='grey', label ='Speedup STD SCALAR')
    if simd_key != "SVE" :
        ax.bar(br3, bars_simd_kokkos, color = 'red', width = barWidth, edgecolor ='grey', label = 'Speedup KOKKOS ' + simd_key + "")
        ax.bar(br4, bars_simd_std, color = 'orange', width = barWidth, edgecolor ='grey', label = 'Speedup STD ' + simd_key + "")
    else:
        ax.bar(br3, bars_simd_std, color = 'orange', width = barWidth, edgecolor ='grey', label = 'Speedup STD ' + simd_key + "")
    ax.set_xticks([r + 2.5*barWidth/2 for r in range(len(labels))],
        labels, rotation='vertical')
    ax.set_ylabel("Speedup w.r.t KOKKOS SCALAR ")
    ax.set_xlabel("Measured Octo-Tiger component")
    plt.title(title_string, fontsize = 16)
    plt.legend(loc=0, prop={'size': 15})
    plt.savefig(result_filename, format="pdf", bbox_inches="tight")
    #plt.show()

def plot_cpu_only_node_level_scaling(data_scalar, data_simd, title_string, result_filename, simd_lib, simd_key):
    # Create copy without the unnecessary columns
    scalar_run = data_scalar[[
        'CORES', 'MEAN']].copy()
    base_value = scalar_run.loc[scalar_run['CORES'] == 1].iloc[0]['MEAN']
    scalar_run["SPEEDUP"] = base_value / scalar_run["MEAN"]
    scalar_run["EFFICIENCY"] = scalar_run["SPEEDUP"] / scalar_run["CORES"] * 100
    scalar_run = scalar_run.round(2)
    print(scalar_run)


    matplotlib.rcParams.update({'font.size': 18})
    simd_run = data_simd[[
        'CORES', 'MEAN']].copy()
    base_value = simd_run.loc[simd_run['CORES'] == 1].iloc[0]['MEAN']
    simd_run["SPEEDUP"] = base_value / simd_run["MEAN"]
    simd_run["EFFICIENCY"] = simd_run["SPEEDUP"] / simd_run["CORES"] * 100
    simd_run = simd_run.round(2)
    print(simd_run)

    scalar_color='blue'
    simd_color='red'
    if simd_lib == "STD":
        scalar_color='cyan'
        simd_color='orange'

    ax = plt.gca()
    plot1 = ax.plot(
        scalar_run["CORES"],
        scalar_run["MEAN"],
        '--o',
        c=scalar_color,
        alpha=0.99,
        markeredgecolor='black',
        label='Computation Time [Scalar]')

    plot2 = ax.plot(
        simd_run["CORES"],
        simd_run["MEAN"],
        '--o',
        c=simd_color,
        alpha=0.99,
        markeredgecolor='black',
        label='Computation Time [' + str(simd_key)+ ']')

    ax.axhline(
        scalar_run['MEAN'].min(),
        linestyle=':', alpha=0.25,
        color='k')
    # ax.axhline(
    #     scalar_run['MEAN'].max(),
    #     linestyle=':', alpha=0.25,
    #     color='k')
    ax.axhline(
        simd_run['MEAN'].min(),
        linestyle=':', alpha=0.25,
        color='k')
    # ax.axhline(
    #     simd_run['MEAN'].max(),
    #     linestyle=':', alpha=0.25,
    #     color='k')

    ax2=ax.twinx()
    plot3 = ax2.plot(
        scalar_run["CORES"],
        scalar_run["EFFICIENCY"],
        ':',
        c=scalar_color,
        alpha=0.85,
        markeredgecolor='none',
        label='Parallel Efficiency [Scalar]')
    plot4 = ax2.plot(
        simd_run["CORES"],
        simd_run["EFFICIENCY"],
        ':',
        c=simd_color,
        alpha=0.85,
        markeredgecolor='none',
        label='Parallel Efficiency [' + str(simd_key)+ ']')
    ax2.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xscale('log')
    # cpu_name = data_scalar.loc[data_scalar['CORES'] == 1].iloc[0]['CPU']
    # plt.title(str("Node-Level Scaling on ") + cpu_name)
    plt.title(title_string, fontsize = 18)
    ax.set_xlim(scalar_run["CORES"].min() - 0.04,
             scalar_run["CORES"].max() + 2)
    ax.set_xlabel("Number of Cores")
    ax.set_xticks(scalar_run["CORES"], scalar_run["CORES"])
    ax.set_ylim([0, scalar_run["MEAN"].max() + 20])
    ax2.set_ylim([0, 130])
    ax.set_ylabel("Runtime in seconds")
    ax2.set_ylabel("Parallel Efficiency in %")

    ax.set_yticks(ticks=[scalar_run["MEAN"].max(),
                         simd_run["MEAN"].max(),
                         scalar_run["MEAN"].min(),
                         simd_run["MEAN"].min()],
                  labels=[scalar_run["MEAN"].max(),
                         simd_run["MEAN"].max(),
                         scalar_run["MEAN"].min(),
                         simd_run["MEAN"].min()])
    ax2.set_yticks(ticks=[10,20,30,40,50,60,70,80,90,100],
                  labels=[10,20,30,40,50,60,70,80,90,100])

    all_plots = plot1 + plot2 + plot3 + plot4
    labels = [l.get_label() for l in all_plots]
    ax.legend(all_plots, labels, loc=0, prop={'size': 17})
    #plt.grid(True)
    plt.savefig(result_filename, format="pdf", bbox_inches="tight")
    plt.clf()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Process/Plot work aggregation runtime data')
    parser.add_argument('--filename', dest='filename', action='store',
                        help='Filename of the runtime data')
    parser.add_argument('--title_prefix', dest='title_prefix', action='store',
                        help='Filename of the runtime data')
    parser.add_argument('--output_prefix', dest='output_prefix', action='store',
                        help='Filename of the runtime data')
    parser.add_argument('--simd_key', dest='simd_key', action='store',
                        help='Filename of the runtime data')
    args = parser.parse_args()
    print(args.filename)

    raw_data_colnames = [
        'CPU',
        'SIMD LIBRARY',
        'SIMD EXTENSION',
        'CORES',
        'NAME',
        'CALLS',
        'SUMMED',
        'MEAN']
    print("Reading " + str(args.filename) + " ...")
    raw_data = pd.read_csv(
        args.filename,
        comment='#',
        names=raw_data_colnames,
        header=None,
        on_bad_lines = 'warn')
    # squelch matplotlib warnings...
    warnings.filterwarnings("ignore")

    raw_data.replace("Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz", "Intel Xeon Platinum 8358", inplace=True)
    raw_data.replace("AMD EPYC 7H12 64-Core Processor", "AMD EPYC 7H12", inplace=True)
    print(raw_data)
    print(raw_data['NAME'])
    cpu_name = raw_data.loc[raw_data['CORES'] == 1].iloc[0]['CPU']

    compute_time_scaling = raw_data.loc[raw_data['NAME'] ==
                                        "Octo-Tiger Compute time"]

    # plot node level scaling with kokkos
    if args.simd_key != "SVE" :
        title_string = args.title_prefix + "on " + cpu_name + ":\n Node-Level Scaling with KOKKOS SIMD"
        print(compute_time_scaling)
        compute_time_scaling_kokkos = compute_time_scaling.loc[compute_time_scaling['SIMD LIBRARY'] ==
                                            "KOKKOS"]
        compute_time_scaling_kokkos_scalar = compute_time_scaling_kokkos.loc[compute_time_scaling_kokkos['SIMD EXTENSION'] ==
                                            "SCALAR"]
        compute_time_scaling_kokkos_avx = compute_time_scaling_kokkos.loc[compute_time_scaling_kokkos['SIMD EXTENSION'] ==
                                            args.simd_key]
        plot_cpu_only_node_level_scaling(compute_time_scaling_kokkos_scalar, compute_time_scaling_kokkos_avx, title_string, args.output_prefix + "kokkos_node_level_scaling.pdf", "KOKKOS", args.simd_key)

    # plot node level scaling with std simd
    title_string = args.title_prefix + "on " + cpu_name + ":\n Node-Level Scaling with STD SIMD"
    compute_time_scaling_std = compute_time_scaling.loc[compute_time_scaling['SIMD LIBRARY'] ==
                                        "STD"]
    compute_time_scaling_std_scalar = compute_time_scaling_std.loc[compute_time_scaling_std['SIMD EXTENSION'] ==
                                        "SCALAR"]
    compute_time_scaling_std_avx = compute_time_scaling_std.loc[compute_time_scaling_std['SIMD EXTENSION'] ==
                                        args.simd_key]
    plot_cpu_only_node_level_scaling(compute_time_scaling_std_scalar, compute_time_scaling_std_avx, title_string,  args.output_prefix + "std_node_level_scaling.pdf", "STD", args.simd_key)

    compute_time_scaling_std = compute_time_scaling.loc[compute_time_scaling['SIMD LIBRARY'] ==
                                        "STD"]

    # plot speedup bars
    title_string = args.title_prefix + "on " + cpu_name + ":\n Component SIMD Speedups"
    plot_cpu_bars(raw_data, title_string, args.output_prefix + "component_simd_speedup.pdf", args.simd_key, 1)


    # plot hydro runtimes
    raw_data_hydro_icelake = pd.read_csv(
        "icelake_legacy_test.data",
        comment='#',
        names=raw_data_colnames,
        header=None,
        on_bad_lines = 'warn')
    raw_data_hydro_epyc = pd.read_csv(
        "epyc_legacy_test.data",
        comment='#',
        names=raw_data_colnames,
        header=None,
        on_bad_lines = 'warn')
    raw_data_hydro_arm = pd.read_csv(
        "arm_legacy_test.data",
        comment='#',
        names=raw_data_colnames,
        header=None,
        on_bad_lines = 'warn')
    plot_hydro_speedup(raw_data_hydro_icelake, raw_data_hydro_epyc, raw_data_hydro_arm)
    exit(0)
