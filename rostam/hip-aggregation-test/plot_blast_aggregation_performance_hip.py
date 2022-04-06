import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = "../work-aggregation/blast_aggregation_test_hip_cpuamr_3_15_2022-04-01_10:09:14_kamand0.rostam.cct.lsu.edu_LOG.txt"
print("Hello World")
raw_data_colnames=['Cores', 'Executors', 'Max Aggregation Slices', 'Computation Time', 'Total Time', 'Reconstruct Kernel Launches', 'Reconstruct Kernel Avg Time', 'Flux Kernel Launches', 'Flux Kernel Avg Time', 'Discs1 Kernel Launches', 'Discs1 Kernel Avg Time', 'Discs2 Kernel Launches', 'Discs2 Kernel Avg Time', 'Pre_Recon Kernel Launches', 'Pre_Recon Kernel Avg Time', 'Profiling Computation Time', 'Profiling Total Time'  ]
raw_data = pd.read_csv(filename, comment='#', names=raw_data_colnames, header=None)
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

# Handle host only run
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
host_only_run = host_only_run[['Cores', 'Computation Time', 'Total Time']].copy()
print(host_only_run)
#plt.scatter(host_only_run["Cores"], host_only_run["Total Time"])
ax = plt.gca()
ax.plot(host_only_run["Cores"], host_only_run["Total Time"], '-o', c='blue', alpha=0.95, markeredgecolor='none')
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
plt.savefig('basic-cpu-scaling.pdf', format="pdf", bbox_inches="tight")
plt.clf()
#plt.show()


non_aggregated_run = raw_data.loc[(raw_data['Executors'] == 128) & (raw_data['Max Aggregation Slices'] == 1)]
non_aggregated_run = non_aggregated_run[['Cores', 'Computation Time', 'Total Time']].copy()
slice_aggregated_run = raw_data.loc[(raw_data['Executors'] == 1) & (raw_data['Max Aggregation Slices'] == 128)]
slice_aggregated_run = slice_aggregated_run[['Cores', 'Computation Time', 'Total Time']].copy()
fully_aggregated_run = raw_data.loc[(raw_data['Executors'] == 8) & (raw_data['Max Aggregation Slices'] == 16)]
fully_aggregated_run = fully_aggregated_run[['Cores', 'Computation Time', 'Total Time']].copy()
fully_aggregated_run2 = raw_data.loc[(raw_data['Executors'] == 8) & (raw_data['Max Aggregation Slices'] == 128)]
fully_aggregated_run2 = fully_aggregated_run2[['Cores', 'Computation Time', 'Total Time']].copy()
print(non_aggregated_run)
ax = plt.gca()
#ax.plot(host_only_run["Cores"], host_only_run["Total Time"], '-o', c='blue', alpha=0.95, markeredgecolor='none', label='CPU only')
ax.plot(non_aggregated_run["Cores"], non_aggregated_run["Total Time"], '-o', c='red', alpha=0.95, markeredgecolor='none', label='Using 128 executors, 1 kernels per launch')
ax.plot(slice_aggregated_run["Cores"], slice_aggregated_run["Total Time"], '-o', c='green', alpha=0.95, markeredgecolor='none', label='Using 1 executors, Up to 128 kernels aggregated per launch')
ax.plot(fully_aggregated_run2["Cores"], fully_aggregated_run2["Total Time"], '-o', c='lightgreen', alpha=0.95, markeredgecolor='none', label='Using 8 executors, Up to 128 kernels aggregated per launch')
ax.plot(fully_aggregated_run["Cores"], fully_aggregated_run["Total Time"], '-o', c='teal', alpha=0.95, markeredgecolor='none', label='Using 8 executors, Up to 16 kernels aggregated per launch')
ax.set_yscale('log')
ax.set_xscale('log')
plt.title("Hydro-only: GPU infrastructure scaling")
plt.xlim(non_aggregated_run["Cores"].min() - 0.1, non_aggregated_run["Cores"].max() + 10)
plt.xlabel("Number of Cores")
plt.xticks(non_aggregated_run["Cores"], non_aggregated_run["Cores"])
#plt.ylim([0, non_aggregated_run["Total Time"].max() + 10])
plt.ylabel("Runtime in seconds")
plt.yticks(fully_aggregated_run["Total Time"], fully_aggregated_run["Total Time"])
#plt.yticks([1, 10, 100], [1, 10, 100])
plt.grid(True)
plt.legend()
plt.savefig('gpu-enabled-cpu-scaling.pdf', format="pdf", bbox_inches="tight")
plt.clf()

reconstruct_starved = raw_data.loc[(raw_data['Executors'] == 1) & (raw_data['Cores'] == 64)]
reconstruct_starved = reconstruct_starved[['Max Aggregation Slices', 'Reconstruct Kernel Launches', 'Reconstruct Kernel Avg Time']]
reconstruct_starved['Avg Kernel Aggregation'] = reconstruct_starved.apply(lambda row: reconstruct_starved['Reconstruct Kernel Launches'].values[0]/row['Reconstruct Kernel Launches'] , axis=1)
reconstruct_starved['Reconstruct Avg Subgrid Runtime'] = reconstruct_starved.apply(lambda row: row['Reconstruct Kernel Avg Time']/row['Avg Kernel Aggregation'] , axis=1)
reconstruct_starved['Aggregation Speedup'] = reconstruct_starved.apply(lambda row: reconstruct_starved['Reconstruct Avg Subgrid Runtime'].values[0]/row['Reconstruct Avg Subgrid Runtime'] , axis=1)
print(reconstruct_starved)

reconstruct_non_starved = raw_data.loc[(raw_data['Executors'] == 64) & (raw_data['Cores'] == 64)]
reconstruct_non_starved = reconstruct_non_starved[['Max Aggregation Slices', 'Reconstruct Kernel Launches', 'Reconstruct Kernel Avg Time']]
reconstruct_non_starved['Avg Kernel Aggregation'] = reconstruct_non_starved.apply(lambda row: reconstruct_non_starved['Reconstruct Kernel Launches'].values[0]/row['Reconstruct Kernel Launches'] , axis=1)
reconstruct_non_starved['Reconstruct Avg Subgrid Runtime'] = reconstruct_non_starved.apply(lambda row: row['Reconstruct Kernel Avg Time']/row['Avg Kernel Aggregation'] , axis=1)
reconstruct_non_starved['Aggregation Speedup'] = reconstruct_non_starved.apply(lambda row: reconstruct_non_starved['Reconstruct Avg Subgrid Runtime'].values[0]/row['Reconstruct Avg Subgrid Runtime'] , axis=1)
print(reconstruct_non_starved)

ax = plt.gca()
#ax.plot(host_only_run["Cores"], host_only_run["Total Time"], '-o', c='blue', alpha=0.95, markeredgecolor='none', label='CPU only')
plot1 = ax.plot(reconstruct_starved["Max Aggregation Slices"], reconstruct_starved["Aggregation Speedup"], '-o', c='red', alpha=0.95, markeredgecolor='none', label='Avg speedup [Starved GPU]')
plot2 = ax.plot(reconstruct_non_starved["Max Aggregation Slices"], reconstruct_non_starved["Aggregation Speedup"], '-o', c='orange', alpha=0.95, markeredgecolor='none', label='Avg speedup [Busy GPU]')
ax.set_xscale('log')
ax.set_title("Hydro-only on AMD MI100: Reconstruct Kernel Aggregation Sub-Grid Speedup")
ax.set_xlim(reconstruct_starved["Max Aggregation Slices"].min() - 0.1, reconstruct_starved["Max Aggregation Slices"].max() + 10)
ax.set_xlabel("Maximum allowed number of aggregated Subgrids per reconstruct kernel launch")
ax.set_xticks(ticks=reconstruct_starved["Max Aggregation Slices"], labels=reconstruct_starved["Max Aggregation Slices"])
ax.set_ylim([0, reconstruct_starved["Aggregation Speedup"].max() + 1])
ax.set_ylabel("Avg Speedup of the Reconstruct Kernel per Sub-grid")
ax.set_yticks(ticks=[1, 2, 4, 8, 16, reconstruct_starved["Aggregation Speedup"].max(), reconstruct_non_starved["Aggregation Speedup"].max()], labels=[1, 2, 4, 8, 16, "" + str(round(reconstruct_starved["Aggregation Speedup"].max(),1)), "" + str(round(reconstruct_non_starved["Aggregation Speedup"].max(),1))])
ax.axhline(reconstruct_starved["Aggregation Speedup"].max(), linestyle='--', color='k')
ax.axhline(reconstruct_non_starved["Aggregation Speedup"].max(), linestyle='--', color='k')
ax2 = ax.twinx()
plot3 = ax2.plot(reconstruct_starved["Max Aggregation Slices"], reconstruct_starved["Avg Kernel Aggregation"], '--', c='red', alpha=0.35, markeredgecolor='none', label='Avg aggregation rate [Starved GPU]')
plot4 = ax2.plot(reconstruct_non_starved["Max Aggregation Slices"], reconstruct_non_starved["Avg Kernel Aggregation"], '--', c='orange', alpha=0.35, markeredgecolor='none', label='Avg aggregation rate [Busy GPU]')
ax2.set_ylabel("Average Aggregation Rate (Subgrids per Launch)")
all_plots = plot1 + plot2 + plot3 + plot4
labels = [l.get_label() for l in all_plots]
ax.legend(all_plots, labels, loc=0)
plt.savefig('reconstruct-speedup-per-subgrid.pdf', format="pdf", bbox_inches="tight")
plt.clf()


ax = plt.gca()
#ax.plot(host_only_run["Cores"], host_only_run["Total Time"], '-o', c='blue', alpha=0.95, markeredgecolor='none', label='CPU only')
plot1 = ax.plot(reconstruct_starved["Max Aggregation Slices"], reconstruct_starved["Reconstruct Avg Subgrid Runtime"], '-o', c='red', alpha=0.95, markeredgecolor='none', label='Avg runtime [Starved GPU]')
plot2 = ax.plot(reconstruct_non_starved["Max Aggregation Slices"], reconstruct_non_starved["Reconstruct Avg Subgrid Runtime"], '-o', c='orange', alpha=0.95, markeredgecolor='none', label='Avg runtime [Busy GPU]')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title("Hydro-only on AMD MI100: Reconstruct Kernel Reconstruct Avg Subgrid Runtime")
ax.set_xlim(reconstruct_starved["Max Aggregation Slices"].min() - 0.1, reconstruct_starved["Max Aggregation Slices"].max() + 10)
ax.set_xlabel("Maximum allowed number of aggregated Subgrids per reconstruct kernel launch")
ax.set_xticks(ticks=reconstruct_starved["Max Aggregation Slices"], labels=reconstruct_starved["Max Aggregation Slices"])
ax.set_ylim([0, reconstruct_non_starved["Reconstruct Avg Subgrid Runtime"].max() + 40 * 1000])
ax.set_ylabel("Avg runtime of the Reconstruct Kernel per Sub-grid")
ax.set_yticks(ticks=[reconstruct_starved["Reconstruct Avg Subgrid Runtime"].max(), reconstruct_non_starved["Reconstruct Avg Subgrid Runtime"].max(), reconstruct_starved["Reconstruct Avg Subgrid Runtime"].min(), reconstruct_non_starved["Reconstruct Avg Subgrid Runtime"].min()], labels=[str(round(reconstruct_starved["Reconstruct Avg Subgrid Runtime"].max()/1000,3)) + str('\u03bcs'), str(round(reconstruct_non_starved["Reconstruct Avg Subgrid Runtime"].max()/1000,3)) + str('\u03bcs'), str(round(reconstruct_starved["Reconstruct Avg Subgrid Runtime"].min()/1000,3)) + str('\u03bcs'), str(round(reconstruct_non_starved["Reconstruct Avg Subgrid Runtime"].min()/1000,3)) + str('\u03bcs')])
ax.axhline(reconstruct_starved["Reconstruct Avg Subgrid Runtime"].max(), linestyle='--', color='k')
ax.axhline(reconstruct_non_starved["Reconstruct Avg Subgrid Runtime"].max(), linestyle='--', color='k')
ax.axhline(reconstruct_starved["Reconstruct Avg Subgrid Runtime"].min(), linestyle='--', color='k')
ax.axhline(reconstruct_non_starved["Reconstruct Avg Subgrid Runtime"].min(), linestyle='--', color='k')
ax2 = ax.twinx()
plot3 = ax2.plot(reconstruct_starved["Max Aggregation Slices"], reconstruct_starved["Avg Kernel Aggregation"], '--', c='red', alpha=0.35, markeredgecolor='none', label='Avg aggregation rate [Starved GPU]')
plot4 = ax2.plot(reconstruct_non_starved["Max Aggregation Slices"], reconstruct_non_starved["Avg Kernel Aggregation"], '--', c='orange', alpha=0.35, markeredgecolor='none', label='Avg aggregation rate [Busy GPU]')
ax2.set_ylabel("Average Aggregation Rate (Subgrids per Launch)")
all_plots = plot1 + plot2 + plot3 + plot4
labels = [l.get_label() for l in all_plots]
ax.legend(all_plots, labels, loc=0)
plt.savefig('reconstruct-runtime-per-subgrid.pdf', format="pdf", bbox_inches="tight")
plt.clf()

ax = plt.gca()
#ax.plot(host_only_run["Cores"], host_only_run["Total Time"], '-o', c='blue', alpha=0.95, markeredgecolor='none', label='CPU only')
plot1 = ax.plot(reconstruct_starved["Max Aggregation Slices"], reconstruct_starved["Reconstruct Kernel Avg Time"], '-o', c='red', alpha=0.95, markeredgecolor='none', label='Avg runtime [Starved GPU]')
plot2 = ax.plot(reconstruct_non_starved["Max Aggregation Slices"], reconstruct_non_starved["Reconstruct Kernel Avg Time"], '-o', c='orange', alpha=0.95, markeredgecolor='none', label='Avg runtime [Busy GPU]')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title("Hydro-only on AMD MI100: Avg Aggregated Reconstruct Kernel Runtime")
ax.set_xlim(reconstruct_starved["Max Aggregation Slices"].min() - 0.1, reconstruct_starved["Max Aggregation Slices"].max() + 10)
ax.set_xlabel("Maximum allowed number of aggregated Subgrids per reconstruct kernel launch")
ax.set_xticks(ticks=reconstruct_starved["Max Aggregation Slices"], labels=reconstruct_starved["Max Aggregation Slices"])
ax.set_ylim([reconstruct_starved["Reconstruct Kernel Avg Time"].min()-20*1000, reconstruct_non_starved["Reconstruct Kernel Avg Time"].max() + 40*1000])
ax.set_ylabel("Avg runtime of the Reconstruct Kernel per kernel launch")
ax.set_yticks(ticks=[reconstruct_starved["Reconstruct Kernel Avg Time"].max(), reconstruct_non_starved["Reconstruct Kernel Avg Time"].max(), reconstruct_starved["Reconstruct Kernel Avg Time"].min(), reconstruct_non_starved["Reconstruct Kernel Avg Time"].min()], labels=[str(round(reconstruct_starved["Reconstruct Kernel Avg Time"].max()/1000,3)) + str('\u03bcs'), str(round(reconstruct_non_starved["Reconstruct Kernel Avg Time"].max()/1000,3)) + str('\u03bcs'), str(round(reconstruct_starved["Reconstruct Kernel Avg Time"].min()/1000,3)) + str('\u03bcs'), str(round(reconstruct_non_starved["Reconstruct Kernel Avg Time"].min()/1000,3)) + str('\u03bcs')], minor=False)
ax.set_yticks(ticks=[], labels=[], minor=True) # Reset minor ticks
ax.axhline(reconstruct_starved["Reconstruct Kernel Avg Time"].max(), linestyle='--', color='k')
ax.axhline(reconstruct_non_starved["Reconstruct Kernel Avg Time"].max(), linestyle='--', color='k')
ax.axhline(reconstruct_starved["Reconstruct Kernel Avg Time"].min(), linestyle='--', color='k')
ax.axhline(reconstruct_non_starved["Reconstruct Kernel Avg Time"].min(), linestyle='--', color='k')
ax2 = ax.twinx()
plot3 = ax2.plot(reconstruct_starved["Max Aggregation Slices"], reconstruct_starved["Avg Kernel Aggregation"], '--', c='red', alpha=0.35, markeredgecolor='none', label='Avg aggregation rate [Starved GPU]')
plot4 = ax2.plot(reconstruct_non_starved["Max Aggregation Slices"], reconstruct_non_starved["Avg Kernel Aggregation"], '--', c='orange', alpha=0.35, markeredgecolor='none', label='Avg aggregation rate [Busy GPU]')
ax2.set_ylabel("Average Aggregation Rate (Subgrids per Launch)")
all_plots = plot1 + plot2 + plot3 + plot4
labels = [l.get_label() for l in all_plots]
ax.legend(all_plots, labels, loc=0)
plt.savefig('reconstruct-runtime-per-launch.pdf', format="pdf", bbox_inches="tight")
plt.clf()
