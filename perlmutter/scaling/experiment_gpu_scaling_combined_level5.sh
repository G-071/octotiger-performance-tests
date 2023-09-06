#!/usr/bin/env bash

set -e
# Timestamp
spack --version
today=`date +%Y-%m-%d_%H_%M`
#git clone --recurse-submodules https://github.com/STEllAR-GROUP/octotiger experiment_combined_benchmark_run_${today}
#cd experiment_combined_benchmark_run_${today}
#git checkout develop
echo "Creating experiment directory..." | tee octotiger_combined_benchmark.log

mkdir benchmark_combined_initial_${today}
cd benchmark_combined_initial_${today}
echo "Print the basic information.."| tee -a octotiger_combined_benchmark.log

echo "# Date of run $today" | tee -a octotiger_combined_benchmark.log
echo "# Modules: $(module list)" | tee -a octotiger_combined_benchmark.log
echo "# Spack Root: $(printenv | grep SPACK_ROOT)" | tee -a octotiger_combined_benchmark.log
echo "# Spack version: $(spack --version)" | tee -a octotiger_combined_benchmark.log
cd $(spack repo list | grep octotiger-spack | awk -F ' ' '{print $2}')
echo "# Spack Octotiger repo version: $(git log --oneline --no-abbrev-commit | head -n 1)" | tee -a octotiger_combined_benchmark.log
cd -

echo "Spec: octotiger@0.10.0 +cuda +kokkos cuda_arch=80 griddim=8 %gcc@11 ^hpx@1.9.1 generator=make max_cpu_count=256 networking=mpi lci_server=ofi ~disable_async_gpu_futures ^python@3.6.15 ^kokkos@4.0.01 ^cppuddle@0.3.0 number_buffer_buckets=64 max_number_gpus=4 +hpx +allocator_counters"

echo "Concretizing spec.."| tee -a octotiger_combined_benchmark.log
spack spec octotiger@0.10.0 +cuda +kokkos cuda_arch=80 griddim=8 %gcc@11 ^hpx@1.9.1 generator=make max_cpu_count=256 networking=mpi lci_server=ofi ~disable_async_gpu_futures ^python@3.6.15 ^kokkos@4.0.01 ^cppuddle@0.3.0 number_buffer_buckets=64 max_number_gpus=4 +hpx +allocator_counters | tee -a octotiger_combined_benchmark.log


echo "Installing spec..."| tee -a octotiger_combined_benchmark.log
spack install octotiger@0.10.0 +cuda +kokkos cuda_arch=80 griddim=8 %gcc@11 ^hpx@1.9.1 generator=make max_cpu_count=256 networking=mpi lci_server=ofi ~disable_async_gpu_futures ^python@3.6.15 ^kokkos@4.0.01 ^cppuddle@0.3.0 number_buffer_buckets=64 max_number_gpus=4 +hpx +allocator_counters | tee -a octotiger_combined_benchmark.log


echo "Loading spec..."| tee -a octotiger_combined_benchmark.log
spack unload --all
spack load octotiger@0.10.0 +cuda +kokkos cuda_arch=80 griddim=8 %gcc@11 ^hpx@1.9.1 generator=make max_cpu_count=256 networking=mpi lci_server=ofi ~disable_async_gpu_futures ^python@3.6.15 ^kokkos@4.0.01 ^cppuddle@0.3.0 number_buffer_buckets=64 max_number_gpus=4 +hpx +allocator_counters

echo "Testing binary..."| tee -a octotiger_combined_benchmark.log
octotiger --help

echo "Move scenario files to experiment directory..."| tee -a octotiger_combined_benchmark.log
cp ../octotiger_combined_scaling_gpu.sbatch .
cp ../rotating_star.ini .
cp ../rotating_star.bin .
cp ../experiment_gpu_scaling_combined.sh .

set +e

echo "Running basic scaling experiment with 16 executors and 8 max_kernels_fused" | tee -a octotiger_combined_benchmark.log
echo "Using binary: $(which octotiger)..." | tee -a octotiger_combined_benchmark.log

echo "Scheduling octotiger_combined_scaling_l5_e16_f8_1_4..." | tee -a octotiger_combined_benchmark.log
OCTOTIGER_MAX_LEVEL=5 OCTOTIGER_EXECUTORS_PER_GPU=16 OCTOTIGER_MAX_KERNELS_FUSED=8 \
       	sbatch --qos=preempt --time=00:10:00 -N 1 -n 4 -o octotiger_combined_scaling_l5_e16_f8_1_4.out -e octotiger_combined_scaling_l5_e16_f8_1_4.out \
       	-J octotiger_combined_scaling_l5_e16_f8_1_4 --wait octotiger_combined_scaling_gpu.sbatch && echo "octotiger_combined_scaling_l5_e16_f8_1_4 DONE" &
echo "Scheduling octotiger_combined_scaling_l5_e16_f8_2_8..." | tee -a octotiger_combined_benchmark.log
OCTOTIGER_MAX_LEVEL=5 OCTOTIGER_EXECUTORS_PER_GPU=16 OCTOTIGER_MAX_KERNELS_FUSED=8 \
	sbatch --qos=preempt --time=00:10:00 -N 2 -n 8 -o octotiger_combined_scaling_l5_e16_f8_2_8.out -e octotiger_combined_scaling_l5_e16_f8_2_8.out \
       	-J octotiger_combined_scaling_l5_e16_f8_2_8 --wait octotiger_combined_scaling_gpu.sbatch && echo "octotiger_combined_scaling_l5_e16_f8_2_8 DONE" &


NodeList="4 8 16 32 64 128"
for nodecount in $NodeList; do
	processcount=$((${nodecount}*4))
	echo "Scheduling octotiger_combined_scaling_l5_e16_f8_${nodecount}_${processcount}..." | tee -a octotiger_combined_benchmark.log
	OCTOTIGER_MAX_LEVEL=5 OCTOTIGER_EXECUTORS_PER_GPU=16 OCTOTIGER_MAX_KERNELS_FUSED=8 \
		sbatch --qos=preempt --time=00:03:00 -N ${nodecount} -n ${processcount} \
	       	-o octotiger_combined_scaling_l5_e16_f8_${nodecount}_${processcount}.out \
		-e octotiger_combined_scaling_l5_e16_f8_${nodecount}_${processcount}.out \
		-J octotiger_combined_scaling_l5_e16_f8_${nodecount}_${processcount} --wait octotiger_combined_scaling_gpu.sbatch \
		&& echo "octotiger_combined_scaling_l5_e16_f8_${nodecount}_${processcount} DONE" &
done

wait
echo "All Runs done! Extracting runtimes..." | tee -a octotiger_combined_benchmark.log

echo "scenario name, max level, future_type, slices, executors, nodes, processcount, total time, computation time, number creation, number allocations, number deallocations, number hydro launches, number aggregated hydro launches, min time-per-timestep, max time-per-timestep," \
     "median time-per-timestep, average time-per-timestep, list of times per timestep" \
     | tee octotiger_combined_benchmark_results.csv
echo "ROTATING_STAR, 4, ASYNC, 8, 16, 1, 4," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_1_4.out | grep Total: | sed -n 1p | awk -F ' ' '{print $2}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_1_4.out | grep Computation: | sed -n 1p | awk -F ' ' '{print $2}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_1_4.out | grep number_creations | awk -F ',' '{print $5}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_1_4.out | grep number_allocations | awk -F ',' '{print $5}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_1_4.out | grep number_deallocations | awk -F ',' '{print $5}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_1_4.out | grep arithmetics.*hydro_kokkos, | awk -F ',' '{print $5}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_1_4.out | grep arithmetics.*hydro_kokkos_aggregated, | awk -F ',' '{print $5}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_1_4.out | grep Min\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_1_4.out | grep Max\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_1_4.out | grep Median\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_1_4.out | grep Average\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
     "\"$(cat octotiger_combined_scaling_l5_e16_f8_1_4.out | grep List\ of\ times-per-timestep | sed -n 1p | cut -d " " -f4-)\"" \
     | tee -a octotiger_combined_benchmark_results.csv
echo "ROTATING_STAR, 4, ASYNC, 8, 16, 2, 8," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_2_8.out | grep Total: | sed -n 1p | awk -F ' ' '{print $2}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_2_8.out | grep Computation: | sed -n 1p | awk -F ' ' '{print $2}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_2_8.out | grep number_creations | awk -F ',' '{print $5}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_2_8.out | grep number_allocations | awk -F ',' '{print $5}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_2_8.out | grep number_deallocations | awk -F ',' '{print $5}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_2_8.out | grep arithmetics.*hydro_kokkos, | awk -F ',' '{print $5}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_2_8.out | grep arithmetics.*hydro_kokkos_aggregated, | awk -F ',' '{print $5}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_2_8.out | grep Min\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_2_8.out | grep Max\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_2_8.out | grep Median\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_2_8.out | grep Average\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
     "\"$(cat octotiger_combined_scaling_l5_e16_f8_2_8.out | grep List\ of\ times-per-timestep | sed -n 1p | cut -d " " -f4-)\"" \
     | tee -a octotiger_combined_benchmark_results.csv

for nodecount in $NodeList; do
	processcount=$((${nodecount}*4))
echo "ROTATING_STAR, 4, ASYNC, 8, 16, ${nodecount}, ${processcount}," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_${nodecount}_${processcount}.out | grep Total: | sed -n 1p | awk -F ' ' '{print $2}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_${nodecount}_${processcount}.out | grep Computation: | sed -n 1p | awk -F ' ' '{print $2}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_${nodecount}_${processcount}.out | grep number_creations | awk -F ',' '{print $5}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_${nodecount}_${processcount}.out | grep number_allocations | awk -F ',' '{print $5}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_${nodecount}_${processcount}.out | grep number_deallocations | awk -F ',' '{print $5}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_${nodecount}_${processcount}.out | grep arithmetics.*hydro_kokkos, | awk -F ',' '{print $5}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_${nodecount}_${processcount}.out | grep arithmetics.*hydro_kokkos_aggregated, | awk -F ',' '{print $5}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_${nodecount}_${processcount}.out | grep Min\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_${nodecount}_${processcount}.out | grep Max\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_${nodecount}_${processcount}.out | grep Median\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
     "$(cat octotiger_combined_scaling_l5_e16_f8_${nodecount}_${processcount}.out | grep Average\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
     "\"$(cat octotiger_combined_scaling_l5_e16_f8_${nodecount}_${processcount}.out | grep List\ of\ times-per-timestep | sed -n 1p | cut -d " " -f4-)\"" \
     | tee -a octotiger_combined_benchmark_results.csv
done


echo "Extracting done! Zipping and printing results..." | tee -a octotiger_combined_benchmark.log
cat octotiger_combined_benchmark_results.csv | tee -a octotiger_combined_benchmark.log

tar -cvf octotiger_combined_benchmark_${today}.tar octotiger_combined_scaling_*.out octotiger_combined_benchmark_results.csv \
       	 octotiger_combined_scaling_gpu.sbatch rotating_star.ini octotiger_combined_benchmark.log experiment_gpu_scaling_combined.sh
