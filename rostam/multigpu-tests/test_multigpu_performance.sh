#!/usr/bin/env bash

set -e

required_modules="gcc/11 cuda/12 hwloc cmake"
corecount="1 2 4 8 16 32 64 "
gpucount="1 2 4 "
executorcount="1 2 4 8 16 32 64 128 "
slicescount="1 2 4 8 16 32 64 128 "
gpu_backends="CUDA KOKKOS_CUDA"
cpu_backends="LEGACY KOKKOS DEVICE_ONLY"
scenario_name="BLAST"
max_level=3
scenario_parameters=" --config_file=blast.ini --unigrid=1 --disable_output=on --max_level=${max_level} --stop_step=25 --stop_time=25 --print_times_per_timestep=1 --optimize_local_communication=1"
hpx_parameters="--hpx:ini=hpx.stacks.use_guard_pages!=0 --hpx:queuing=local-priority-lifo "
counter_parameters="--hpx:print-counter=/octotiger*/compute/gpu*kokkos* --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_kokkos --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_kokkos_aggregated --hpx:print-counter=/arithmetics/add@/cppuddle*/number_creations/ --hpx:print-counter=/arithmetics/add@/cppuddle*/number_allocations/ --hpx:print-counter=/arithmetics/add@/cppuddle*/number_deallocations/"
spack_spec="octotiger@0.10.0 +cuda +kokkos cuda_arch=80 griddim=8 %gcc@11 ^hpx@1.9.1 generator=make max_cpu_count=256 networking=none lci_server=ofi ~disable_async_gpu_futures ^python@3.6.15 ^kokkos@4.0.01 ^cppuddle@0.3.0 number_buffer_buckets=128 max_number_gpus=8 +hpx +allocator_counters"

# Timestamp
today=`date +%Y-%m-%d_%H_%M`
#git clone --recurse-submodules https://github.com/STEllAR-GROUP/octotiger experiment_pure_hydro_benchmark_run_${today}
#cd experiment_pure_hydro_benchmark_run_${today}
#git checkout develop
echo "Creating experiment directory..." | tee octotiger_benchmark.log

mkdir octotiger_benchmark_${scenario_name}-${max_level}
cd benchmark_pure_hydro_initial_${today}
echo "Loading required modules: ${required_modules}" | tee -a octotiger_benchmark.log
module load ${required_modules}
echo "Print the basic information.."| tee -a octotiger_benchmark.log

# Output to log
echo "# Date of run $today" | tee -a octotiger_benchmark.log
echo "# Modules: $(module list)" | tee -a octotiger_benchmark.log
echo "# Spack Root: $(printenv | grep SPACK_ROOT)" | tee -a octotiger_benchmark.log
echo "# Spack version: $(spack --version)" | tee -a octotiger_benchmark.log
cd $(spack repo list | grep octotiger-spack | awk -F ' ' '{print $2}')
echo "# Spack Octotiger repo version: $(git log --oneline --no-abbrev-commit | head -n 1)" | tee -a octotiger_benchmark.log
cd -
echo "# Spack spec spec: ${spack_spec}"
echo "Concretizing spec.."| tee -a octotiger_benchmark.log
echo "# Concrete spec: $(spack spec ${spack_spec})"  | tee -a octotiger_benchmark.log

# Output to CSV
echo "# Octo-Tiger Node-Level Benchmark run" | tee octotiger_benchmark.csv
echo "# ===================================" | tee -a octotiger_benchmark.csv 
echo "# Date of run $today" | tee -a octotiger_benchmark.csv
echo "# Scenario name ${scenario_name} at max level ${max_level}" | tee
echo "# Modules: $(module list)" | tee -a octotiger_benchmark.csv
echo "# Spack Root: $(printenv | grep SPACK_ROOT)" | tee -a octotiger_benchmark.csv
echo "# Spack version: $(spack --version)" | tee -a octotiger_benchmark.csv
cd $(spack repo list | grep octotiger-spack | awk -F ' ' '{print $2}')
echo "# Spack Octotiger repo version: $(git log --oneline --no-abbrev-commit | head -n 1)" | tee -a octotiger_benchmark.csv
cd -
echo "# Spack spec spec: ${spack_spec}" | tee -a octotiger_benchmark.csv
echo "# Concrete spec: $(spack spec ${spack_spec})"  | tee -a octotiger_benchmark.csv
echo "# Core Axis: ${corecount}" | tee -a octotiger_benchmark.csv
echo "# GPU  Axis: ${gpucount}" | tee -a octotiger_benchmark.csv
echo "# Executors Axis: ${executorcount}" | tee -a octotiger_benchmark.csv
echo "# Fusion Axis: ${slicescount}" | tee -a octotiger_benchmark.csv
echo "# CPU Kernel Axis: ${cpu_backends}" | tee -a octotiger_benchmark.csv
echo "# GPU Kernel Axis: ${gpu_backends}" | tee -a octotiger_benchmark.csv
echo "# Octo-Tiger parameters: ${scenario_parameters}" | tee -a octotiger_benchmark.csv
echo "# Fixed HPX parameters: ${hpx_parameters}" | tee -a octotiger_benchmark.csv
echo "# Counter parameters: ${counter_parameters}" | tee -a octotiger_benchmark.csv


#Setup
echo "Installing spec..."| tee -a octotiger_benchmark.log
spack install ${spack_spec} | tee -a octotiger_benchmark.log
echo "Loading spec..."| tee -a octotiger_benchmark.log
spack unload --all
spack load ${spack_spec} 

echo "Testing binary..."| tee -a octotiger_benchmark.log
octotiger --help

echo "Move scenario files to experiment directory..."| tee -a octotiger_benchmark.log
cp ../multigpu-tests*.sh .
cp ../*.ini .

set +e

echo "Using binary: $(which octotiger)..." | tee -a octotiger_benchmark.log






echo "scenario name, max level, cpu kernel type, gpu kernel type, future_type, slices, executors, nodes, processcount, total time, computation time, number creation, number allocations, number deallocations, number hydro kokkos launches, number aggregated hydro kokkos launches, number hydro cuda launches, number aggregated cuda kokkos launches, min time-per-timestep, max time-per-timestep," \
     "median time-per-timestep, average time-per-timestep, list of times per timestep" \
     | tee octotiger_benchmark.csv

for cpu_kernel_type in $cpu_backends; do

for cores in $corecount; do
  if [[ "${cpu_kernel_type}" == "DEVICE_ONLY" ]]; then
    for gpu_kernel_type in $gpu_backends; do
      for gpus in $gpucount; do
        for executors in $executorcount; do
          for slices in $slicescount; do
            performance_parameters="--hpx:threads=${cores} --multipole_device_kernel_type=${gpu_kernel_type} --multipole_host_kernel_type=${cpu_kernel_type} --monopole_device_kernel_type=${gpu_kernel_type} --monopole_host_kernel_type=${cpu_kernel_type} --hydro_device_kernel_type=${gpu_kernel_type} --hydro_host_kernel_type=${cpu_kernel_type} --amr_boundary_kernel_type=AMR_OPTIMIZED --max_kernels_fused=${slices} --executors_per_gpu=${executors} --number_gpus=${gpus}"
            # run gpu-only scenario:
            echo "octotiger ${scenario_parameters} ${hpx_parameters} ${performance_parameters} ${counter_parameters}" | tee current_run.out
            octotiger ${scenario_parameters} ${hpx_parameters} ${performance_parameters} ${counter_parameters} | tee current_run.out
            # append output to log
            cat current_run.ot >> octotiger_benchmark.log
            # analyse output for relevant counters and runtime
            echo "${scenario_name}, ${max_level}, ${cpu_kernel_type}, ${gpu_kernel_type}, ASYNC, ${slices}, ${executors}, 1, 1, " \
                 "$(cat current_run.out | grep Total: | sed -n 1p | awk -F ' ' '{print $2}')," \
                 "$(cat current_run.out | grep Computation: | sed -n 1p | awk -F ' ' '{print $2}')," \
                 "$(cat current_run.out | grep number_creations | awk -F ',' '{print $5}')," \
                 "$(cat current_run.out | grep number_allocations | awk -F ',' '{print $5}')," \
                 "$(cat current_run.out | grep number_deallocations | awk -F ',' '{print $5}')," \
                 "$(cat current_run.out | grep arithmetics.*hydro_kokkos, | awk -F ',' '{print $5}')," \
                 "$(cat current_run.out | grep arithmetics.*hydro_kokkos_aggregated, | awk -F ',' '{print $5}')," \
                 "$(cat current_run.out | grep arithmetics.*hydro_cuda, | awk -F ',' '{print $5}')," \
                 "$(cat current_run.out | grep arithmetics.*hydro_cuda_aggregated, | awk -F ',' '{print $5}')," \
                 "$(cat current_run.out | grep Min\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
                 "$(cat current_run.out | grep Max\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
                 "$(cat current_run.out | grep Median\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
                 "$(cat current_run.out | grep Average\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
                 "\"$(cat current_run.out | grep List\ of\ times-per-timestep | sed -n 1p | cut -d " " -f4-)\"" \
                 | tee -a octotiger_benchmark.csv
          done # end slices
        done # end executors
      done # end gpus
    done #gpu_kernel_type
  else
    executors=0
    slices=1
    gpus=0
    gpu_kernel_type="OFF"
    performance_parameters="--hpx:threads=${cores} --multipole_device_kernel_type=${gpu_kernel_type} --multipole_host_kernel_type=${cpu_kernel_type} --monopole_device_kernel_type=${gpu_kernel_type} --monopole_host_kernel_type=${cpu_kernel_type} --hydro_device_kernel_type=${gpu_kernel_type} --hydro_host_kernel_type=${cpu_kernel_type} --amr_boundary_kernel_type=AMR_OPTIMIZED --max_kernels_fused=${slices} --executors_per_gpu=${executors} --number_gpus=${gpus}"
    echo "octotiger ${scenario_parameters} ${hpx_parameters} ${performance_parameters} ${counter_parameters}" | tee current_run.out
    octotiger ${scenario_parameters} ${hpx_parameters} ${performance_parameters} ${counter_parameters} | tee current_run.out
    # append output to log
    cat current_run.ot >> octotiger_benchmark.log
    # analyse output for relevant counters and runtime
    echo "${scenario_name}, ${max_level}, ${cpu_kernel_type}, ${gpu_kernel_type}, ASYNC, ${slices}, ${executors}, 1, 1, " \
         "$(cat current_run.out | grep Total: | sed -n 1p | awk -F ' ' '{print $2}')," \
         "$(cat current_run.out | grep Computation: | sed -n 1p | awk -F ' ' '{print $2}')," \
         "$(cat current_run.out | grep number_creations | awk -F ',' '{print $5}')," \
         "$(cat current_run.out | grep number_allocations | awk -F ',' '{print $5}')," \
         "$(cat current_run.out | grep number_deallocations | awk -F ',' '{print $5}')," \
         "$(cat current_run.out | grep arithmetics.*hydro_kokkos, | awk -F ',' '{print $5}')," \
         "$(cat current_run.out | grep arithmetics.*hydro_kokkos_aggregated, | awk -F ',' '{print $5}')," \
         "$(cat current_run.out | grep arithmetics.*hydro_cuda, | awk -F ',' '{print $5}')," \
         "$(cat current_run.out | grep arithmetics.*hydro_cuda_aggregated, | awk -F ',' '{print $5}')," \
         "$(cat current_run.out | grep Min\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
         "$(cat current_run.out | grep Max\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
         "$(cat current_run.out | grep Median\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
         "$(cat current_run.out | grep Average\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
         "\"$(cat current_run.out | grep List\ of\ times-per-timestep | sed -n 1p | cut -d " " -f4-)\"" \
         | tee -a octotiger_benchmark.csv
  fi
done #end cores
done #cpu_kernel_type

exit 0



echo "Extracting done! Zipping and printing results..." | tee -a octotiger_benchmark.log
cat octotiger_benchmark.csv | tee -a octotiger_benchmark.log

scriptname=$(basename "$0")
tar -cvf octotiger_benchmark_${scenario_name}-${max_level}_${today}.tar octotiger_benchmark.log octotiger_benchmark.csv \
       	 blast.ini scriptname
