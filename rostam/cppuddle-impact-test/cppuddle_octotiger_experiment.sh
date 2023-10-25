#!/bin/bash

# Config
# --------------
scenario_name="hydro_cppuddle"
corelist="16 32 64"  
max_level=3
hpx_parameters="--hpx:use-process-mask --hpx:numa-sensitive --hpx:print-bind"
scenario_parameters=" --problem=blast --odt=0.1 gravity=off --unigrid=1 --cuda_number_gpus=1 --theta=0.34  --max_level=${max_level} --correct_am_hydro=0 --stop_time=25 --stop_step=25  --disable_output=1 --print_times_per_timestep=1"
kernel_parameters="--monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_CUDA --multipole_device_kernel_type=KOKKOS_CUDA --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=DEVICE_ONLY --amr_boundary_kernel_type=AMR_OPTIMIZED"
counter_parameters="--hpx:print-counter=/octotiger*/compute/gpu*kokkos* --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_cuda --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_cuda_aggregated --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_kokkos --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_kokkos_aggregated --hpx:print-counter=/arithmetics/add@/cppuddle*/number_creations/ --hpx:print-counter=/arithmetics/add@/cppuddle*/number_allocations/ --hpx:print-counter=/arithmetics/add@/cppuddle*/number_deallocations/"
function extract_times_and_counters {
	local output_string=$(\
    echo "$(cat $1 | grep Total: | sed -n 1p | awk -F ' ' '{print $2}')," \
         "$(cat $1 | grep Computation: | sed -n 1p | awk -F ' ' '{print $2}')," \
         "$(cat $1 | grep number_creations | awk -F ',' '{print $5}')," \
         "$(cat $1 | grep number_allocations | awk -F ',' '{print $5}')," \
         "$(cat $1 | grep number_deallocations | awk -F ',' '{print $5}')," \
         "$(cat $1 | grep arithmetics.*hydro_kokkos, | awk -F ',' '{print $5}')," \
         "$(cat $1 | grep arithmetics.*hydro_kokkos_aggregated, | awk -F ',' '{print $5}')," \
         "$(cat $1 | grep arithmetics.*hydro_cuda, | awk -F ',' '{print $5}')," \
         "$(cat $1 | grep arithmetics.*hydro_cuda_aggregated, | awk -F ',' '{print $5}')," \
         "$(cat $1 | grep Min\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
         "$(cat $1 | grep Max\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
         "$(cat $1 | grep Median\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
         "$(cat $1 | grep Average\ time-per-timestep: | sed -n 1p | awk -F ' ' '{print $3}')," \
         "\"$(cat $1 | grep List\ of\ times-per-timestep | sed -n 1p | cut -d " " -f4-)\"" \
       )
	echo "${output_string}"
}	

# 0. Intro Infos
# --------------
scriptname=$(basename "$0")
# Timestamp
today=`date +%Y-%m-%d_%H_%M`
echo "# Experiment name: ${scenario_name}" | tee ${scenario_name}_benchmark.log
echo "# Experiment date: ${today}" | tee -a ${scenario_name}_benchmark.log
echo "# Basic spec: octotiger@master build_type=Release +kokkos +cuda cuda_arch=80 %gcc@11 ^kokkos@4.0.01 ^hpx@1.9.1 max_cpu_count=256 networking=none " | tee -a ${scenario_name}_benchmark.log
echo "# Scenario parameters: ${scenario_parameters}" | tee -a ${scenario_name}_benchmark.log
echo "# HPX parameters: ${hpx_parameter}" | tee -a ${scenario_name}_benchmark.log
echo "# Kernel parameters: ${kernel_parameter}" | tee -a ${scenario_name}_benchmark.log
echo "# Counter parameters: ${counter_parameter}" | tee -a ${scenario_name}_benchmark.log
cp ${scenario_name}_benchmark.log  ${scenario_name}_benchmark.csv
echo "scenario name, max_level, hpx threads, slices, executors, total time, computation time, number creation, number allocations, number deallocations, number hydro kokkos launches, number aggregated hydro kokkos launches, number hydro cuda launches, number aggregated cuda kokkos launches, min time-per-timestep, max time-per-timestep," \
     "median time-per-timestep, average time-per-timestep, list of times per timestep" \
     | tee -a ${scenario_name}_benchmark.csv


# 1. Baseline without any integrations
# -------------------------------------

slices=1
executors=1
echo "Starting baseline testing" | tee -a ${scenario_name}_benchmark.log
echo "Installing spec..." | tee -a ${scenario_name}_benchmark.log
spack install --fresh octotiger@master build_type=Release +kokkos +cuda cuda_arch=80 %gcc@11 ^kokkos@4.0.01 ^hpx@1.9.1 max_cpu_count=256 networking=none ~async_gpu_futures ^cppuddle@0.3.1 ~buffer_recycling ~buffer_content_recycling ~executor_recycling max_number_gpus=8 +allocator_counters ^silo~mpi
spack load octotiger@master build_type=Release +kokkos +cuda cuda_arch=80 %gcc@11 ^kokkos@4.0.01 ^hpx@1.9.1 max_cpu_count=256 networking=none ~async_gpu_futures ^cppuddle@0.3.1 ~buffer_recycling ~buffer_content_recycling ~executor_recycling max_number_gpus=8 +allocator_counters ^silo~mpi
for cores in ${corelist}; do
      echo "Running with ${cores} threads..." | tee -a ${scenario_name}_benchmark.log
      octotiger --hpx:threads=${cores} ${hpx_parameters} ${scenario_parameters} --executors_per_gpu=${executors} --max_kernels_fused=${slices} ${kernel_parameters} ${counter_parameters} | tee current_output.log
      cat current_output.log >> ${scenario_name}_benchmark.log
      output_metrics=$(extract_times_and_counters "current_output.log")
      echo "${scenario_name}_baseline, ${max_level}, ${cores}, ${slices}, ${executors}, ${output_metrics} " \
           | tee -a ${scenario_name}_benchmark.csv
done
spack unload

# 2. Run with buffer recycling
# -------------------------------------
echo "Starting testing with buffer recycling" | tee -a ${scenario_name}_benchmark.log
echo "Installing spec..." | tee -a ${scenario_name}_benchmark.log
spack install --fresh octotiger@master build_type=Release +kokkos +cuda cuda_arch=80 %gcc@11 ^kokkos@4.0.01 ^hpx@1.9.1 max_cpu_count=256 networking=none ~async_gpu_futures ^cppuddle@0.3.1 +buffer_recycling +buffer_content_recycling ~executor_recycling max_number_gpus=8 +allocator_counters ^silo~mpi
spack load octotiger@master build_type=Release +kokkos +cuda cuda_arch=80 %gcc@11 ^kokkos@4.0.01 ^hpx@1.9.1 max_cpu_count=256 networking=none ~async_gpu_futures ^cppuddle@0.3.1 ~buffer_recycling ~buffer_content_recycling ~executor_recycling max_number_gpus=8 +allocator_counters ^silo~mpi
for cores in ${corelist}; do
      echo "Running with ${cores} threads..." | tee -a ${scenario_name}_benchmark.log
      octotiger --hpx:threads=${cores} ${hpx_parameters} ${scenario_parameters} --executors_per_gpu=${executors} --max_kernels_fused=${slices} ${kernel_parameters} ${counter_parameters} | tee current_output.log
      cat current_output.log >> ${scenario_name}_benchmark.log
      output_metrics=$(extract_times_and_counters "current_output.log")
      echo "${scenario_name}_with_buffer_recycling, ${max_level}, ${cores}, ${slices}, ${executors}, ${output_metrics} " \
           | tee -a ${scenario_name}_benchmark.csv
done
spack unload

# 3. Run with executor recycling
# -------------------------------------
echo "Starting testing with buffer,executor recycling" | tee -a ${scenario_name}_benchmark.log
echo "Installing spec..." | tee -a ${scenario_name}_benchmark.log
spack install --fresh octotiger@master build_type=Release +kokkos +cuda cuda_arch=80 %gcc@11 ^kokkos@4.0.01 ^hpx@1.9.1 max_cpu_count=256 networking=none ~async_gpu_futures ^cppuddle@0.3.1 +buffer_recycling +buffer_content_recycling +executor_recycling max_number_gpus=8 +allocator_counters ^silo~mpi
spack load octotiger@master build_type=Release +kokkos +cuda cuda_arch=80 %gcc@11 ^kokkos@4.0.01 ^hpx@1.9.1 max_cpu_count=256 networking=none ~async_gpu_futures ^cppuddle@0.3.1 +buffer_recycling +buffer_content_recycling +executor_recycling max_number_gpus=8 +allocator_counters ^silo~mpi
for cores in ${corelist}; do
      echo "Running with ${cores} threads..." | tee -a ${scenario_name}_benchmark.log
      octotiger --hpx:threads=${cores} ${hpx_parameters} ${scenario_parameters} --executors_per_gpu=${executors} --max_kernels_fused=${slices} ${kernel_parameters} ${counter_parameters} | tee current_output.log
      cat current_output.log >> ${scenario_name}_benchmark.log
      output_metrics=$(extract_times_and_counters "current_output.log")
      echo "${scenario_name}_with_buffer_executor_recycling, ${max_level}, ${cores}, ${slices}, ${executors}, ${output_metrics} " \
           | tee -a ${scenario_name}_benchmark.csv
done
spack unload

# 4. Run with more executors 
# -------------------------------------
executors=64
echo "Starting testing with buffer,executor recycling and more executors" | tee -a ${scenario_name}_benchmark.log
echo "Installing spec..." | tee -a ${scenario_name}_benchmark.log
spack install --fresh octotiger@master build_type=Release +kokkos +cuda cuda_arch=80 %gcc@11 ^kokkos@4.0.01 ^hpx@1.9.1 max_cpu_count=256 networking=none ~async_gpu_futures ^cppuddle@0.3.1 +buffer_recycling +buffer_content_recycling +executor_recycling max_number_gpus=8 +allocator_counters ^silo~mpi
spack load octotiger@master build_type=Release +kokkos +cuda cuda_arch=80 %gcc@11 ^kokkos@4.0.01 ^hpx@1.9.1 max_cpu_count=256 networking=none ~async_gpu_futures ^cppuddle@0.3.1 +buffer_recycling +buffer_content_recycling +executor_recycling max_number_gpus=8 +allocator_counters ^silo~mpi
for cores in ${corelist}; do
      echo "Running with ${cores} threads..." | tee -a ${scenario_name}_benchmark.log
      octotiger --hpx:threads=${cores} ${hpx_parameters} ${scenario_parameters} --executors_per_gpu=${executors} --max_kernels_fused=${slices} ${kernel_parameters} ${counter_parameters} | tee current_output.log
      cat current_output.log >> ${scenario_name}_benchmark.log
      output_metrics=$(extract_times_and_counters "current_output.log")
      echo "${scenario_name}_with_buffer_executor_recycling_more_executors, ${max_level}, ${cores}, ${slices}, ${executors}, ${output_metrics} " \
           | tee -a ${scenario_name}_benchmark.csv
done
spack unload

# 5. Run with with work aggregation
# -------------------------------------
slices=8
echo "Starting testing with buffer,executor recycling and more executors and slices" | tee -a ${scenario_name}_benchmark.log
echo "Installing spec..." | tee -a ${scenario_name}_benchmark.log
spack install --fresh octotiger@master build_type=Release +kokkos +cuda cuda_arch=80 %gcc@11 ^kokkos@4.0.01 ^hpx@1.9.1 max_cpu_count=256 networking=none ~async_gpu_futures ^cppuddle@0.3.1 +buffer_recycling +buffer_content_recycling +executor_recycling max_number_gpus=8 +allocator_counters ^silo~mpi
spack load octotiger@master build_type=Release +kokkos +cuda cuda_arch=80 %gcc@11 ^kokkos@4.0.01 ^hpx@1.9.1 max_cpu_count=256 networking=none ~async_gpu_futures ^cppuddle@0.3.1 +buffer_recycling +buffer_content_recycling +executor_recycling max_number_gpus=8 +allocator_counters ^silo~mpi
for cores in ${corelist}; do
      echo "Running with ${cores} threads..." | tee -a ${scenario_name}_benchmark.log
      octotiger --hpx:threads=${cores} ${hpx_parameters} ${scenario_parameters} --executors_per_gpu=${executors} --max_kernels_fused=${slices} ${kernel_parameters} ${counter_parameters} | tee current_output.log
      cat current_output.log >> ${scenario_name}_benchmark.log
      output_metrics=$(extract_times_and_counters "current_output.log")
      echo "${scenario_name}_with_buffer_executor_recycling_more_executors_slices, ${max_level}, ${cores}, ${slices}, ${executors}, ${output_metrics} " \
           | tee -a ${scenario_name}_benchmark.csv
done
spack unload

# 6. Run with async futures
# -------------------------------------
echo "Starting testing with buffer,executor recycling and more executors and slices and async futures" | tee -a ${scenario_name}_benchmark.log
echo "Installing spec..." | tee -a ${scenario_name}_benchmark.log
spack install --fresh octotiger@master build_type=Release +kokkos +cuda cuda_arch=80 %gcc@11 ^kokkos@4.0.01 ^hpx@1.9.1 max_cpu_count=256 networking=none +async_gpu_futures ^cppuddle@0.3.1 +buffer_recycling +buffer_content_recycling +executor_recycling max_number_gpus=8 +allocator_counters ^silo~mpi
spack load octotiger@master build_type=Release +kokkos +cuda cuda_arch=80 %gcc@11 ^kokkos@4.0.01 ^hpx@1.9.1 max_cpu_count=256 networking=none +async_gpu_futures ^cppuddle@0.3.1 +buffer_recycling +buffer_content_recycling +executor_recycling max_number_gpus=8 +allocator_counters ^silo~mpi
for cores in ${corelist}; do
      echo "Running with ${cores} threads..." | tee -a ${scenario_name}_benchmark.log
      octotiger --hpx:threads=${cores} ${hpx_parameters} ${scenario_parameters} --executors_per_gpu=${executors} --max_kernels_fused=${slices} ${kernel_parameters} ${counter_parameters} | tee current_output.log
      cat current_output.log >> ${scenario_name}_benchmark.log
      output_metrics=$(extract_times_and_counters "current_output.log")
      echo "${scenario_name}_with_buffer_executor_recycling_more_executors_slices_async_fut, ${max_level}, ${cores}, ${slices}, ${executors}, ${output_metrics} " \
           | tee -a ${scenario_name}_benchmark.csv
done
spack unload
