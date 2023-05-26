#!/bin/bash -l


module load llvm/12.0.1
module unload boost
module load hwloc cuda

# Timestamp
today=`date +%Y-%m-%d_%H:%M:%S`
# for measuring
start=`date +%s`

# Maschine
max_level=3
stop_step=15

# software config
toolchain_commit="$(git log --oneline | head -n 1)"
cd src/octotiger
octotiger_commit="$(git log --oneline | head -n 1)"
cd ../..
cd src/hpx
hpx_commit="$(git log --oneline | head -n 1)"
cd ../..
cd src/kokkos
kokkos_commit="$(git log --oneline | head -n 1)"
cd ../..
cd src/hpx-kokkos
hpxkokkos_commit="$(git log --oneline | head -n 1)"
cd ../..
cd src/cppuddle
cppuddle_commit="$(git log --oneline | head -n 1)"
cd ../..

log_filename=blast_aggregation_test_cuda_cpuamr_${max_level}_${stop_step}_${today}_$(hostname)_LOG.txt
debug_log_filename=blast_aggregation_test_cuda_cpuamr_${max_level}_${stop_step}_${today}_$(hostname)_DEBUG_LOG.txt

# Default Kernel configuration
kernel_args="--cuda_streams_per_gpu=128 --cuda_buffer_capacity=1024 --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=DEVICE_ONLY --amr_boundary_kernel_type=AMR_OPTIMIZED --max_executor_slices=1 --cuda_number_gpus=1 "
# Scenario
octotiger_args="--config_file=src/octotiger/test_problems/blast/blast.ini --unigrid=1 --disable_output=on --max_level=${max_level} --stop_step=${stop_step}"
# HPX configuration without the threads argument
hpx_args="-Ihpx.scheduler=local-priority-lifo -Ihpx.stacks.use_guard_pages=0"

echo "# Date of run $today" | tee ${log_filename}
echo "# Running aggregation test on $(hostname)" | tee -a ${log_filename}
echo "# " | tee -a ${log_filename}
echo "# Octotiger Scenario Args: ${octotiger_args}" | tee -a ${log_filename}
echo "# Default Kernel Args: ${kernel_args}" | tee -a ${log_filename}
echo "# HPX Config Args: ${hpx_args}" | tee -a ${log_filename}
echo "# " | tee -a ${log_filename}
echo "# Buildscripts Commit: ${toolchain_commit}" | tee -a ${log_filename}
echo "# Octotiger Commit: ${octotiger_commit}" | tee -a ${log_filename}
echo "# HPX Commit: ${hpx_commit}" | tee -a ${log_filename}
echo "# Kokkos Commit: ${kokkos_commit}" | tee -a ${log_filename}
echo "# HPX-Kokkos Commit: ${hpxkokkos_commit}" | tee -a ${log_filename}
echo "# CPPuddle Commit: ${cppuddle_commit}" | tee -a ${log_filename}
echo "# " | tee -a ${log_filename}
echo "# " | tee -a ${log_filename}
echo "# Experiment:" | tee -a ${log_filename}
echo "# cores, executors, max slices, computation time (s), total time (s), number reconstruct launches, avg reconstruct time (ns), number_flux_launches, avg flux time (ns), number_discs_phase1 launches, avg discs phase1 time (ns), number_discs_phase2 launches, avg discs phase2 time (ns), number_pre recon_launches, avg pre_recon time (ns), profiler run computation time(s), profiler run total time (s)" | tee -a ${log_filename}
echo "DEBUG: Starting ${debug_log_filename}..." > ${debug_log_filename}

CoreList="1 2 4 8 16 32"
ExecutorList="0 1 2 4 8 16 32 64 128"
AggregationSizes="1 2 4 8 16 32 64"

for cores in ${CoreList}; do
  for executors in $ExecutorList; do
    if (( ${executors} == 0 )) ; then
      echo "DEBUG: Starting cpu-only run with ${cores} cores ..." >> ${debug_log_filename}
      kernel_args=" --hpx:threads=${cores} --cuda_streams_per_gpu=${executors} --cuda_buffer_capacity=1024 --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY --amr_boundary_kernel_type=AMR_OPTIMIZED --max_executor_slices=1 --cuda_number_gpus=1 "
      output1="$(./build/octotiger/build/octotiger ${hpx_args} ${kernel_args} ${octotiger_args} )"
      echo "DEBUG: ${output1}" >> ${debug_log_filename}
      compute_time="${cores}, 0, 1, $(echo "$output1" | grep "Computation: " | sed 's/   Computation: //g' | sed 's/ (.*)//g'), $(echo "$output1" | grep "Total: " | sed 's/   Total: //g'), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"
      echo "$compute_time" | tee -a ${log_filename}
    else
      for slices in $AggregationSizes; do
        echo "DEBUG: Starting normal run with ${cores} cores, ${executors} executors and ${slices} aggregation slices ..." >> ${debug_log_filename}
      kernel_args=" --hpx:threads=${cores} --cuda_streams_per_gpu=${executors} --cuda_buffer_capacity=1024 --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=DEVICE_ONLY --amr_boundary_kernel_type=AMR_OPTIMIZED --max_executor_slices=${slices} --cuda_number_gpus=1 "
        output1="$(./build/octotiger/build/octotiger ${hpx_args} ${kernel_args} ${octotiger_args} )"
        echo "DEBUG: ${output1}" >> ${debug_log_filename}
        # cleanup
        output_nsys="$(nsys profile --stats=true -o todelete --force-overwrite true ./build/octotiger/build/octotiger ${hpx_args} ${kernel_args} ${octotiger_args} )"
        echo "DEBUG: ${output_nsys}" >> ${debug_log_filename}
        # cleanup
        rm todelete.nsys-rep
        rm todelete.sqlite
        # Create data entry in the csv file for this run
        compute_time="${cores}, ${executors}, ${slices}, \
$(echo "$output1" | grep "Computation: " | sed 's/   Computation: //g' | sed 's/ (.*)//g'), \
$(echo "$output1" | grep "Total: " | sed 's/   Total: //g'), \
$(echo "$output_nsys" | grep "reconstruct" | awk '{ print $3 }' | sed 's/,//g'), \
$(echo "$output_nsys" | grep "reconstruct" | awk '{ print $4 }' | sed 's/,//g'), \
$(echo "$output_nsys" | grep "flux" | awk '{ print $3 }' | sed 's/,//g'), \
$(echo "$output_nsys" | grep "flux" | awk '{ print $4 }' | sed 's/,//g'), \
$(echo "$output_nsys" | grep "discs_phase1" | awk '{ print $3 }' | sed 's/,//g'), \
$(echo "$output_nsys" | grep "discs_phase1" | awk '{ print $4 }' | sed 's/,//g'), \
$(echo "$output_nsys" | grep "discs_phase2" | awk '{ print $3 }' | sed 's/,//g'), \
$(echo "$output_nsys" | grep "discs_phase2" | awk '{ print $4 }' | sed 's/,//g'), \
$(echo "$output_nsys" | grep "pre_recon" | awk '{ print $3 }' | sed 's/,//g'), \
$(echo "$output_nsys" | grep "pre_recon" | awk '{ print $4 }' | sed 's/,//g'), \
$(echo "$output_nsys" | grep "Computation: " | sed 's/   Computation: //g' | sed 's/ (.*)//g'), \
$(echo "$output_nsys" | grep "Total: " | sed 's/   Total: //g')"
        echo "$compute_time" | tee -a ${log_filename}
      done
    fi
  done 
done

cp ${log_filename} performance_results.log
cat performance_results.log
