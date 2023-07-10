#!/bin/bash
set -eux
IFS=$'\n\t'

function extract_compute_time {
	local compute_time=$(echo $1 | grep 'Computation:' | sed 's/Computation://g' | sed 's/(.*)//g')
	echo "${compute_time}"
}	
function extract_total_time {
	local total_time=$(echo $1 | grep 'Total:' | sed 's/Total://g' | sed 's/(.*)//g')
	echo "${total_time}"
}	

# Timestamp
today=`date +%Y-%m-%d_%H:%M`
# for measuring
start=`date +%s`

# Experiment configuration (obsolete, gets sourced from extra file below)
#experiment_description="Run rotating star node-level scaling with KOKKOS SIMD and STD SIMD on Intel icelake"
apex_args=" APEX_SCREEN_OUTPUT=1 APEX_CSV_OUTPUT=1 KOKKOS_PROFILE_LIBRARY=build/hpx/lib/libhpx_apex.so "
kernel_args=" --hydro_device_kernel_type=OFF --multipole_device_kernel_type=OFF --monopole_device_kernel_type=OFF --hydro_host_kernel_type=KOKKOS --multipole_host_kernel_type=KOKKOS --monopole_host_kernel_type=KOKKOS --amr_boundary_kernel_type=AMR_OPTIMIZED --optimize_local_communication=1"
octotiger_args=" --config_file=src/octotiger/test_problems/blast/blast.ini --unigrid=1 --max_level=3 --stop_step=15 --stop_time=25 --disable_output=1 "
#cpu_platform="$(lscpu | grep "Model name:" | sed 's/Model name: *//')"
#simd_extensions="SCALAR AVX AVX512"
#simd_libraries="KOKKOS STD"
#NodeList="8 16 32 64"
#output_file="icelake_simd_test_$today"
#debug_output_file="debug_icelake_simd_test_$today"

#gpu_backends="CUDA OFF"
#max_worker_list="1 2 3 4 8 16 32 64"
#max_gpu_list="1 2 4"
#core_iterate_list="1 2 3 4 8 16 32 64"

disable_recycling_modes="OFF ON "
hpx_aware_allocators_modes="ON OFF "
gpu_backends="OFF_LEGACY OFF_KOKKOS CUDA KOKKOS_CUDA "
max_core_count=40
max_worker_list="10 20 40"
max_gpu_list="1 2"
core_iterate_list="10 20 40"

workers_per_gpu="1 2 4 8 10 20 40"

#echo "Loading config file $1 ..."
#source "$1"

# Software config
toolchain_commit="$(git log --oneline | head -n 1)"
cd src/octotiger
octotiger_commit="$(git log --oneline | head -n 1)"
cd -
cd src/hpx
hpx_commit="$(git log --oneline | head -n 1)"
cd -
cd src/kokkos
kokkos_commit="$(git log --oneline | head -n 1)"
cd -
cd src/hpx-kokkos
hpxkokkos_commit="$(git log --oneline | head -n 1)"
cd -
echo "# Date of run $today" | tee LOG.txt
echo "# " | tee -a LOG.txt
echo "# " | tee -a LOG.txt
echo "# Octotiger Args: ${octotiger_args}" | tee -a LOG.txt
echo "# Kernel Args: ${kernel_args}" | tee -a LOG.txt
echo "# " | tee -a LOG.txt
echo "# Buildscripts Commit: ${toolchain_commit}" | tee -a LOG.txt
echo "# Octotiger Commit: ${octotiger_commit}" | tee -a LOG.txt
echo "# HPX Commit: ${hpx_commit}" | tee -a LOG.txt
echo "# Kokkos Commit: ${kokkos_commit}" | tee -a LOG.txt
echo "# HPX-Kokkos Commit: ${hpxkokkos_commit}" | tee -a LOG.txt
echo "# " | tee -a LOG.txt
echo "# " | tee -a LOG.txt
echo "# Kernel args: ${kernel_args}" | tee -a LOG.txt
echo "# Octo-Tiger Scenario: ${octotiger_args}" | tee -a LOG.txt
echo "# " | tee -a LOG.txt
echo "# " | tee -a LOG.txt
echo "# Experiment:" | tee -a LOG.txt
echo "# host kernel type, gpu kernel type, recycling mode, hpx aware allocators, hpx mutex mode, max workers, number gpus, workers per gpu, cores, computation time, total time" | tee -a LOG.txt
echo "DEBUG: Starting DEBUG-LOG.txt..." > DEBUG-LOG.txt

module load cuda llvm/12 hwloc cmake
#rm -rf build src/hdf5 
#./build-all.sh Release with-CC-clang with-cuda without-mpi without-papi with-apex with-kokkos with-simd with-hpx-backend-multipole with-hpx-backend-monopole with-hpx-cuda-polling without-otf2 boost hdf5 silo jemalloc vc hpx kokkos cppuddle octotiger

export APEX_SCREEN_OUTPUT=1 
export APEX_CSV_OUTPUT=1 
#export KOKKOS_PROFILE_LIBRARY=$(pwd)/build/hpx/lib64/libhpx_apex.so


# Run scaling test
IFS=' '
for recycling_mode in ${disable_recycling_modes}; do
for hpx_aware_allocators_mode in ${hpx_aware_allocators_modes}; do
for gpu_kernel_type in ${gpu_backends}; do
  host_kernel_type="DEVICE_ONLY"
  max_slices=4
  if [ "${gpu_kernel_type}" = "OFF_LEGACY" ]; then
    host_kernel_type="LEGACY"
    gpu_kernel_type="OFF"
    max_slices=1
  fi
  if [ "${gpu_kernel_type}" = "OFF_KOKKOS" ]; then
    host_kernel_type="KOKKOS"
    gpu_kernel_type="OFF"
    max_slices=1
  fi
  kernel_args=" --hydro_device_kernel_type=${gpu_kernel_type} --multipole_device_kernel_type=${gpu_kernel_type} --monopole_device_kernel_type=${gpu_kernel_type} --hydro_host_kernel_type=${host_kernel_type} --multipole_host_kernel_type=${host_kernel_type} --monopole_host_kernel_type=${host_kernel_type} --amr_boundary_kernel_type=AMR_OPTIMIZED --optimize_local_communication=1 --cuda_streams_per_gpu=32 --max_executor_slices=${max_slices}"
  # iterate maximum workers
  for max_worker in ${max_worker_list}; do
    # early exit for cpu-only run
    if [[ "${host_kernel_type}" != "DEVICE_ONLY" ]]; then
      if [[ ${max_worker} -lt ${max_core_count} ]]; then
        continue
      fi
    fi
    # iterate gpus
    for number_gpus in ${max_gpu_list}; do
      # early exit for cpu-only run
      if [[ "${host_kernel_type}" != "DEVICE_ONLY" ]]; then
        if [[ ${number_gpus} -gt 1 ]]; then
          break
        fi
      fi
      # early exit for non-hpxWARE
      if [[ "${hpx_aware_allocators_mode}" != "OFF" ]]; then
        if [[ ${number_gpus} -gt 1 ]]; then
          break
        fi
      fi
      sed -i "s/-DCPPUDDLE_WITH_MAX_NUMBER_WORKERS=.*/-DCPPUDDLE_WITH_MAX_NUMBER_WORKERS=${max_worker} \\\\/g" build-cppuddle.sh
      sed -i "s/-DCPPUDDLE_WITH_NUMBER_GPUS=.*/-DCPPUDDLE_WITH_NUMBER_GPUS=${number_gpus} \\\\/g" build-cppuddle.sh
      sed -i "s/-DCPPUDDLE_DEACTIVATE_BUFFER_RECYCLING=.*/-DCPPUDDLE_DEACTIVATE_BUFFER_RECYCLING=${recycling_mode} \\\\/g" build-cppuddle.sh
      sed -i "s/-DCPPUDDLE_WITH_HPX_AWARE_ALLOCATORS=.*/-DCPPUDDLE_WITH_HPX_AWARE_ALLOCATORS=${hpx_aware_allocators_mode} \\\\/g" build-cppuddle.sh
      sed -i "s/-DCPPUDDLE_WITH_HPX=.*/-DCPPUDDLE_WITH_HPX=${hpx_aware_allocators_mode} \\\\/g" build-cppuddle.sh
      sed -i "s/-DCPPUDDLE_WITH_HPX_MUTEX=.*/-DCPPUDDLE_WITH_HPX_MUTEX=${hpx_aware_allocators_mode} \\\\/g" build-cppuddle.sh
      if [[ "${hpx_aware_allocators_mode}" == "OFF" ]]; then
        if [[ ${number_gpus} -gt 1 ]]; then
          break
        fi
        sed -i "s/-DCPPUDDLE_WITH_MAX_NUMBER_WORKERS=.*/-DCPPUDDLE_WITH_MAX_NUMBER_WORKERS=1 \\\\/g" build-cppuddle.sh
      fi
        ./build-all.sh Release with-CC-clang with-cuda without-mpi without-papi with-apex with-kokkos with-simd with-hpx-backend-multipole with-hpx-backend-monopole with-hpx-cuda-polling without-otf2 cppuddle octotiger
      # Number of cores used (up until max worker is hit)
      for number_cores in ${core_iterate_list}; do
        cores_per_gpu=$((max_worker/number_gpus))
        echo "${gpu_kernel_type},${max_worker},${number_gpus},${cores_per_gpu},${number_cores}"
        output1="$(build/octotiger/build/octotiger --hpx:threads=${number_cores} ${octotiger_args} ${kernel_args})"
        echo "DEBUG: ${output1}" >> DEBUG-LOG.txt
        compute_time=$(extract_compute_time "${output1}")
        total_time=$(extract_total_time "${output1}")
        compute_time_entry="${host_kernel_type},${gpu_kernel_type},${recycling_mode},${hpx_aware_allocators_mode},${max_worker},${number_gpus},${cores_per_gpu},${number_cores},${compute_time}, ${total_time}"
        echo "$compute_time_entry" | tee -a LOG.txt
        if [[ ${number_cores} -eq ${max_worker} ]]; then
          break
        fi
      done
    done
  done
done
done
done 

# Total experiment runtime:
end=`date +%s` # for measuring
runtime=$((end-start))
echo "# " | tee -a LOG.txt
echo "# " | tee -a LOG.txt
echo "# Total experiment runtime: ${runtime}" | tee -a LOG.txt
echo "DEBUG: Experiment ended sucessfully!" >> DEBUG-LOG.txt
cp LOG.txt ${output_file}
cp DEBUG-LOG.txt ${debug_output_file}


