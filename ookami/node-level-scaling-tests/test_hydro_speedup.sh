#!/bin/bash
set -eux
IFS=$'\n\t'

# Timestamp
today=`date +%Y-%m-%d_%H:%M`
# for measuring
start=`date +%s`

# Experiment configuration (obsolete, gets sourced from extra file below)
#experiment_description="Run rotating star node-level scaling with KOKKOS SIMD and STD SIMD on Intel icelake"
#apex_args=" APEX_SCREEN_OUTPUT=1 APEX_CSV_OUTPUT=1 KOKKOS_PROFILE_LIBRARY=build/hpx/lib/libhpx_apex.so "
#kernel_args=" --hydro_device_kernel_type=OFF --multipole_device_kernel_type=OFF --monopole_device_kernel_type=OFF --hydro_host_kernel_type=KOKKOS --multipole_host_kernel_type=KOKKOS --monopole_host_kernel_type=KOKKOS --amr_boundary_kernel_type=AMR_OPTIMIZED --optimize_local_communication=1"
#octotiger_args="--config_file=src/octotiger/test_problems/rotating_star/rotating_star.ini --unigrid=1 --max_level=3 --stop_step=10 --correct_am_hydro=0 --theta=0.34 --disable_output=1 "
#simd_extensions="SCALAR AVX SVE"
#simd_libraries="KOKKOS STD"
#NodeList="8 16 32 48"
#output_file="icelake_simd_test_$today"
#debug_output_file="debug_icelake_simd_test_$today"


cpu_platform="$(lscpu | grep "Model name:" | sed 's/Model name: *//')"
output_file="legacy_test_$today"
debug_output_file="debug_legacy_test_$today"

function extract_compute_time {
  local compute_time=$(echo $1 | grep 'Computation:' | sed 's/Computation://g' | sed 's/(.*)//g')
  echo "${compute_time}"
}
function extract_total_time {
  local total_time=$(echo $1 | grep 'Total:' | sed 's/Total://g' | sed 's/(.*)//g')
  echo "${total_time}"
}

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
echo "# SIMD extensions ${simd_extensions}:" | tee -a LOG.txt
echo "# SIMD libraries ${simd_libraries}:" | tee -a LOG.txt
echo "# Number cores: ${NodeList}" | tee -a LOG.txt
echo "# " | tee -a LOG.txt
echo "# " | tee -a LOG.txt
echo "# Experiment:" | tee -a LOG.txt
echo "# cpu, SIMD library, SIMD extension, cores, function name, total calls, runtime of all calls, average runtime per call" | tee -a LOG.txt
echo "DEBUG: Starting DEBUG-LOG.txt..." > DEBUG-LOG.txt

export APEX_SCREEN_OUTPUT=1 
export APEX_CSV_OUTPUT=1 
export KOKKOS_PROFILE_LIBRARY=$(pwd)/build/hpx/lib/libhpx_apex.so
IFS=' '

sed -i "s/-DOCTOTIGER_KOKKOS_SIMD_LIBRARY=.*/-DOCTOTIGER_KOKKOS_SIMD_LIBRARY=KOKKOS \\\\/g" build-octotiger.sh
sed -i "s/-DOCTOTIGER_KOKKOS_SIMD_EXTENSION=.*/-DOCTOTIGER_KOKKOS_SIMD_EXTENSION=SCALAR \\\\/g" build-octotiger.sh
./build-all.sh Release with-CC without-cuda without-mpi without-papi with-apex with-kokkos with-simd with-hpx-backend-multipole with-hpx-backend-monopole with-hpx-cuda-polling without-otf2 octotiger

# Run legacy all cores
i=48
lib=LEGACY
extension=LEGACY
output_legacy_all_cores="$(./build/octotiger/build/octotiger -t${i} --config_file=src/octotiger/test_problems/blast/blast.ini --unigrid=1 --cuda_number_gpus=1 --disable_output=on --max_level=3 --stop_step=15 --cuda_streams_per_gpu=128 --cuda_buffer_capacity=1024 --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY --amr_boundary_kernel_type=AMR_OPTIMIZED --optimize_local_communication=1)"
echo "DEBUG: ${output_legacy_all_cores}" >> DEBUG-LOG.txt
compute_time=$(extract_compute_time "${output_legacy_all_cores}")
compute_time_entry="${cpu_platform},${lib},${extension},$i,Octo-Tiger Compute time,1,${compute_time},${compute_time}"
total_time=$(extract_total_time "${output_legacy_all_cores}")
total_time_entry="${cpu_platform},${lib},${extension},$i,Octo-Tiger Total time,1,${total_time},${total_time}"
echo "$compute_time_entry" | tee -a LOG.txt
echo "$total_time_entry" | tee -a LOG.txt
kernel_times=$(cat apex.0.csv | grep "kernel " | awk -F',' -v cores=${i} -v lib=${lib} -v extension=${extension} -v cpu_platform="${cpu_platform}"  -v OFS=',' '{ print cpu_platform,lib,extension,cores,$1,$3,$8/1000000,$8/$3/1000000 }')
echo "$kernel_times" | tee -a LOG.txt

# Run Kokkos scalar all cores
i=48
lib=STD
extension=SCALAR
output_scalar_all_cores="$(./build/octotiger/build/octotiger -t${i} --config_file=src/octotiger/test_problems/blast/blast.ini --unigrid=1 --cuda_number_gpus=1 --disable_output=on --max_level=3 --stop_step=15 --cuda_streams_per_gpu=128 --cuda_buffer_capacity=1024 --hydro_device_kernel_type=OFF --hydro_host_kernel_type=KOKKOS --amr_boundary_kernel_type=AMR_OPTIMIZED --optimize_local_communication=1)"
echo "DEBUG: ${output_scalar_all_cores}" >> DEBUG-LOG.txt
compute_time=$(extract_compute_time "${output_scalar_all_cores}")
compute_time_entry="${cpu_platform},${lib},${extension},$i,Octo-Tiger Compute time,1,${compute_time},${compute_time}"
total_time=$(extract_total_time "${output_scalar_all_cores}")
total_time_entry="${cpu_platform},${lib},${extension},$i,Octo-Tiger Total time,1,${total_time},${total_time}"
echo "$compute_time_entry" | tee -a LOG.txt
echo "$total_time_entry" | tee -a LOG.txt
kernel_times=$(cat apex.0.csv | grep "kernel " | awk -F',' -v cores=${i} -v lib=${lib} -v extension=${extension} -v cpu_platform="${cpu_platform}"  -v OFS=',' '{ print cpu_platform,lib,extension,cores,$1,$3,$8/1000000,$8/$3/1000000 }')
echo "$kernel_times" | tee -a LOG.txt

# Run legacy 1 cores
i=1
lib=LEGACY
extension=LEGACY
output_legacy_all_cores="$(./build/octotiger/build/octotiger -t${i} --config_file=src/octotiger/test_problems/blast/blast.ini --unigrid=1 --cuda_number_gpus=1 --disable_output=on --max_level=3 --stop_step=15 --cuda_streams_per_gpu=128 --cuda_buffer_capacity=1024 --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY --amr_boundary_kernel_type=AMR_OPTIMIZED --optimize_local_communication=1)"
echo "DEBUG: ${output_legacy_all_cores}" >> DEBUG-LOG.txt
compute_time=$(extract_compute_time "${output_legacy_all_cores}")
compute_time_entry="${cpu_platform},${lib},${extension},$i,Octo-Tiger Compute time,1,${compute_time},${compute_time}"
total_time=$(extract_total_time "${output_legacy_all_cores}")
total_time_entry="${cpu_platform},${lib},${extension},$i,Octo-Tiger Total time,1,${total_time},${total_time}"
echo "$compute_time_entry" | tee -a LOG.txt
echo "$total_time_entry" | tee -a LOG.txt
kernel_times=$(cat apex.0.csv | grep "kernel " | awk -F',' -v cores=${i} -v lib=${lib} -v extension=${extension} -v cpu_platform="${cpu_platform}"  -v OFS=',' '{ print cpu_platform,lib,extension,cores,$1,$3,$8/1000000,$8/$3/1000000 }')
echo "$kernel_times" | tee -a LOG.txt

# Run Kokkos scalar one cores
i=1
lib=STD
extension=SCALAR
output_scalar_all_cores="$(./build/octotiger/build/octotiger -t${i} --config_file=src/octotiger/test_problems/blast/blast.ini --unigrid=1 --cuda_number_gpus=1 --disable_output=on --max_level=3 --stop_step=15 --cuda_streams_per_gpu=128 --cuda_buffer_capacity=1024 --hydro_device_kernel_type=OFF --hydro_host_kernel_type=KOKKOS --amr_boundary_kernel_type=AMR_OPTIMIZED --optimize_local_communication=1)"
echo "DEBUG: ${output_scalar_all_cores}" >> DEBUG-LOG.txt
compute_time=$(extract_compute_time "${output_scalar_all_cores}")
compute_time_entry="${cpu_platform},${lib},${extension},$i,Octo-Tiger Compute time,1,${compute_time},${compute_time}"
total_time=$(extract_total_time "${output_scalar_all_cores}")
total_time_entry="${cpu_platform},${lib},${extension},$i,Octo-Tiger Total time,1,${total_time},${total_time}"
echo "$compute_time_entry" | tee -a LOG.txt
echo "$total_time_entry" | tee -a LOG.txt
kernel_times=$(cat apex.0.csv | grep "kernel " | awk -F',' -v cores=${i} -v lib=${lib} -v extension=${extension} -v cpu_platform="${cpu_platform}"  -v OFS=',' '{ print cpu_platform,lib,extension,cores,$1,$3,$8/1000000,$8/$3/1000000 }')
echo "$kernel_times" | tee -a LOG.txt

sed -i "s/-DOCTOTIGER_KOKKOS_SIMD_LIBRARY=.*/-DOCTOTIGER_KOKKOS_SIMD_LIBRARY=STD \\\\/g" build-octotiger.sh
sed -i "s/-DOCTOTIGER_KOKKOS_SIMD_EXTENSION=.*/-DOCTOTIGER_KOKKOS_SIMD_EXTENSION=SVE \\\\/g" build-octotiger.sh
./build-all.sh Release with-CC without-cuda without-mpi without-papi with-apex with-kokkos with-simd with-hpx-backend-multipole with-hpx-backend-monopole with-hpx-cuda-polling without-otf2 octotiger

# Run Kokkos simd all cores
i=48
lib=STD
extension=SVE
output_scalar_all_cores="$(./build/octotiger/build/octotiger -t${i} --config_file=src/octotiger/test_problems/blast/blast.ini --unigrid=1 --cuda_number_gpus=1 --disable_output=on --max_level=3 --stop_step=15 --cuda_streams_per_gpu=128 --cuda_buffer_capacity=1024 --hydro_device_kernel_type=OFF --hydro_host_kernel_type=KOKKOS --amr_boundary_kernel_type=AMR_OPTIMIZED --optimize_local_communication=1)"
echo "DEBUG: ${output_scalar_all_cores}" >> DEBUG-LOG.txt
compute_time=$(extract_compute_time "${output_scalar_all_cores}")
compute_time_entry="${cpu_platform},${lib},${extension},$i,Octo-Tiger Compute time,1,${compute_time},${compute_time}"
total_time=$(extract_total_time "${output_scalar_all_cores}")
total_time_entry="${cpu_platform},${lib},${extension},$i,Octo-Tiger Total time,1,${total_time},${total_time}"
echo "$compute_time_entry" | tee -a LOG.txt
echo "$total_time_entry" | tee -a LOG.txt
kernel_times=$(cat apex.0.csv | grep "kernel " | awk -F',' -v cores=${i} -v lib=${lib} -v extension=${extension} -v cpu_platform="${cpu_platform}"  -v OFS=',' '{ print cpu_platform,lib,extension,cores,$1,$3,$8/1000000,$8/$3/1000000 }')
echo "$kernel_times" | tee -a LOG.txt

# Run Kokkos simd one cores
i=1
lib=STD
extension=SVE
output_scalar_all_cores="$(./build/octotiger/build/octotiger -t${i} --config_file=src/octotiger/test_problems/blast/blast.ini --unigrid=1 --cuda_number_gpus=1 --disable_output=on --max_level=3 --stop_step=15 --cuda_streams_per_gpu=128 --cuda_buffer_capacity=1024 --hydro_device_kernel_type=OFF --hydro_host_kernel_type=KOKKOS --amr_boundary_kernel_type=AMR_OPTIMIZED --optimize_local_communication=1)"
echo "DEBUG: ${output_scalar_all_cores}" >> DEBUG-LOG.txt
compute_time=$(extract_compute_time "${output_scalar_all_cores}")
compute_time_entry="${cpu_platform},${lib},${extension},$i,Octo-Tiger Compute time,1,${compute_time},${compute_time}"
total_time=$(extract_total_time "${output_scalar_all_cores}")
total_time_entry="${cpu_platform},${lib},${extension},$i,Octo-Tiger Total time,1,${total_time},${total_time}"
echo "$compute_time_entry" | tee -a LOG.txt
echo "$total_time_entry" | tee -a LOG.txt
kernel_times=$(cat apex.0.csv | grep "kernel " | awk -F',' -v cores=${i} -v lib=${lib} -v extension=${extension} -v cpu_platform="${cpu_platform}"  -v OFS=',' '{ print cpu_platform,lib,extension,cores,$1,$3,$8/1000000,$8/$3/1000000 }')
echo "$kernel_times" | tee -a LOG.txt


# Total experiment runtime:
end=`date +%s` # for measuring
runtime=$((end-start))
echo "# " | tee -a LOG.txt
echo "# " | tee -a LOG.txt
echo "# Total experiment runtime: ${runtime}" | tee -a LOG.txt
echo "DEBUG: Experiment ended sucessfully!" >> DEBUG-LOG.txt
cp LOG.txt ${output_file}
cp DEBUG-LOG.txt ${debug_output_file}


