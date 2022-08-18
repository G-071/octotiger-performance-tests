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
#cpu_platform="$(lscpu | grep "Model name:" | sed 's/Model name: *//')"
#simd_extensions="SCALAR AVX AVX512"
#simd_libraries="KOKKOS STD"
#NodeList="8 16 32 64"
#output_file="icelake_simd_test_$today"
#debug_output_file="debug_icelake_simd_test_$today"

echo "Loading config file $1 ..."
source "$1"

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


# Run scaling test
IFS=' '
for extension in ${simd_extensions}; do
  for lib in ${simd_libraries}; do
    echo $lib
    sed -i "s/-DOCTOTIGER_KOKKOS_SIMD_LIBRARY=.*/-DOCTOTIGER_KOKKOS_SIMD_LIBRARY=${lib} \\\\/g" build-octotiger.sh
    sed -i "s/-DOCTOTIGER_KOKKOS_SIMD_EXTENSION=.*/-DOCTOTIGER_KOKKOS_SIMD_EXTENSION=${extension} \\\\/g" build-octotiger.sh
    ./build-all.sh Release with-CC without-cuda without-mpi without-papi with-apex with-kokkos with-simd with-hpx-backend-multipole with-hpx-backend-monopole with-hpx-cuda-polling without-otf2 octotiger
    for i in $NodeList; do
      echo "DEBUG: Starting run $i ..." >> DEBUG-LOG.txt
      output1="$(build/octotiger/build/octotiger -t$i ${octotiger_args} ${kernel_args})"
      echo "DEBUG: ${output1}" >> DEBUG-LOG.txt
      compute_time=$(extract_compute_time "${output1}")
      compute_time_entry="${cpu_platform},${lib},${extension},Octo-Tiger Compute time,$i,1,${compute_time},${compute_time}"
      total_time=$(extract_total_time "${output1}")
      total_time_entry="${cpu_platform},${lib},${extension},Octo-Tiger Total time,$i,1,${total_time},${total_time}"
      echo "$compute_time_entry" | tee -a LOG.txt
      echo "$total_time_entry" | tee -a LOG.txt
      kernel_times=$(cat apex.0.csv | grep "kernel " | awk -F',' -v cores=${i} -v lib=${lib} -v extension=${extension} -v cpu_platform="${cpu_platform}"  -v OFS=',' '{ print cpu_platform,lib,extension,cores,$1,$2,$3/1000000,$3/$2/1000000 }')
      echo "$kernel_times" | tee -a LOG.txt
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


