#!/usr/bin/env bash

#/ Runs performance tests for octotiger over all combinations of 
#/ Performance parameters and machine configurations (cores / gpus)
#/
#/ Usage: octotiger-performance-tests 
#/  --octotiger_version="git commit/tag"
#/  --required modules="space-separated list of modules"
#/  --corecount="space-separated list of core numbers"
#/  --gpucount="space-separated list of core numbers"
#/  --executorcount="space-separated list of gpu executor numbers"
#/  --slicescount="space-separated list of max kernels fused for dynamic work aggregation"
#/  --gpu_backends="space-separated list of gpu backends to test [CUDA|HIP|KOKKOS_CUDA|KOKKOS_HIP|KOKKOS_SYCL]"
#/  --cpu_backends="space-separated list of cpu backends to test [LEGACY|KOKKOS|DEVICE_ONLY]"
#/  --scenario_name="How to call this experiment?"
#/  --help
#/
#/ Example usage:
#/ ./octotiger-performance-tests \
#/  --octotiger_version="v0.10.0" \
#/  --spack_spec="octotiger@0.10.0 +cuda +kokkos cuda_arch=80 griddim=8 %gcc@11 ^hpx@1.9.1 generator=make max_cpu_count=256 networking=none ~disable_async_gpu_futures ^python@3.9.16 ^kokkos@4.0.01 ^cppuddle@0.3.0 number_buffer_buckets=128 max_number_gpus=8 +hpx +allocator_counters" \
#/  --required_modules="gcc/11 cuda/12 hwloc cmake" \
#/  --corecount="1 2 4 8 16 32 64 128 " \
#/  --gpucount="1 2  " \
#/  --executorcount="1 2 4 8 16 32 64 128 " \
#/  --slicescount="1 2 4 8 16 32 64 128 " \
#/  --gpu_backends="CUDA KOKKOS_CUDA" \
#/  --cpu_backends="LEGACY KOKKOS DEVICE_ONLY" \
#/  --scenario_name="BLAST" \
#/  --max_level=3 \
#/  --scenario_parameters=" --config_file=blast.ini --unigrid=1 --disable_output=on --max_level=3 --stop_step=25 --stop_time=25 --print_times_per_timestep=1 --optimize_local_communication=1" \
#/  --hpx_parameters="--hpx:ini=hpx.stacks.use_guard_pages!=0 --hpx:queuing=local-priority-lifo " \
#/  --counter_parameters="--hpx:print-counter=/octotiger*/compute/gpu*kokkos* --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_cuda --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_cuda_aggregated --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_kokkos --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_kokkos_aggregated --hpx:print-counter=/arithmetics/add@/cppuddle*/number_creations/ --hpx:print-counter=/arithmetics/add@/cppuddle*/number_allocations/ --hpx:print-counter=/arithmetics/add@/cppuddle*/number_deallocations/" \

usage() {
    grep '^#/' "$0" | cut -c4-
}

for i in "$@"; do
  case $i in
    --octotiger_version=*)
      octotiger_version="${i#*=}"
      shift # past argument=value
      ;;
    --required_modules=*)
      required_modules="${i#*=}"
      shift # past argument=value
      ;;
    --corecount=*)
      corecount="${i#*=}"
      shift # past argument=value
      ;;
    --gpucount=*)
      gpucount="${i#*=}"
      shift # past argument=value
      ;;
    --executorcount=*)
      executorcount="${i#*=}"
      shift # past argument=value
      ;;
    --slicescount=*)
      slicescount="${i#*=}"
      shift # past argument=value
      ;;
    --gpu_backends=*)
      gpu_backends="${i#*=}"
      shift # past argument=value
      ;;
    --cpu_backends=*)
      cpu_backends="${i#*=}"
      shift # past argument=value
      ;;
    --scenario_name=*)
      scenario_name="${i#*=}"
      shift # past argument=value
      ;;
    --max_level=*)
      max_level="${i#*=}"
      shift # past argument=value
      ;;
    --scenario_parameters=*)
      scenario_parameters="${i#*=}"
      shift # past argument=value
      ;;
    --hpx_parameters=*)
      hpx_parameters="${i#*=}"
      shift # past argument=value
      ;;
    --counter_parameters=*)
      counter_parameters="${i#*=}"
      shift # past argument=value
      ;;
    --spack_spec=*)
      spack_spec="${i#*=}"
      shift # past argument=value
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Error: Unknown option $i"
      usage
      exit 1
      ;;
  esac
done

if [ -z "${octotiger_version}" ]
then
	echo "ERROR: Missing --octotiger_version=\"git commit/tag\""
	echo "       Script requires a valid octotiger version/commit/tag"
	echo "       Example: --octotiger_version=\"v0.10.0\""
	echo ""
	echo "Usage: "
	usage
	exit 1
fi
echo "octotiger_version  = ${octotiger_version}"
if [ -z "${spack_spec}" ]
then
	echo "ERROR: Missing --spack_spec=\"spack spec used for build-env\""
	echo "       Example: --spack_spec=\"octotiger@0.10.0 +cuda +kokkos cuda_arch=80 griddim=8 %gcc@11 ^hpx@1.9.1 generator=make max_cpu_count=256 networking=none ~disable_async_gpu_futures ^python@3.9.16 ^kokkos@4.0.01 ^cppuddle@0.3.0 number_buffer_buckets=128 max_number_gpus=8 +hpx +allocator_counters \""
	echo ""
	echo "Usage: "
	usage
	exit 1
fi
echo "spack_spec     = ${spack_spec}"
if [ -z "${required_modules}" ]
then
	echo "ERROR: Missing --required_modules=\"list of modules\""
	echo "       Script requires a list of modules to be loaded (may be empty)"
	echo "       Example: --required_modules=\"gcc/11 cuda/12 hwloc cmake\""
	echo "       Example: --required_modules=\" \""
	echo ""
	echo "Usage: "
	usage
	exit 1
fi
echo "required_modules   = ${required_modules}"
if [ -z "${corecount}" ]
then
	echo "ERROR: Missing --corecount=\"list of core numbers\""
	echo "       Script requires a list of core numbers to iterate over"
	echo "       Example: --corecount=\"1 2 4 8 16 32 64 128 \""
	echo ""
	echo "Usage: "
	usage
	exit 1
fi
echo "corecount          = ${corecount}"
if [ -z "${gpucount}" ]
then
	echo "ERROR: Missing --gpucount=\"list of gpu numbers\""
	echo "       Script requires a list of gpu numbers to iterate over"
	echo "       Example: --corecount=\"1 2 4 \""
	echo ""
	echo "Usage: "
	usage
	exit 1
fi
echo "gpucount           = ${gpucount}"
if [ -z "${executorcount}" ]
then
	echo "ERROR: Missing --executorcount=\"list of gpu executor numbers\""
	echo "       Script requires a list of gpu executor numbers to iterate over"
	echo "       Example: --executorcount=\"1 2 4 8 16 32 64 128 \""
	echo ""
	echo "Usage: "
	usage
	exit 1
fi
echo "executorcount      = ${executorcount}"
if [ -z "${slicescount}" ]
then
	echo "ERROR: Missing --slicescount=\"list of max kernels fused for dynamic work aggregation\""
	echo "       Script requires a list of max kernels fused to iterate over"
	echo "       Example: --slicescount=\"1 2 4 8 16 32 64 128 \""
	echo ""
	echo "Usage: "
	usage
	exit 1
fi
echo "slicescount        = ${slicescount}"
if [ -z "${gpu_backends}" ]
then
	echo "ERROR: Missing --gpu_backends=\"list of cpu backends to be tesedt [HIP|CUDA|KOKKOS_HIP|KOKKOS_CUDA|KOKKOS_SYCL]\""
	echo "       Define gpu backends to test. Note: Add DEVICE_ONLY for GPU execution."
	echo "       Example: --gpu_backends=\"CUDA KOKKOS_CUDA \""
	echo ""
	echo "Usage: "
	usage
	exit 1
fi
echo "gpu_backends      = ${gpu_backends}"
if [ -z "${cpu_backends}" ]
then
	echo "ERROR: Missing --cpu_backends=\"list of cpu backends to be tested [LEGACY|KOKKOS|DEVICE_ONLY]\""
	echo "       Define cpu backends to test. Note: Add DEVICE_ONLY for GPU execution."
	echo "       Example: --cpu_backends=\"LEGACY KOKKOS DEVICE_ONLY \""
	echo ""
	echo "Usage: "
	usage
	exit 1
fi
echo "cpu_backends      = ${cpu_backends}"
if [ -z "${scenario_name}" ]
then
	echo "ERROR: Missing --scenario_name=\"Name of this experiment\""
	echo "       Example: --scenario_name=\"BLAST \""
	echo ""
	echo "Usage: "
	usage
	exit 1
fi
echo "scenario_name     = ${scenario_name}"
if [ -z "${max_level}" ]
then
	echo "ERROR: Missing --max_level=\"Max octtree level of this scenario\""
	echo "       Example: --max_level=3"
	echo ""
	echo "Usage: "
	usage
	exit 1
fi
echo "max_level        = ${max_level}"
if [ -z "${scenario_parameters}" ]
then
	echo "ERROR: Missing --scenario_parameters=\"list of octotiger parameters\""
	echo "       List of required octotiger scenario parameters"
	echo "       Example: --scenario_parameters=\"--config_file=blast.ini --unigrid=1 --disable_output=on --max_level=4 --stop_step=25 --stop_time=25 --print_times_per_timestep=1 --optimize_local_communication=1 \""
	echo ""
	echo "Usage: "
	usage
	exit 1
fi
echo "scenario_parameters = ${scenario_parameters}"
if [ -z "${hpx_parameters}" ]
then
	echo "ERROR: Missing --hpx_parameters=\"list of fixed HPX parameters\""
	echo "       Example: --hpx_parameters=\"--hpx:ini=hpx.stacks.use_guard_pages!=0 --hpx:queuing=local-priority-lifo  \""
	echo ""
	echo "Usage: "
	usage
	exit 1
fi
echo "hpx_parameters     = ${hpx_parameters}"

if [ -z "${counter_parameters}" ]
then
	echo "ERROR: Missing --counter_parameters=\"list of HPX counter parameters\""
	echo "       Example: --counter_parameters=\"--hpx:print-counter=/octotiger*/compute/gpu*kokkos* --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_cuda --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_cuda_aggregated --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_kokkos --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_kokkos_aggregated --hpx:print-counter=/arithmetics/add@/cppuddle*/number_creations/ --hpx:print-counter=/arithmetics/add@/cppuddle*/number_allocations/ --hpx:print-counter=/arithmetics/add@/cppuddle*/number_deallocations/  \""
	echo ""
	echo "Usage: "
	usage
	exit 1
fi
echo "counter_parameters     = ${counter_parameters}"


set -e

# Defaults I used...
#octotiger_version="v0.10.0"
#required_modules="gcc/11 cuda/12 hwloc cmake"
#corecount="1 2 4 8 16 32 64 128 "
#gpucount="1 2  "
#executorcount="1 2 4 8 16 32 64 128 "
#slicescount="1 2 4 8 16 32 64 128 "
#gpu_backends="CUDA KOKKOS_CUDA"
#cpu_backends="LEGACY KOKKOS DEVICE_ONLY"
#scenario_name="BLAST"
#max_level=3
#scenario_parameters=" --config_file=blast.ini --unigrid=1 --disable_output=on --max_level=${max_level} --stop_step=25 --stop_time=25 --print_times_per_timestep=1 --optimize_local_communication=1"
#hpx_parameters="--hpx:ini=hpx.stacks.use_guard_pages!=0 --hpx:queuing=local-priority-lifo "
#counter_parameters="--hpx:print-counter=/octotiger*/compute/gpu*kokkos* --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_cuda --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_cuda_aggregated --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_kokkos --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_kokkos_aggregated --hpx:print-counter=/arithmetics/add@/cppuddle*/number_creations/ --hpx:print-counter=/arithmetics/add@/cppuddle*/number_allocations/ --hpx:print-counter=/arithmetics/add@/cppuddle*/number_deallocations/"
#spack_spec="octotiger@0.10.0 +cuda +kokkos cuda_arch=80 griddim=8 %gcc@11 ^hpx@1.9.1 generator=make max_cpu_count=256 networking=none ~disable_async_gpu_futures ^python@3.9.16 ^kokkos@4.0.01 ^cppuddle@0.3.0 number_buffer_buckets=128 max_number_gpus=8 +hpx +allocator_counters"

scriptname=$(basename "$0")

# Timestamp
today=`date +%Y-%m-%d_%H_%M`
#git clone --recurse-submodules https://github.com/STEllAR-GROUP/octotiger experiment_pure_hydro_benchmark_run_${today}
#cd experiment_pure_hydro_benchmark_run_${today}
#git checkout develop

mkdir octotiger_benchmark_${scenario_name}-${max_level}_${today}
cd octotiger_benchmark_${scenario_name}-${max_level}_${today}
echo "Created experiment directory with timestamp ${today}..." | tee -a octotiger_benchmark.log
echo "Running ${scriptname}..." | tee octotiger_benchmark.log
echo "Loading required modules: ${required_modules}" | tee -a octotiger_benchmark.log
module load ${required_modules}
echo "Print the basic information.."| tee -a octotiger_benchmark.log

# Output to log
echo "# Date of run $today" | tee -a octotiger_benchmark.log
echo "# Modules: ${required_modules}" | tee -a octotiger_benchmark.log
echo "# Spack Root: $(printenv | grep SPACK_ROOT)" | tee -a octotiger_benchmark.log
echo "# Spack version: $(spack --version)" | tee -a octotiger_benchmark.log
cd $(spack repo list | grep octotiger-spack | awk -F ' ' '{print $2}')
echo "# Spack Octotiger repo version: $(git log --oneline --no-abbrev-commit | head -n 1)" | tee -a octotiger_benchmark.log
cd -
echo "# Spack octo spec: ${spack_spec}"| tee -a octotiger_benchmark.log
echo "# Octo src version: ${octotiger_version}" | tee -a octotiger_benchmark.log
echo "Concretizing spec.."| tee -a octotiger_benchmark.log
echo "# Concrete spec: $(spack spec ${spack_spec})"  | tee -a octotiger_benchmark.log

# Output to CSV
echo "# Octo-Tiger Node-Level Benchmark run" | tee octotiger_benchmark.csv
echo "# ===================================" | tee -a octotiger_benchmark.csv 
echo "# Date of run $today" | tee -a octotiger_benchmark.csv
echo "# Scenario name ${scenario_name} at max level ${max_level}" | tee
echo "# Modules: ${required_modules}" | tee -a octotiger_benchmark.csv
echo "# Spack Root: $(printenv | grep SPACK_ROOT)" | tee -a octotiger_benchmark.csv
echo "# Spack version: $(spack --version)" | tee -a octotiger_benchmark.csv
cd $(spack repo list | grep octotiger-spack | awk -F ' ' '{print $2}')
echo "# Spack Octotiger repo version: $(git log --oneline --no-abbrev-commit | head -n 1)" | tee -a octotiger_benchmark.csv
cd -
echo "# Spack octo spec: ${spack_spec}" | tee -a octotiger_benchmark.csv
echo "# Octo src version: ${octotiger_version}" | tee -a octotiger_benchmark.csv
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
echo "Loading spec..."| tee -a octotiger_benchmark.log
git clone --recurse-submodules https://github.com/STEllAR-GROUP/octotiger.git octotiger-src | tee -a octotiger_benchmark.log
cd octotiger-src
git checkout --recurse-submodules ${octotiger_version}
spack dev-build --fresh --until build ${spack_spec} | tee -a octotiger_benchmark.log
cd spack-build
echo 'build-env...'| tee -a octotiger_benchmark.log
#spack build-env --dump build-env.sh ${spack_spec} --
#echo 'apply tmux / spack bugfix for build-env.sh...'| tee -a octotiger_benchmark.log
#sed -i '/tmux_version/d' build-env.sh
#echo 'sourcing...'| tee -a octotiger_benchmark.log
#source build-env.sh
#echo 'building...'|  tee -a octotiger_benchmark.log
#make -j32 VERBOSE=1| tee -a octotiger_benchmark.log
echo "Testing binary..."| tee -a octotiger_benchmark.log
./octotiger --help
#ctest --output-on-failure | tee -a octotiger_benchmark.log
cd ../..


echo "Move scenario files to experiment directory..."| tee -a octotiger_benchmark.log
cp ../${scriptname} .
cp ../*.ini .

set +e


echo "scenario name, max level, cpu kernel type, gpu kernel type, future_type, hpx threads, number gpus, slices, executors, nodes, processcount, total time, computation time, number creation, number allocations, number deallocations, number hydro kokkos launches, number aggregated hydro kokkos launches, number hydro cuda launches, number aggregated cuda kokkos launches, min time-per-timestep, max time-per-timestep," \
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
            octotiger-src/spack-build/octotiger ${scenario_parameters} ${hpx_parameters} ${performance_parameters} ${counter_parameters} | tee current_run.out
            # append output to log
            cat current_run.out >> octotiger_benchmark.log
            # analyse output for relevant counters and runtime
            echo "${scenario_name}, ${max_level}, ${cpu_kernel_type}, ${gpu_kernel_type}, ASYNC, ${cores}, ${gpus}, ${slices}, ${executors}, 1, 1, " \
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
    octotiger-src/spack-build/octotiger ${scenario_parameters} ${hpx_parameters} ${performance_parameters} ${counter_parameters} | tee current_run.out
    # append output to log
    cat current_run.out >> octotiger_benchmark.log
    # analyse output for relevant counters and runtime
    echo "${scenario_name}, ${max_level}, ${cpu_kernel_type}, ${gpu_kernel_type}, ASYNC, ${cores}, ${gpus}, ${slices}, ${executors}, 1, 1, " \
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

echo "Extracting done! Zipping and printing results..." | tee -a octotiger_benchmark.log
cat octotiger_benchmark.csv | tee -a octotiger_benchmark.log

tar -cvf octotiger_benchmark_${scenario_name}-${max_level}_${today}.tar octotiger_benchmark.log octotiger_benchmark.csv \
       	 blast.ini ${scriptname}
cd ..
cp -r octotiger_benchmark_${scenario_name}-${max_level}_${today} octotiger_benchmark_${scenario_name}-${max_level}_last_run
