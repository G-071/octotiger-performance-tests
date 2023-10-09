#!/bin/bash -l
#SBATCH --job-name="Nasrin Octo-Tiger Performance test"
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --partition=cuda-A100
#SBATCH --nodelist=nasrin1
#SBATCH -e nasrin_blast_test.out
#SBATCH -o nasrin_blast_test.out

time ./test_multigpu_performance2.sh \
	--octotiger_version="v0.10.0" \
       	--required_modules="gcc/11 cuda/12 hwloc cmake" \
       	--corecount="1 2 4 8 16 " \
       	--gpucount="1 " \
       	--executorcount=" 16 " \
       	--slicescount="8 " \
       	--gpu_backends="KOKKOS_CUDA " \
       	--cpu_backends="LEGACY KOKKOS DEVICE_ONLY" \
	--scenario_name="NASRIN_BLAST_SCALING_TEST" \
       	--max_level=4 \
       	--scenario_parameters="--config_file=blast.ini --unigrid=1 --disable_output=on --max_level=4 --stop_step=25 --stop_time=25 --print_times_per_timestep=1 --optimize_local_communication=1" \
       	--hpx_parameters="--hpx:ini=hpx.stacks.use_guard_pages!=0 --hpx:queuing=local-priority-lifo  " \
       	--counter_parameters="--hpx:print-counter=/octotiger*/compute/gpu*kokkos* --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_cuda --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_cuda_aggregated --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_kokkos --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_kokkos_aggregated --hpx:print-counter=/arithmetics/add@/cppuddle*/number_creations/ --hpx:print-counter=/arithmetics/add@/cppuddle*/number_allocations/ --hpx:print-counter=/arithmetics/add@/cppuddle*/number_deallocations/  " \
	 --spack_spec="octotiger@0.10.0 +cuda +kokkos cuda_arch=80 griddim=8 %gcc@11 ^hpx@1.9.1 generator=make max_cpu_count=256 networking=none ~disable_async_gpu_futures ^python@3.9.16 ^kokkos@4.0.01 ^cppuddle@0.3.0 number_buffer_buckets=128 max_number_gpus=8 +hpx +allocator_counters"

