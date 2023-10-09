#!/bin/bash -l
#SBATCH --job-name="MI100 Octo-Tiger Performance test"
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=mi100
#SBATCH --nodelist=kamand0
#SBATCH -e kamand_multigpu_callback.out
#SBATCH -o kamand_multigpu_callback.out

./test_multigpu_performance.sh \
	--octotiger_version="v0.10.0" \
       	--required_modules=" rocm/5.4.6 gcc/11 hwloc cmake" \
       	--corecount="1 2 4 8 16 32 64 128 " \
       	--gpucount="1 2 " \
       	--executorcount="1 2 4 8 16 32 64 128 " \
       	--slicescount="1 2 4 8 16 32 64 128 " \
       	--gpu_backends="HIP KOKKOS_HIP " \
       	--cpu_backends="LEGACY KOKKOS DEVICE_ONLY" \
       	--scenario_name="KAMAND_BLAST_CALLBACK_" \
       	--max_level=3 \
       	--scenario_parameters="--config_file=blast.ini --unigrid=1 --disable_output=on --max_level=3 --stop_step=25 --stop_time=25 --print_times_per_timestep=1 --optimize_local_communication=1" \
       	--hpx_parameters="--hpx:ini=hpx.stacks.use_guard_pages!=0 --hpx:queuing=local-priority-lifo  " \
       	--counter_parameters="--hpx:print-counter=/octotiger*/compute/gpu*kokkos* --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_cuda --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_cuda_aggregated --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_kokkos --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_kokkos_aggregated --hpx:print-counter=/arithmetics/add@/cppuddle*/number_creations/ --hpx:print-counter=/arithmetics/add@/cppuddle*/number_allocations/ --hpx:print-counter=/arithmetics/add@/cppuddle*/number_deallocations/  " \
	 --spack_spec="octotiger@0.10.0 +rocm+kokkos amdgpu_target=gfx908%rocmcc@5.4.6 ^asio@1.16.0^hpx max_cpu_count=128 amdgpu_target=gfx908%rocmcc@5.4.6 ^kokkos@4.0.01 amdgpu_target=gfx908%rocmcc@5.4.6 ^hpx-kokkos%rocmcc@5.4.6 future_type=callback ^silo~mpi%gcc@11 ^cppuddle@0.3.0 number_buffer_buckets=128 max_number_gpus=2 +hpx +allocator_counters%rocmcc@5.4.6"

