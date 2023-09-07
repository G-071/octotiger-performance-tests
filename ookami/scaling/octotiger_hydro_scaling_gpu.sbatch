#!/usr/bin/env bash
#SBATCH --ntasks-per-node=4
#SBATCH -c 12
#
echo "======================="
echo "Beginning SBATCH"
echo ""

export SLURM_CPU_BIND="cores"

#module load PrgEnv-cray
##export MPICH_OFI_NUM_NICS="1:1"
export MPICH_OFI_NIC_VERBOSE=1
export LCI_SERVER_MAX_SENDS=512
export LCI_SERVER_MAX_RECVS=16384
export LCI_SERVER_NUM_PKTS=65536
export LCI_SERVER_MAX_CQES=65536
export LCI_PACKET_SIZE=12288
export LCI_USE_DREG=0
export APEX_DISABLE=1
export APEX_SCREEN_OUTPUT=1
export APEX_ENABLE_CUDA=1
##export LCM_LOG_LEVEL="info"
##
#
echo ""
echo "SLURM Configuration"
echo "----------------------"
echo "-> SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "-> Number nodes: ${SLURM_NNODES}"
echo "-> Number nodes: ${SLURM_NNODES}"
echo "-> Tasks per node: ${SLURM_NTASKS_PER_NODE}"
echo "-> CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "-> GPUs per task: ${SLURM_GPUS_PER_TASK}"
echo "-> Local node id: ${SLURM_PROCID}"
echo "-> Local node id: ${SLURM_LOCALID}"
echo "-> External Time limit?: ${SBATCH_TIMELIMIT}"
#
echo ""
echo "LCI / FI_CXI Config"
echo "----------------------"
echo "$(printenv | grep LCI)"
echo "$(printenv | grep FI_CXI)"

echo ""
echo "Module Configuration"
echo "----------------------"
module list
#
echo ""
echo "Octotiger binary"
echo "----------------------"
echo "$(which octotiger)"

echo ""
echo "Custom Args"
echo "----------------------"
if [ -z "${OCTOTIGER_MAX_LEVEL}" ]
then
	echo "ERROR: OCTOTIGER_MAX_LEVEL not set"
	echo "Lauch script aborting before srun..."
	exit 1
else
	echo "OCTOTIGER_MAX_LEVEL=${OCTOTIGER_MAX_LEVEL}"
	re='^[0-9]+$'
	if ! [[ ${OCTOTIGER_MAX_LEVEL} =~ $re ]] ; then
		   echo "ERROR: OCTOTIGER_MAX_LEVEL is not a number"; exit 1
	fi
fi
if [ -z "${OCTOTIGER_EXECUTORS_PER_GPU}" ]
then
	echo "ERROR: OCTOTIGER_EXECUTORS_PER_GPU not set"
	echo "Lauch script aborting before srun..."
	exit 1
else
	echo "OCTOTIGER_EXECUTORS_PER_GPU=${OCTOTIGER_EXECUTORS_PER_GPU}"
	re='^[0-9]+$'
	if ! [[ ${OCTOTIGER_EXECUTORS_PER_GPU} =~ $re ]] ; then
		   echo "ERROR: OCTOTIGER_EXECUTORS_PER_GPU is not a number"; exit 1
	fi
fi
if [ -z "${OCTOTIGER_MAX_KERNELS_FUSED}" ]
then
	echo "ERROR: OCTOTIGER_MAX_KERNELS_FUSED not set"
	echo "Lauch script aborting before srun..."
	exit 1
else
	echo "OCTOTIGER_EXECUTORS_PER_GPU=${OCTOTIGER_MAX_KERNELS_FUSED}"
	re='^[0-9]+$'
	if ! [[ ${OCTOTIGER_MAX_KERNELS_FUSED} =~ $re ]] ; then
		   echo "ERROR: OCTOTIGER_MAX_KERNELS_FUSED is not a number"; exit 1
	fi
fi


echo ""
echo "======================="
echo "Beginning SBATCH"
echo ""

#srun octotiger --multipole_device_kernel_type=OFF --multipole_host_kernel_type=KOKKOS --monopole_device_kernel_type=OFF --monopole_host_kernel_type=KOKKOS --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY --amr_boundary_kernel_type=AMR_OPTIMIZED --max_kernels_fused=${OCTOTIGER_MAX_KERNELS_FUSED} --executors_per_gpu=${OCTOTIGER_EXECUTORS_PER_GPU} --number_gpus=1 --config_file=blast.ini --unigrid=1 --disable_output=on --max_level=${OCTOTIGER_MAX_LEVEL} --stop_step=25 --stop_time=25 --print_times_per_timestep=1 --optimize_local_communication=1 --hpx:ini=hpx.stacks.use_guard_pages!=0 --hpx:ini=hpx.parcel.mpi.priority=1000 --hpx:ini=hpx.parcel.bootstrap=mpi --hpx:ini=hpx.parcel.mpi.enable=1 --hpx:ini=hpx.parcel.mpi.zero_copy_optimization=0 --hpx:queuing=local-priority-lifo --hpx:print-counter=/octotiger*/compute/gpu*kokkos* --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_kokkos --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_kokkos_aggregated --hpx:print-counter=/arithmetics/add@/cppuddle*/number_creations/ --hpx:print-counter=/arithmetics/add@/cppuddle*/number_allocations/ --hpx:print-counter=/arithmetics/add@/cppuddle*/number_deallocations/
srun octotiger --multipole_device_kernel_type=OFF --multipole_host_kernel_type=KOKKOS --monopole_device_kernel_type=OFF --monopole_host_kernel_type=KOKKOS --hydro_device_kernel_type=OFF --hydro_host_kernel_type=KOKKOS --amr_boundary_kernel_type=AMR_OPTIMIZED --max_kernels_fused=${OCTOTIGER_MAX_KERNELS_FUSED} --executors_per_gpu=${OCTOTIGER_EXECUTORS_PER_GPU} --number_gpus=1 --config_file=rotating_star.ini --unigrid=1 --disable_output=on --max_level=${OCTOTIGER_MAX_LEVEL} --stop_step=25 --stop_time=25 --print_times_per_timestep=1 --optimize_local_communication=1 --hpx:ini=hpx.stacks.use_guard_pages!=0 --hpx:ini=hpx.parcel.mpi.priority=1000 --hpx:ini=hpx.parcel.bootstrap=mpi --hpx:ini=hpx.parcel.mpi.enable=1 --hpx:ini=hpx.parcel.mpi.zero_copy_optimization=0 --hpx:queuing=local-priority-lifo --hpx:print-counter=/octotiger*/compute/gpu*kokkos* --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_kokkos --hpx:print-counter=/arithmetics/add@/octotiger*/compute/gpu/hydro_kokkos_aggregated --hpx:print-counter=/arithmetics/add@/cppuddle*/number_creations/ --hpx:print-counter=/arithmetics/add@/cppuddle*/number_allocations/ --hpx:print-counter=/arithmetics/add@/cppuddle*/number_deallocations/

echo "Ending SBATCH"
echo "======================="
echo ""
echo ""
