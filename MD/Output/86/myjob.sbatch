#!/bin/bash

#SBATCH --account=haohuali
#SBATCH --partition=hpib
#SBATCH --nodes=4
#SBATCH --ntasks=32


module load lammps/202003/lmp_intelcpu_intelmpi

srun -n $SLURM_NTASKS lmp_intelcpu_intelmpi -in in.ind












