#!/bin/bash
#SBATCH -o logs/output.%j.out
#SBATCH -e logs/output.%j.out

# Set environment variables
source startup.sh

# Activate conda env
export VENV=tensorflow_conda
source activate $VENV

# Fix issue related to missing libraries
export LD_LIBRARY_PATH=$JFTEST/envs/$VENV/lib:$LD_LIBRARY_PATH

# Fix issue related to unavailable GPU devices 
# - https://github.com/NVIDIA/nvidia-docker/issues/262
# - https://github.com/tensorflow/tensorflow/issues/152
#export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export CUDA_VISIBLE_DEVICES='1'

# Some checks
echo 'Checking PATH ...'
echo $PATH
echo 'Checking LD_LIBRARY_PATH ...'
echo $LD_LIBRARY_PATH
echo 'Checking PYTHONPATH ...'
echo $PYTHONPATH
echo 'Checking visible devices ...'
echo $CUDA_VISIBLE_DEVICES
echo 'Checking conda ...'
which conda
echo 'Checking python ...'
which python
echo 'Checking nvidia-smi ...'
nvidia-smi

# System info
#sudo /mops/linux/sysinfo/sysinfo
#env | grep SLURM

# Run python
python test.py

# Run skopt
export CMSSW_VERSION=
python nn_gridsearch.py
