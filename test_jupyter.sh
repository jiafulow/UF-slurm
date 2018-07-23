#!/bin/bash
#SBATCH -o logs/output.%j.out
#SBATCH -e logs/output.%j.out

# Root directory
export JFTEST=/home/uf/jlow/jftest2/miniconda3

# CUDA environment variables
#export CUDA_HOME=/opt/cuda9.2
#export CUDNN_HOME=/opt/cuda9.2/cuDNN7.1
#export CUDA_VISIBLE_DEVICES=1
#export PATH=$CUDA_HOME/bin:$PATH
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDNN_HOME/lib64:$LD_LIBRARY_PATH

# Anaconda environment variables
export PATH=$JFTEST/bin:$PATH
export LD_LIBRARY_PATH=$JFTEST/lib:$LD_LIBRARY_PATH

# Virtual env
#export VENV=tensorflow
#export VENV=tensorflow_gpu
export VENV=tensorflow_conda
source activate $VENV

# Fix issue related to missing libraries
export LD_LIBRARY_PATH=$JFTEST/envs/$VENV/lib:$LD_LIBRARY_PATH

# Fix issue related to unavailable GPU devices 
# - https://github.com/NVIDIA/nvidia-docker/issues/262
# - https://github.com/tensorflow/tensorflow/issues/152
#export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export CUDA_VISIBLE_DEVICES='7'

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

# Run jupyter
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute mykeras2_noroot.ipynb --output mykeras2_noroot_evaluated.ipynb
