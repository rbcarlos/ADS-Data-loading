#!/bin/bash

#SBATCH --job-name=ads    # Job name
#SBATCH --output=job.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --workdir=/home/roba/ADS-Data-loading/
#SBATCH --cpus-per-task=16        # Schedule 8 cores (includes hyperthreading)
#SBATCH --mem=0
#SBATCH --gres=gpu               # Schedule a GPU, or more with gpu:2 etc
#SBATCH --time=02:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --partition=brown,red    # Run on either the Red or Brown queue


echo "Running on $(hostname):"
nvidia-smi

free -m

module load Anaconda3/2021.05
#module load CUDAcore/11.0.2
#conda env create -f environment.yaml
source /home/roba/.bashrc
conda activate nn_env

python3 5.py --logfile "out/5py.csv"
python3 5.py --tracefile "out/trace5py.json" -p
python3 6.py --logfile "out/6py.csv"
python3 6.py --tracefile "out/trace6py.json" -p
python3 7.py --logfile "out/7py.csv"
python3 7.py --tracefile "out/trace7py.json" -p
python3 8.py --logfile "out/8py.csv"
python3 8.py --tracefile "out/trace8py.json" -p
python3 9.py -j 1 --logfile "out/9py_1worker.csv"
python3 9.py -j 1 --tracefile "out/trace9py_1worker.json" -p
python3 9.py -j 2 --logfile "out/9py_2worker.csv"
python3 9.py -j 2 --tracefile "out/trace9py_2worker.json" -p
python3 9.py -j 3 --logfile "out/9py_3worker.csv"
python3 9.py -j 3 --tracefile "out/trace9py_3worker.json" -p
python3 9.py -j 4 --logfile "out/9py_4worker.csv"
python3 9.py -j 4 --tracefile "out/trace9py_4worker.json" -p