#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="exp 4 symptom"
#SBATCH --output=out/exp4-symptom-%j.out

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate lungsounds
cd ../models

for i in 1 2 3 4 5
do
  cp -a ../heart/logs/2_1/spec-pre-large/. ../data/logs/2_1/spec-pre-crackle-heart-large

  python contrastive.py --mode train --task crackle --log_dir 2_1/spec-pre-crackle-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task crackle --log_dir 2_1/spec-pre-crackle-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task crackle --log_dir 2_1/spec-pre-crackle-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task crackle --log_dir 2_1/spec-pre-crackle-heart-large --data ../data --evaluator linear

  cp -a ../data/logs/2_1/spec-pre-large/. ../data/logs/2_1/spec-pre-crackle-lung-large

  python contrastive.py --mode train --task crackle --log_dir 2_1/spec-pre-crackle-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task crackle --log_dir 2_1/spec-pre-crackle-lung-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task crackle --log_dir 2_1/spec-pre-crackle-lung-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task crackle --log_dir 2_1/spec-pre-crackle-lung-large --data ../data --evaluator linear

  cp -a ../heart/logs/2_1/spec-pre-large/. ../data/logs/2_1/spec-pre-wheeze-heart-large

  python contrastive.py --mode train --task wheeze --log_dir 2_1/spec-pre-wheeze-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task wheeze --log_dir 2_1/spec-pre-wheeze-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task wheeze --log_dir 2_1/spec-pre-wheeze-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task wheeze --log_dir 2_1/spec-pre-wheeze-heart-large --data ../data --evaluator linear

  cp -a ../data/logs/2_1/spec-pre-large/. ../data/logs/2_1/spec-pre-wheeze-lung-large

  python contrastive.py --mode train --task wheeze --log_dir 2_1/spec-pre-wheeze-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task wheeze --log_dir 2_1/spec-pre-wheeze-lung-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task wheeze --log_dir 2_1/spec-pre-wheeze-lung-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task wheeze --log_dir 2_1/spec-pre-wheeze-lung-large --data ../data --evaluator linear


  cp -a ../heart/logs/2_1/time-pre-large/. ../data/logs/2_1/time-pre-crackle-heart-large

  python contrastive.py --mode train --task crackle --log_dir 2_1/time-pre-crackle-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task crackle --log_dir 2_1/time-pre-crackle-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task crackle --log_dir 2_1/time-pre-crackle-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task crackle --log_dir 2_1/time-pre-crackle-heart-large --data ../data --evaluator linear

  cp -a ../data/logs/2_1/time-pre-large/. ../data/logs/2_1/time-pre-crackle-lung-large

  python contrastive.py --mode train --task crackle --log_dir 2_1/time-pre-crackle-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task crackle --log_dir 2_1/time-pre-crackle-lung-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task crackle --log_dir 2_1/time-pre-crackle-lung-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task crackle --log_dir 2_1/time-pre-crackle-lung-large --data ../data --evaluator linear

  cp -a ../heart/logs/2_1/time-pre-large/. ../data/logs/2_1/time-pre-wheeze-heart-large

  python contrastive.py --mode train --task wheeze --log_dir 2_1/time-pre-wheeze-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task wheeze --log_dir 2_1/time-pre-wheeze-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task wheeze --log_dir 2_1/time-pre-wheeze-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task wheeze --log_dir 2_1/time-pre-wheeze-heart-large --data ../data --evaluator linear

  cp -a ../data/logs/2_1/time-pre-large/. ../data/logs/2_1/time-pre-wheeze-lung-large

  python contrastive.py --mode train --task wheeze --log_dir 2_1/time-pre-wheeze-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task wheeze --log_dir 2_1/time-pre-wheeze-lung-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task wheeze --log_dir 2_1/time-pre-wheeze-lung-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task wheeze --log_dir 2_1/time-pre-wheeze-lung-large --data ../data --evaluator linear



  cp -a ../heart/logs/2_1/freq-pre-large/. ../data/logs/2_1/freq-pre-crackle-heart-large

  python contrastive.py --mode train --task crackle --log_dir 2_1/freq-pre-crackle-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task crackle --log_dir 2_1/freq-pre-crackle-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task crackle --log_dir 2_1/freq-pre-crackle-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task crackle --log_dir 2_1/freq-pre-crackle-heart-large --data ../data --evaluator linear

  cp -a ../data/logs/2_1/freq-pre-large/. ../data/logs/2_1/freq-pre-crackle-lung-large

  python contrastive.py --mode train --task crackle --log_dir 2_1/freq-pre-crackle-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task crackle --log_dir 2_1/freq-pre-crackle-lung-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task crackle --log_dir 2_1/freq-pre-crackle-lung-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task crackle --log_dir 2_1/freq-pre-crackle-lung-large --data ../data --evaluator linear

  cp -a ../heart/logs/2_1/freq-pre-large/. ../data/logs/2_1/freq-pre-wheeze-heart-large

  python contrastive.py --mode train --task wheeze --log_dir 2_1/freq-pre-wheeze-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task wheeze --log_dir 2_1/freq-pre-wheeze-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task wheeze --log_dir 2_1/freq-pre-wheeze-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task wheeze --log_dir 2_1/freq-pre-wheeze-heart-large --data ../data --evaluator linear

  cp -a ../data/logs/2_1/freq-pre-large/. ../data/logs/2_1/freq-pre-wheeze-lung-large

  python contrastive.py --mode train --task wheeze --log_dir 2_1/freq-pre-wheeze-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task wheeze --log_dir 2_1/freq-pre-wheeze-lung-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task wheeze --log_dir 2_1/freq-pre-wheeze-lung-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task wheeze --log_dir 2_1/freq-pre-wheeze-lung-large --data ../data --evaluator linear


  cp -a ../heart/logs/2_1/split-pre-large/. ../data/logs/2_1/split-pre-crackle-heart-large

  python contrastive.py --mode train --task crackle --log_dir 2_1/split-pre-crackle-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task crackle --log_dir 2_1/split-pre-crackle-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task crackle --log_dir 2_1/split-pre-crackle-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task crackle --log_dir 2_1/split-pre-crackle-heart-large --data ../data --evaluator linear

  cp -a ../data/logs/2_1/split-pre-large/. ../data/logs/2_1/split-pre-crackle-lung-large

  python contrastive.py --mode train --task crackle --log_dir 2_1/split-pre-crackle-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task crackle --log_dir 2_1/split-pre-crackle-lung-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task crackle --log_dir 2_1/split-pre-crackle-lung-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task crackle --log_dir 2_1/split-pre-crackle-lung-large --data ../data --evaluator linear

  cp -a ../heart/logs/2_1/split-pre-large/. ../data/logs/2_1/split-pre-wheeze-heart-large

  python contrastive.py --mode train --task wheeze --log_dir 2_1/split-pre-wheeze-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task wheeze --log_dir 2_1/split-pre-wheeze-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task wheeze --log_dir 2_1/split-pre-wheeze-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task wheeze --log_dir 2_1/split-pre-wheeze-heart-large --data ../data --evaluator linear

  cp -a ../data/logs/2_1/split-pre-large/. ../data/logs/2_1/split-pre-wheeze-lung-large

  python contrastive.py --mode train --task wheeze --log_dir 2_1/split-pre-wheeze-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task wheeze --log_dir 2_1/split-pre-wheeze-lung-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task wheeze --log_dir 2_1/split-pre-wheeze-lung-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task wheeze --log_dir 2_1/split-pre-wheeze-lung-large --data ../data --evaluator linear


  cp -a ../heart/logs/2_1/spec-split-pre-large/. ../data/logs/2_1/spec-split-pre-crackle-heart-large

  python contrastive.py --mode train --task crackle --log_dir 2_1/spec-split-pre-crackle-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task crackle --log_dir 2_1/spec-split-pre-crackle-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task crackle --log_dir 2_1/spec-split-pre-crackle-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task crackle --log_dir 2_1/spec-split-pre-crackle-heart-large --data ../data --evaluator linear

  cp -a ../data/logs/2_1/spec-split-pre-large/. ../data/logs/2_1/spec-split-pre-crackle-lung-large

  python contrastive.py --mode train --task crackle --log_dir 2_1/spec-split-pre-crackle-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task crackle --log_dir 2_1/spec-split-pre-crackle-lung-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task crackle --log_dir 2_1/spec-split-pre-crackle-lung-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task crackle --log_dir 2_1/spec-split-pre-crackle-lung-large --data ../data --evaluator linear

  cp -a ../heart/logs/2_1/spec-split-pre-large/. ../data/logs/2_1/spec-split-pre-wheeze-heart-large

  python contrastive.py --mode train --task wheeze --log_dir 2_1/spec-split-pre-wheeze-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task wheeze --log_dir 2_1/spec-split-pre-wheeze-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task wheeze --log_dir 2_1/spec-split-pre-wheeze-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task wheeze --log_dir 2_1/spec-split-pre-wheeze-heart-large --data ../data --evaluator linear

  cp -a ../data/logs/2_1/spec-split-pre-large/. ../data/logs/2_1/spec-split-pre-wheeze-lung-large

  python contrastive.py --mode train --task wheeze --log_dir 2_1/spec-split-pre-wheeze-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task wheeze --log_dir 2_1/spec-split-pre-wheeze-lung-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task wheeze --log_dir 2_1/spec-split-pre-wheeze-lung-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task wheeze --log_dir 2_1/spec-split-pre-wheeze-lung-large --data ../data --evaluator linear

  python contrastive.py --mode train --task crackle --log_dir 2_1/supervised-crackle-large --data ../data --train_prop 1 --epochs 25 --evaluator fine-tune
  python contrastive.py --mode test --task crackle --log_dir 2_1/supervised-crackle-large --data ../data --evaluator fine-tune
  #	python contrastive.py --mode train --task crackle --log_dir 2_1/supervised-crackle-raw-medium --data ../data --augment raw --train_prop 1 --epochs 25 --evaluator fine-tune
  #	python contrastive.py --mode test --task crackle --log_dir 2_1/supervised-crackle-raw-large --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task crackle --log_dir 2_1/supervised-crackle-spec-large --data ../data --augment spec --train_prop 1 --epochs 25 --evaluator fine-tune
#  python contrastive.py --mode test --task crackle --log_dir 2_1/supervised-crackle-spec-large --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task crackle --log_dir 2_1/supervised-crackle-time-large --data ../data --augment time --train_prop 1 --epochs 25 --evaluator fine-tune
#  python contrastive.py --mode test --task crackle --log_dir 2_1/supervised-crackle-time-large --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task crackle --log_dir 2_1/supervised-crackle-freq-large --data ../data --augment freq --train_prop 1 --epochs 25 --evaluator fine-tune
#  python contrastive.py --mode test --task crackle --log_dir 2_1/supervised-crackle-freq-large --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task crackle --log_dir 2_1/supervised-crackle-split-large --data ../data --augment split --train_prop 1 --epochs 25 --evaluator fine-tune
#  python contrastive.py --mode test --task crackle --log_dir 2_1/supervised-crackle-split-large --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task crackle --log_dir 2_1/supervised-crackle-spec-split-large --data ../data --augment spec+split --train_prop 1 --epochs 25 --evaluator fine-tune
#  python contrastive.py --mode test --task crackle --log_dir 2_1/supervised-crackle-spec-split-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task crackle --log_dir 2_1/supervised-crackle-large-full --data ../data --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
  python contrastive.py --mode test --task crackle --log_dir 2_1/supervised-crackle-large-full --data ../data --evaluator fine-tune
  #	python contrastive.py --mode train --task crackle --log_dir 2_1/supervised-crackle-raw-large-full --data ../data --augment raw --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
  #	python contrastive.py --mode test --task crackle --log_dir 2_1/supervised-crackle-raw-large-full --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task crackle --log_dir 2_1/supervised-crackle-spec-large-full --data ../data --augment spec --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#  python contrastive.py --mode test --task crackle --log_dir 2_1/supervised-crackle-spec-large-full --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task crackle --log_dir 2_1/supervised-crackle-time-large-full --data ../data --augment time --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#  python contrastive.py --mode test --task crackle --log_dir 2_1/supervised-crackle-time-large-full --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task crackle --log_dir 2_1/supervised-crackle-freq-large-full --data ../data --augment freq --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#  python contrastive.py --mode test --task crackle --log_dir 2_1/supervised-crackle-freq-large-full --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task crackle --log_dir 2_1/supervised-crackle-split-large-full --data ../data --augment split --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#  python contrastive.py --mode test --task crackle --log_dir 2_1/supervised-crackle-split-large-full --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task crackle --log_dir 2_1/supervised-crackle-spec-split-large-full --data ../data --augment spec+split --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#  python contrastive.py --mode test --task crackle --log_dir 2_1/supervised-crackle-spec-split-large-full --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task wheeze --log_dir 2_1/supervised-wheeze-large --data ../data --train_prop 1 --epochs 25 --evaluator fine-tune
  python contrastive.py --mode test --task wheeze --log_dir 2_1/supervised-wheeze-large --data ../data --evaluator fine-tune
  #	python contrastive.py --mode train --task wheeze --log_dir 2_1/supervised-wheeze-raw-medium --data ../data --augment raw --train_prop 1 --epochs 25 --evaluator fine-tune
  #	python contrastive.py --mode test --task wheeze --log_dir 2_1/supervised-wheeze-raw-large --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task wheeze --log_dir 2_1/supervised-wheeze-spec-large --data ../data --augment spec --train_prop 1 --epochs 25 --evaluator fine-tune
#  python contrastive.py --mode test --task wheeze --log_dir 2_1/supervised-wheeze-spec-large --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task wheeze --log_dir 2_1/supervised-wheeze-time-large --data ../data --augment time --train_prop 1 --epochs 25 --evaluator fine-tune
#  python contrastive.py --mode test --task wheeze --log_dir 2_1/supervised-wheeze-time-large --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task wheeze --log_dir 2_1/supervised-wheeze-freq-large --data ../data --augment freq --train_prop 1 --epochs 25 --evaluator fine-tune
#  python contrastive.py --mode test --task wheeze --log_dir 2_1/supervised-wheeze-freq-large --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task wheeze --log_dir 2_1/supervised-wheeze-split-large --data ../data --augment split --train_prop 1 --epochs 25 --evaluator fine-tune
#  python contrastive.py --mode test --task wheeze --log_dir 2_1/supervised-wheeze-split-large --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task wheeze --log_dir 2_1/supervised-wheeze-spec-split-large --data ../data --augment spec+split --train_prop 1 --epochs 25 --evaluator fine-tune
#  python contrastive.py --mode test --task wheeze --log_dir 2_1/supervised-wheeze-spec-split-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task wheeze --log_dir 2_1/supervised-wheeze-large-full --data ../data --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
  python contrastive.py --mode test --task wheeze --log_dir 2_1/supervised-wheeze-large-full --data ../data --evaluator fine-tune
  #	python contrastive.py --mode train --task wheeze --log_dir 2_1/supervised-wheeze-raw-large-full --data ../data --augment raw --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
  #	python contrastive.py --mode test --task wheeze --log_dir 2_1/supervised-wheeze-raw-large-full --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task wheeze --log_dir 2_1/supervised-wheeze-spec-large-full --data ../data --augment spec --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#  python contrastive.py --mode test --task wheeze --log_dir 2_1/supervised-wheeze-spec-large-full --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task wheeze --log_dir 2_1/supervised-wheeze-time-large-full --data ../data --augment time --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#  python contrastive.py --mode test --task wheeze --log_dir 2_1/supervised-wheeze-time-large-full --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task wheeze --log_dir 2_1/supervised-wheeze-freq-large-full --data ../data --augment freq --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#  python contrastive.py --mode test --task wheeze --log_dir 2_1/supervised-wheeze-freq-large-full --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task wheeze --log_dir 2_1/supervised-wheeze-split-large-full --data ../data --augment split --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#  python contrastive.py --mode test --task wheeze --log_dir 2_1/supervised-wheeze-split-large-full --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task wheeze --log_dir 2_1/supervised-wheeze-spec-split-large-full --data ../data --augment spec+split --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#  python contrastive.py --mode test --task wheeze --log_dir 2_1/supervised-wheeze-spec-split-large-full --data ../data --evaluator fine-tune
done
# done
echo "Done"

