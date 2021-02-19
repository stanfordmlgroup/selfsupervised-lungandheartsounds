#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="exp 4 heartchallenge"
#SBATCH --output=out/exp4-heartchallenge-%j.out

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
  cp -a ../heart/logs/2_1/spec-pre-large/. ../heartchallenge/logs/2_1/spec-pre-heart-large

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/spec-pre-heart-large --data ../heartchallenge --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/spec-pre-heart-large --data ../heartchallenge --evaluator fine-tune

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/spec-pre-heart-large --data ../heartchallenge --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/spec-pre-heart-large --data ../heartchallenge --evaluator linear

  cp -a ../data/logs/2_1/spec-pre-large/. ../heartchallenge/logs/2_1/spec-pre-lung-large

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/spec-pre-lung-large --data ../heartchallenge --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/spec-pre-lung-large --data ../heartchallenge --evaluator fine-tune

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/spec-pre-lung-large --data ../heartchallenge --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/spec-pre-lung-large --data ../heartchallenge --evaluator linear


  cp -a ../heart/logs/2_1/time-pre-large/. ../heartchallenge/logs/2_1/time-pre-heart-large

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/time-pre-heart-large --data ../heartchallenge --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/time-pre-heart-large --data ../heartchallenge --evaluator fine-tune

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/time-pre-heart-large --data ../heartchallenge --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/time-pre-heart-large --data ../heartchallenge --evaluator linear

  cp -a ../data/logs/2_1/time-pre-large/. ../heartchallenge/logs/2_1/time-pre-lung-large

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/time-pre-lung-large --data ../heartchallenge --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/time-pre-lung-large --data ../heartchallenge --evaluator fine-tune

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/time-pre-lung-large --data ../heartchallenge --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/time-pre-lung-large --data ../heartchallenge --evaluator linear


  cp -a ../heart/logs/2_1/freq-pre-large/. ../heartchallenge/logs/2_1/freq-pre-heart-large

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/freq-pre-heart-large --data ../heartchallenge --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/freq-pre-heart-large --data ../heartchallenge --evaluator fine-tune

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/freq-pre-heart-large --data ../heartchallenge --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/freq-pre-heart-large --data ../heartchallenge --evaluator linear

  cp -a ../data/logs/2_1/freq-pre-large/. ../heartchallenge/logs/2_1/freq-pre-lung-large

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/freq-pre-lung-large --data ../heartchallenge --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/freq-pre-lung-large --data ../heartchallenge --evaluator fine-tune

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/freq-pre-lung-large --data ../heartchallenge --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/freq-pre-lung-large --data ../heartchallenge --evaluator linear


  cp -a ../heart/logs/2_1/split-pre-large/. ../heartchallenge/logs/2_1/split-pre-heart-large

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/split-pre-heart-large --data ../heartchallenge --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/split-pre-heart-large --data ../heartchallenge --evaluator fine-tune

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/split-pre-heart-large --data ../heartchallenge --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/split-pre-heart-large --data ../heartchallenge --evaluator linear

  cp -a ../data/logs/2_1/split-pre-large/. ../heartchallenge/logs/2_1/split-pre-lung-large

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/split-pre-lung-large --data ../heartchallenge --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/split-pre-lung-large --data ../heartchallenge --evaluator fine-tune

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/split-pre-lung-large --data ../heartchallenge --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/split-pre-lung-large --data ../heartchallenge --evaluator linear


  cp -a ../heart/logs/2_1/spec-split-pre-large/. ../heartchallenge/logs/2_1/spec-split-pre-heart-large

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/spec-split-pre-heart-large --data ../heartchallenge --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/spec-split-pre-heart-large --data ../heartchallenge --evaluator fine-tune

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/spec-split-pre-heart-large --data ../heartchallenge --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/spec-split-pre-heart-large --data ../heartchallenge --evaluator linear

  cp -a ../data/logs/2_1/spec-split-pre-large/. ../heartchallenge/logs/2_1/spec-split-pre-lung-large

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/spec-split-pre-lung-large --data ../heartchallenge --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/spec-split-pre-lung-large --data ../heartchallenge --evaluator fine-tune

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/spec-split-pre-lung-large --data ../heartchallenge --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/spec-split-pre-lung-large --data ../heartchallenge --evaluator linear

  python contrastive.py --mode train --task heartchallenge --log_dir 2_1/supervised-large --data ../heartchallenge --train_prop 1 --epochs 25 --evaluator fine-tune
  python contrastive.py --mode test --task heartchallenge --log_dir 2_1/supervised-large --data ../heartchallenge --evaluator fine-tune
  #	python contrastive.py --mode train --task heartchallenge --log_dir 2_1/supervised-raw-medium --data ../heartchallenge --augment raw --train_prop 1 --epochs 25 --evaluator fine-tune
  #	python contrastive.py --mode test --task heartchallenge --log_dir 2_1/supervised-raw-large --data ../heartchallenge --evaluator fine-tune
  #python contrastive.py --mode train --task heartchallenge --log_dir 2_1/supervised-spec-large --data ../heartchallenge --augment spec --train_prop 1 --epochs 25 --evaluator fine-tune
  #python contrastive.py --mode test --task heartchallenge --log_dir 2_1/supervised-spec-large --data ../heartchallenge --evaluator fine-tune
  #python contrastive.py --mode train --task heartchallenge --log_dir 2_1/supervised-time-large --data ../heartchallenge --augment time --train_prop 1 --epochs 25 --evaluator fine-tune
  #python contrastive.py --mode test --task heartchallenge --log_dir 2_1/supervised-time-large --data ../heartchallenge --evaluator fine-tune
  #python contrastive.py --mode train --task heartchallenge --log_dir 2_1/supervised-freq-large --data ../heartchallenge --augment freq --train_prop 1 --epochs 25 --evaluator fine-tune
  #python contrastive.py --mode test --task heartchallenge --log_dir 2_1/supervised-freq-large --data ../heartchallenge --evaluator fine-tune
  #python contrastive.py --mode train --task heartchallenge --log_dir 2_1/supervised-split-large --data ../heartchallenge --augment split --train_prop 1 --epochs 25 --evaluator fine-tune
  #python contrastive.py --mode test --task heartchallenge --log_dir 2_1/supervised-split-large --data ../heartchallenge --evaluator fine-tune
  #python contrastive.py --mode train --task heartchallenge --log_dir 2_1/supervised-spec-split-large --data ../heartchallenge --augment spec+split --train_prop 1 --epochs 25 --evaluator fine-tune
  #python contrastive.py --mode test --task heartchallenge --log_dir 2_1/supervised-spec-split-large --data ../heartchallenge --evaluator fine-tune
done
# done
echo "Done"

