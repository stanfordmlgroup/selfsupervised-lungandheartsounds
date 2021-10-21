#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="exp 5 disease"
#SBATCH --output=out/exp5-disease-%j.out

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
  python contrastive.py --mode train --task disease --log_dir 2_1/spec-pre-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-pre-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/spec-pre-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-pre-heart-large --data ../data --evaluator linear


  python contrastive.py --mode train --task disease --log_dir 2_1/spec-pre-fine-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-pre-fine-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/spec-pre-fine-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-pre-fine-heart-large --data ../data --evaluator linear


  python contrastive.py --mode train --task disease --log_dir 2_1/time-pre-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/time-pre-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/time-pre-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/time-pre-heart-large --data ../data --evaluator linear

  python contrastive.py --mode train --task disease --log_dir 2_1/time-pre-fine-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/time-pre-fine-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/time-pre-fine-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/time-pre-fine-heart-large --data ../data --evaluator linear



  python contrastive.py --mode train --task disease --log_dir 2_1/freq-pre-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/freq-pre-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/freq-pre-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/freq-pre-heart-large --data ../data --evaluator linear

  python contrastive.py --mode train --task disease --log_dir 2_1/freq-pre-fine-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/freq-pre-fine-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/freq-pre-fine-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/freq-pre-fine-heart-large --data ../data --evaluator linear

  python contrastive.py --mode train --task disease --log_dir 2_1/split-pre-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/split-pre-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/split-pre-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/split-pre-heart-large --data ../data --evaluator linear

  python contrastive.py --mode train --task disease --log_dir 2_1/split-pre-fine-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/split-pre-fine-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/split-pre-fine-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/split-pre-fine-heart-large --data ../data --evaluator linear

  python contrastive.py --mode train --task disease --log_dir 2_1/spec-split-pre-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-split-pre-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/spec-split-pre-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-split-pre-heart-large --data ../data --evaluator linear

  python contrastive.py --mode train --task disease --log_dir 2_1/spec-split-pre-fine-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-split-pre-fine-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/spec-split-pre-fine-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-split-pre-fine-heart-large --data ../data --evaluator linear
done

python contrastive.py --mode pretrain --task disease --log_dir 2_1/spec-pre-heart-large --data ../data --augment spec --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task disease --log_dir 2_1/spec-pre-fine-heart-large --data ../data --augment spec --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task disease --log_dir 2_1/time-pre-heart-large --data ../data --augment time --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task disease --log_dir 2_1/time-pre-fine-heart-large --data ../data --augment time --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task disease --log_dir 2_1/freq-pre-heart-large --data ../data --augment freq --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task disease --log_dir 2_1/freq-pre-fine-heart-large --data ../data --augment freq --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task disease --log_dir 2_1/split-pre-heart-large --data ../data --augment split --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task disease --log_dir 2_1/split-pre-fine-heart-large --data ../data --augment split --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task disease --log_dir 2_1/spec-split-pre-heart-large --data ../data --augment spec+split --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task disease --log_dir 2_1/spec-split-pre-fine-heart-large --data ../data --augment spec+split --train_prop 1.0 --epoch 10

for i in 1 2 3 4 5
do
  python contrastive.py --mode train --task disease --log_dir 2_1/spec-pre-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-pre-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/spec-pre-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-pre-heart-large --data ../data --evaluator linear

  python contrastive.py --mode train --task disease --log_dir 2_1/spec-pre-fine-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-pre-fine-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/spec-pre-fine-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-pre-fine-heart-large --data ../data --evaluator linear


  python contrastive.py --mode train --task disease --log_dir 2_1/time-pre-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/time-pre-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/time-pre-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/time-pre-heart-large --data ../data --evaluator linear

  python contrastive.py --mode train --task disease --log_dir 2_1/time-pre-fine-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/time-pre-fine-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/time-pre-fine-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/time-pre-fine-heart-large --data ../data --evaluator linear



  python contrastive.py --mode train --task disease --log_dir 2_1/freq-pre-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/freq-pre-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/freq-pre-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/freq-pre-heart-large --data ../data --evaluator linear

  python contrastive.py --mode train --task disease --log_dir 2_1/freq-pre-fine-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/freq-pre-fine-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/freq-pre-fine-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/freq-pre-fine-heart-large --data ../data --evaluator linear


  python contrastive.py --mode train --task disease --log_dir 2_1/split-pre-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/split-pre-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/split-pre-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/split-pre-heart-large --data ../data --evaluator linear

  python contrastive.py --mode train --task disease --log_dir 2_1/split-pre-fine-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/split-pre-fine-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/split-pre-fine-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/split-pre-fine-heart-large --data ../data --evaluator linear

  python contrastive.py --mode train --task disease --log_dir 2_1/spec-split-pre-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-split-pre-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/spec-split-pre-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-split-pre-heart-large --data ../data --evaluator linear

  python contrastive.py --mode train --task disease --log_dir 2_1/spec-split-pre-fine-heart-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-split-pre-fine-heart-large --data ../data --evaluator fine-tune

  python contrastive.py --mode train --task disease --log_dir 2_1/spec-split-pre-fine-heart-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-split-pre-fine-heart-large --data ../data --evaluator linear
done

# done
echo "Done"

