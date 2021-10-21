#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="exp 5 heart"
#SBATCH --output=out/exp5-heart-%j.out

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate lungsounds
cd ../models

#python contrastive.py --mode train --task disease --log_dir 2_1/spec-pre-large --data ../data --evaluator fine-tune --train_prop 1 --epoch 25
#python contrastive.py --mode train --task disease --log_dir 2_1/time-pre-large --data ../data --evaluator fine-tune --train_prop 1 --epoch 25
#python contrastive.py --mode train --task disease --log_dir 2_1/freq-pre-large --data ../data --evaluator fine-tune --train_prop 1 --epoch 25
#python contrastive.py --mode train --task disease --log_dir 2_1/split-pre-large --data ../data --evaluator fine-tune --train_prop 1 --epoch 25
#python contrastive.py --mode train --task disease --log_dir 2_1/spec-split-pre-large --data ../data --evaluator fine-tune --train_prop 1 --epoch 25
#python contrastive.py --mode train --task heart --log_dir 2_1/spec-pre-large --data ../heart --evaluator fine-tune --train_prop 1 --epoch 25
#python contrastive.py --mode train --task heart --log_dir 2_1/time-pre-large --data ../heart --evaluator fine-tune --train_prop 1 --epoch 25
#python contrastive.py --mode train --task heart --log_dir 2_1/freq-pre-large --data ../heart --evaluator fine-tune --train_prop 1 --epoch 25
#python contrastive.py --mode train --task heart --log_dir 2_1/split-pre-large --data ../heart --evaluator fine-tune --train_prop 1 --epoch 25
#python contrastive.py --mode train --task heart --log_dir 2_1/spec-split-pre-large --data ../heart --evaluator fine-tune --train_prop 1 --epoch 25


cp -a ../data/logs/2_1/spec-pre-large/. ../heart/logs/2_1/spec-pre-lung-large
cp -a ../heart/logs/2_1/spec-pre-large/. ../data/logs/2_1/spec-pre-heart-large
python extract.py --log_dir ../data/logs/2_1/spec-pre-large --model_file evaluator_0.pt
python extract.py --log_dir ../heart/logs/2_1/spec-pre-large --model_file evaluator_0.pt
cp -a ../data/logs/2_1/spec-pre-large/. ../heart/logs/2_1/spec-pre-fine-lung-large
cp -a ../heart/logs/2_1/spec-pre-large/. ../data/logs/2_1/spec-pre-fine-heart-large


cp -a ../data/logs/2_1/time-pre-large/. ../heart/logs/2_1/time-pre-lung-large
cp -a ../heart/logs/2_1/time-pre-large/. ../data/logs/2_1/time-pre-heart-large
python extract.py --log_dir ../data/logs/2_1/time-pre-large --model_file evaluator_0.pt
python extract.py --log_dir ../heart/logs/2_1/time-pre-large --model_file evaluator_0.pt
cp -a ../data/logs/2_1/time-pre-large/. ../heart/logs/2_1/time-pre-fine-lung-large
cp -a ../heart/logs/2_1/time-pre-large/. ../data/logs/2_1/time-pre-fine-heart-large


cp -a ../data/logs/2_1/freq-pre-large/. ../heart/logs/2_1/freq-pre-lung-large
cp -a ../heart/logs/2_1/freq-pre-large/. ../data/logs/2_1/freq-pre-heart-large
python extract.py --log_dir ../data/logs/2_1/freq-pre-large --model_file evaluator_0.pt
python extract.py --log_dir ../heart/logs/2_1/freq-pre-large --model_file evaluator_0.pt
cp -a ../data/logs/2_1/freq-pre-large/. ../heart/logs/2_1/freq-pre-fine-lung-large
cp -a ../heart/logs/2_1/freq-pre-large/. ../data/logs/2_1/freq-pre-fine-heart-large

cp -a ../data/logs/2_1/split-pre-large/. ../heart/logs/2_1/split-pre-lung-large
cp -a ../heart/logs/2_1/split-pre-large/. ../data/logs/2_1/split-pre-heart-large
python extract.py --log_dir ../data/logs/2_1/split-pre-large --model_file evaluator_0.pt
python extract.py --log_dir ../heart/logs/2_1/split-pre-large --model_file evaluator_0.pt
cp -a ../data/logs/2_1/split-pre-large/. ../heart/logs/2_1/split-pre-fine-lung-large
cp -a ../heart/logs/2_1/split-pre-large/. ../data/logs/2_1/split-pre-fine-heart-large

cp -a ../heart/logs/2_1/spec-split-pre-large/. ../data/logs/2_1/spec-split-pre-heart-large
cp -a ../data/logs/2_1/spec-split-pre-large/. ../heart/logs/2_1/spec-split-pre-lung-large
python extract.py --log_dir ../data/logs/2_1/spec-split-pre-large --model_file evaluator_0.pt
python extract.py --log_dir ../heart/logs/2_1/spec-split-pre-large --model_file evaluator_0.pt
cp -a ../data/logs/2_1/spec-split-pre-large/. ../heart/logs/2_1/spec-split-pre-fine-lung-large
cp -a ../heart/logs/2_1/spec-split-pre-large/. ../data/logs/2_1/spec-split-pre-fine-heart-large

wait
cd ../scripts
sbatch Exp5Lung.sh &
cd ../models

for i in 1 2 3 4 5
do

  python contrastive.py --mode train --task heart --log_dir 2_1/spec-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/spec-pre-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/spec-pre-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/spec-pre-lung-large --data ../heart --evaluator linear

  python contrastive.py --mode train --task heart --log_dir 2_1/spec-pre-fine-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/spec-pre-fine-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/spec-pre-fine-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/spec-pre-fine-lung-large --data ../heart --evaluator linear


  python contrastive.py --mode train --task heart --log_dir 2_1/time-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/time-pre-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/time-pre-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/time-pre-lung-large --data ../heart --evaluator linear

  python contrastive.py --mode train --task heart --log_dir 2_1/time-pre-fine-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/time-pre-fine-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/time-pre-fine-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/time-pre-fine-lung-large --data ../heart --evaluator linear

  python contrastive.py --mode train --task heart --log_dir 2_1/freq-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/freq-pre-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/freq-pre-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/freq-pre-lung-large --data ../heart --evaluator linear

  python contrastive.py --mode train --task heart --log_dir 2_1/freq-pre-fine-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/freq-pre-fine-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/freq-pre-fine-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/freq-pre-fine-lung-large --data ../heart --evaluator linear


  python contrastive.py --mode train --task heart --log_dir 2_1/split-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/split-pre-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/split-pre-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/split-pre-lung-large --data ../heart --evaluator linear

  python contrastive.py --mode train --task heart --log_dir 2_1/split-pre-fine-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/split-pre-fine-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/split-pre-fine-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/split-pre-fine-lung-large --data ../heart --evaluator linear


  python contrastive.py --mode train --task heart --log_dir 2_1/spec-split-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/spec-split-pre-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/spec-split-pre-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/spec-split-pre-lung-large --data ../heart --evaluator linear

  python contrastive.py --mode train --task heart --log_dir 2_1/spec-split-pre-fine-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/spec-split-pre-fine-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/spec-split-pre-fine-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/spec-split-pre-fine-lung-large --data ../heart --evaluator linear
done

python contrastive.py --mode pretrain --task heart --log_dir 2_1/spec-pre-lung-large --data ../heart --augment spec --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task heart --log_dir 2_1/spec-pre-fine-lung-large --data ../heart --augment spec --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task heart --log_dir 2_1/time-pre-lung-large --data ../heart --augment time --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task heart --log_dir 2_1/time-pre-fine-lung-large --data ../heart --augment time --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task heart --log_dir 2_1/freq-pre-lung-large --data ../heart --augment freq --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task heart --log_dir 2_1/freq-pre-fine-lung-large --data ../heart --augment freq --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task heart --log_dir 2_1/split-pre-lung-large --data ../heart --augment split --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task heart --log_dir 2_1/split-pre-fine-lung-large --data ../heart --augment split --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task heart --log_dir 2_1/spec-split-pre-lung-large --data ../heart --augment spec+split --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task heart --log_dir 2_1/spec-split-pre-fine-lung-large --data ../heart --augment spec+split --train_prop 1.0 --epoch 10

for i in 1 2 3 4 5
do

  python contrastive.py --mode train --task heart --log_dir 2_1/spec-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/spec-pre-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/spec-pre-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/spec-pre-lung-large --data ../heart --evaluator linear

  python contrastive.py --mode train --task heart --log_dir 2_1/spec-pre-fine-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/spec-pre-fine-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/spec-pre-fine-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/spec-pre-fine-lung-large --data ../heart --evaluator linear


  python contrastive.py --mode train --task heart --log_dir 2_1/time-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/time-pre-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/time-pre-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/time-pre-lung-large --data ../heart --evaluator linear

  python contrastive.py --mode train --task heart --log_dir 2_1/time-pre-fine-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/time-pre-fine-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/time-pre-fine-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/time-pre-fine-lung-large --data ../heart --evaluator linear

  python contrastive.py --mode train --task heart --log_dir 2_1/freq-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/freq-pre-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/freq-pre-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/freq-pre-lung-large --data ../heart --evaluator linear

  python contrastive.py --mode train --task heart --log_dir 2_1/freq-pre-fine-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/freq-pre-fine-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/freq-pre-fine-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/freq-pre-fine-lung-large --data ../heart --evaluator linear


  python contrastive.py --mode train --task heart --log_dir 2_1/split-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/split-pre-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/split-pre-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/split-pre-lung-large --data ../heart --evaluator linear

  python contrastive.py --mode train --task heart --log_dir 2_1/split-pre-fine-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/split-pre-fine-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/split-pre-fine-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/split-pre-fine-lung-large --data ../heart --evaluator linear


  python contrastive.py --mode train --task heart --log_dir 2_1/spec-split-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/spec-split-pre-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/spec-split-pre-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/spec-split-pre-lung-large --data ../heart --evaluator linear

  python contrastive.py --mode train --task heart --log_dir 2_1/spec-split-pre-fine-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
  python contrastive.py --mode test --task heart --log_dir 2_1/spec-split-pre-fine-lung-large --data ../heart --evaluator fine-tune

  python contrastive.py --mode train --task heart --log_dir 2_1/spec-split-pre-fine-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
  python contrastive.py --mode test --task heart --log_dir 2_1/spec-split-pre-fine-lung-large --data ../heart --evaluator linear
done

# done
echo "Done"

