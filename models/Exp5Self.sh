#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="aihc lungheart exp 5 selfsuper"
#SBATCH --output=out/exp5-selfsupervised-%j.out

# only use the following if you want email notification
#SBATCH --mail-user=prathams@stanford.edu
#SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
source ~/.bashrc
conda activate lungsounds

# python contrastive.py --mode pretrain --task heart --log_dir raw-pre-lung-large --data ../heart --augment raw --train_prop 1.0 --epoch 10

# python contrastive.py --mode train --task heart --log_dir raw-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
# python contrastive.py --mode test --task heart --log_dir raw-pre-lung-large --data ../heart --evaluator fine-tune

# python contrastive.py --mode train --task heart --log_dir raw-pre-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
# python contrastive.py --mode test --task heart --log_dir raw-pre-lung-large --data ../heart --evaluator linear

# python contrastive.py --mode pretrain --task heart --log_dir spec-pre-lung-large --data ../heart --augment spec --train_prop 1.0 --epoch 10

# python contrastive.py --mode train --task heart --log_dir spec-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
# python contrastive.py --mode test --task heart --log_dir spec-pre-lung-large --data ../heart --evaluator fine-tune

# python contrastive.py --mode train --task heart --log_dir spec-pre-lung-large --data ../heart --evaluator linear --train_prop 1.--epoch 1000
# python contrastive.py --mode test --task heart --log_dir spec-pre-lung-large --data ../heart --evaluator linear

# python contrastive.py --mode pretrain --task heart --log_dir split-pre-lung-large --data ../heart --augment split --train_prop 1.0 --epoch 10

# python contrastive.py --mode train --task heart --log_dir split-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
# python contrastive.py --mode test --task heart --log_dir split-pre-lung-large --data ../heart --evaluator fine-tune

# python contrastive.py --mode train --task heart --log_dir split-pre-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
# python contrastive.py --mode test --task heart --log_dir split-pre-lung-large --data ../heart --evaluator linear

# python contrastive.py --mode pretrain --task heart --log_dir spec-split-pre-lung-large --data ../heart --augment spec+split --train_prop 1.0 --epoch 10

# python contrastive.py --mode train --task heart --log_dir spec-split-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
# python contrastive.py --mode test --task heart --log_dir spec-split-pre-lung-large --data ../heart --evaluator fine-tune

# python contrastive.py --mode train --task heart --log_dir spec-split-pre-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
# python contrastive.py --mode test --task heart --log_dir spec-split-pre-lung-large --data ../heart --evaluator linear

python contrastive.py --mode pretrain --task heart --log_dir raw-pre-fine-lung-large --data ../heart --augment raw --train_prop 1.0 --epoch 10

python contrastive.py --mode train --task heart --log_dir raw-pre-fine-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heart --log_dir raw-pre-fine-lung-large --data ../heart --evaluator fine-tune

python contrastive.py --mode train --task heart --log_dir raw-pre-fine-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
python contrastive.py --mode test --task heart --log_dir raw-pre-fine-lung-large --data ../heart --evaluator linear

python contrastive.py --mode pretrain --task heart --log_dir spec-pre-fine-lung-large --data ../heart --augment spec --train_prop 1.0 --epoch 10

python contrastive.py --mode train --task heart --log_dir spec-pre-fine-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heart --log_dir spec-pre-fine-lung-large --data ../heart --evaluator fine-tune

python contrastive.py --mode train --task heart --log_dir spec-pre-fine-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
python contrastive.py --mode test --task heart --log_dir spec-pre-fine-lung-large --data ../heart --evaluator linear

python contrastive.py --mode pretrain --task heart --log_dir split-pre-fine-lung-large --data ../heart --augment split --train_prop 1.0 --epoch 10

python contrastive.py --mode train --task heart --log_dir split-pre-fine-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heart --log_dir split-pre-fine-lung-large --data ../heart --evaluator fine-tune

python contrastive.py --mode train --task heart --log_dir split-pre-fine-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
python contrastive.py --mode test --task heart --log_dir split-pre-fine-lung-large --data ../heart --evaluator linear

python contrastive.py --mode pretrain --task heart --log_dir spec-split-pre-fine-lung-large --data ../heart --augment spec+split --train_prop 1.0 --epoch 10

python contrastive.py --mode train --task heart --log_dir spec-split-pre-fine-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heart --log_dir spec-split-pre-fine-lung-large --data ../heart --evaluator fine-tune

python contrastive.py --mode train --task heart --log_dir spec-split-pre-fine-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
python contrastive.py --mode test --task heart --log_dir spec-split-pre-fine-lung-large --data ../heart --evaluator linear

python contrastive.py --mode pretrain --task disease --log_dir raw-pre-large --data ../data --augment raw --train_prop 1.0 --epoch 10

python contrastive.py --mode train --task disease --log_dir raw-pre-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task disease --log_dir raw-pre-large --data ../data --evaluator fine-tune

python contrastive.py --mode train --task disease --log_dir raw-pre-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
python contrastive.py --mode test --task disease --log_dir raw-pre-large --data ../data --evaluator linear

python contrastive.py --mode pretrain --task disease --log_dir spec-pre-large --data ../data --augment spec --train_prop 1.0 --epoch 10

python contrastive.py --mode train --task disease --log_dir spec-pre-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task disease --log_dir spec-pre-large --data ../data --evaluator fine-tune

python contrastive.py --mode train --task disease --log_dir spec-pre-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
python contrastive.py --mode test --task disease --log_dir spec-pre-large --data ../data --evaluator linear

python contrastive.py --mode pretrain --task disease --log_dir split-pre-large --data ../data --augment split --train_prop 1.0 --epoch 10

python contrastive.py --mode train --task disease --log_dir split-pre-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task disease --log_dir split-pre-large --data ../data --evaluator fine-tune

python contrastive.py --mode train --task disease --log_dir split-pre-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
python contrastive.py --mode test --task disease --log_dir split-pre-large --data ../data --evaluator linear

python contrastive.py --mode pretrain --task disease --log_dir spec-split-pre-large --data ../data --augment spec+split --train_prop 1.0 --epoch 10

python contrastive.py --mode train --task disease --log_dir spec-split-pre-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task disease --log_dir spec-split-pre-large --data ../data --evaluator fine-tune

python contrastive.py --mode train --task disease --log_dir spec-split-pre-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
python contrastive.py --mode test --task disease --log_dir spec-split-pre-large --data ../data --evaluator linear

python contrastive.py --mode pretrain --task disease --log_dir raw-pre-fine-large --data ../data --augment raw --train_prop 1.0 --epoch 10

python contrastive.py --mode train --task disease --log_dir raw-pre-fine-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task disease --log_dir raw-pre-fine-large --data ../data --evaluator fine-tune

python contrastive.py --mode train --task disease --log_dir raw-pre-fine-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
python contrastive.py --mode test --task disease --log_dir raw-pre-fine-large --data ../data --evaluator linear

python contrastive.py --mode pretrain --task disease --log_dir spec-pre-fine-large --data ../data --augment spec --train_prop 1.0 --epoch 10

python contrastive.py --mode train --task disease --log_dir spec-pre-fine-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task disease --log_dir spec-pre-fine-large --data ../data --evaluator fine-tune

python contrastive.py --mode train --task disease --log_dir spec-pre-fine-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
python contrastive.py --mode test --task disease --log_dir spec-pre-fine-large --data ../data --evaluator linear

python contrastive.py --mode pretrain --task disease --log_dir split-pre-fine-large --data ../data --augment split --train_prop 1.0 --epoch 10

python contrastive.py --mode train --task disease --log_dir split-pre-fine-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task disease --log_dir split-pre-fine-large --data ../data --evaluator fine-tune

python contrastive.py --mode train --task disease --log_dir split-pre-fine-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
python contrastive.py --mode test --task disease --log_dir split-pre-fine-large --data ../data --evaluator linear

python contrastive.py --mode pretrain --task disease --log_dir spec-split-pre-fine-large --data ../data --augment spec+split --train_prop 1.0 --epoch 10

python contrastive.py --mode train --task disease --log_dir spec-split-pre-fine-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task disease --log_dir spec-split-pre-fine-large --data ../data --evaluator fine-tune

python contrastive.py --mode train --task disease --log_dir spec-split-pre-fine-large --data ../data --evaluator linear --train_prop 1. --epoch 1000
python contrastive.py --mode test --task disease --log_dir spec-split-pre-fine-large --data ../data --evaluator linear

# done
echo "Done"

