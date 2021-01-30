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

cp -r ../data/logs/1_24/split-pre-large ../heart/logs/1_24/split-pre-lung-large

python contrastive.py --mode train --task heart --log_dir 1_24/split-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heart --log_dir 1_24/split-pre-lung-large --data ../heart --evaluator fine-tune

python contrastive.py --mode train --task heart --log_dir 1_24/split-pre-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
python contrastive.py --mode test --task heart --log_dir 1_24/split-pre-lung-large --data ../heart --evaluator linear

python contrastive.py --mode pretrain --task heart --log_dir 1_24/split-pre-lung-large --data ../heart --augment split --train_prop 1.0 --epoch 10

python contrastive.py --mode train --task heart --log_dir 1_24/split-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heart --log_dir 1_24/split-pre-lung-large --data ../heart --evaluator fine-tune

python contrastive.py --mode train --task heart --log_dir 1_24/split-pre-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
python contrastive.py --mode test --task heart --log_dir 1_24/split-pre-lung-large --data ../heart --evaluator linear

python contrastive.py --mode train --task disease --log_dir 1_24/split-pre-large --data ../data --evaluator fine-tune --train_prop 1 --epoch 25
python extract.py --log_dir ../data/logs/1_24/split-pre-large --model_file evaluator_0.pt
cp -r ../data/logs/1_24/split-pre-large ../heart/logs/1_24/split-pre-fine-lung-large

python contrastive.py --mode pretrain --task heart --log_dir 1_24/split-pre-fine-lung-large --data ../heart --augment split --train_prop 1.0 --epoch 10

python contrastive.py --mode train --task heart --log_dir 1_24/split-pre-fine-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heart --log_dir 1_24/split-pre-fine-lung-large --data ../heart --evaluator fine-tune

python contrastive.py --mode train --task heart --log_dir 1_24/split-pre-fine-lung-large --data ../heart --evaluator linear --train_prop 1. --epoch 1000
python contrastive.py --mode test --task heart --log_dir 1_24/split-pre-fine-lung-large --data ../heart --evaluator linear




# done
echo "Done"

