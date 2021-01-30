#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="aihc lungheart exp 2 selfsuper"
#SBATCH --output=out/exp2-selfsupervised-%j.out

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

# python contrastive.py --mode pretrain --task disease --log_dir raw-pre-lung-large --data ../data --augment raw --train_prop 1.0 --epoch 10

# python contrastive.py --mode train --task disease --log_dir raw-pre-lung-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
# python contrastive.py --mode test --task disease --log_dir raw-pre-lung-large --data ../data
# python contrastive.py --mode train --task disease --log_dir raw-pre-lung-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
# python contrastive.py --mode test --task disease --log_dir raw-pre-lung-large --data ../data
# python contrastive.py --mode train --task disease --log_dir raw-pre-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
# python contrastive.py --mode test --task disease --log_dir raw-pre-lung-large --data ../data

# python contrastive.py --mode train --task disease --log_dir raw-pre-lung-large --data ../data --evaluator linear --train_prop .01
# python contrastive.py --mode test --task disease --log_dir raw-pre-lung-large --data ../data
# python contrastive.py --mode train --task disease --log_dir raw-pre-lung-large --data ../data --evaluator linear --train_prop .1
# python contrastive.py --mode test --task disease --log_dir raw-pre-lung-large --data ../data
# python contrastive.py --mode train --task disease --log_dir raw-pre-lung-large --data ../data --evaluator linear --train_prop 1.
# python contrastive.py --mode test --task disease --log_dir raw-pre-lung-large --data ../data

# python contrastive.py --mode pretrain --task disease --log_dir spec-pre-lung-large --data ../data --augment spec --train_prop 1.0 --epoch 10

# python contrastive.py --mode train --task disease --log_dir spec-pre-lung-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
# python contrastive.py --mode test --task disease --log_dir spec-pre-lung-large --data ../data
# python contrastive.py --mode train --task disease --log_dir spec-pre-lung-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
# python contrastive.py --mode test --task disease --log_dir spec-pre-lung-large --data ../data
# python contrastive.py --mode train --task disease --log_dir spec-pre-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
# python contrastive.py --mode test --task disease --log_dir spec-pre-lung-large --data ../data

# python contrastive.py --mode train --task disease --log_dir spec-pre-lung-large --data ../data --evaluator linear --train_prop .01
# python contrastive.py --mode test --task disease --log_dir spec-pre-lung-large --data ../data
# python contrastive.py --mode train --task disease --log_dir spec-pre-lung-large --data ../data --evaluator linear --train_prop .1
# python contrastive.py --mode test --task disease --log_dir spec-pre-lung-large --data ../data
# python contrastive.py --mode train --task disease --log_dir spec-pre-lung-large --data ../data --evaluator linear --train_prop 1.
# python contrastive.py --mode test --task disease --log_dir spec-pre-lung-large --data ../data

# python contrastive.py --mode pretrain --task disease --log_dir split-pre-lung-large --data ../data --augment split --train_prop 1.0 --epoch 10

# python contrastive.py --mode train --task disease --log_dir split-pre-lung-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
# python contrastive.py --mode test --task disease --log_dir split-pre-lung-large --data ../data
# python contrastive.py --mode train --task disease --log_dir split-pre-lung-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
# python contrastive.py --mode test --task disease --log_dir split-pre-lung-large --data ../data
# python contrastive.py --mode train --task disease --log_dir split-pre-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
# python contrastive.py --mode test --task disease --log_dir split-pre-lung-large --data ../data

# python contrastive.py --mode train --task disease --log_dir split-pre-lung-large --data ../data --evaluator linear --train_prop .01
# python contrastive.py --mode test --task disease --log_dir split-pre-lung-large --data ../data
# python contrastive.py --mode train --task disease --log_dir split-pre-lung-large --data ../data --evaluator linear --train_prop .1
# python contrastive.py --mode test --task disease --log_dir split-pre-lung-large --data ../data
# python contrastive.py --mode train --task disease --log_dir split-pre-lung-large --data ../data --evaluator linear --train_prop 1.
# python contrastive.py --mode test --task disease --log_dir split-pre-lung-large --data ../data

python contrastive.py --mode pretrain --task disease --log_dir spec-split-pre-lung-large --data ../data --augment spec+split --train_prop 1.0 --epoch 10

python contrastive.py --mode train --task disease --log_dir spec-split-pre-lung-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
python contrastive.py --mode test --task disease --log_dir spec-split-pre-lung-large --data ../data
python contrastive.py --mode train --task disease --log_dir spec-split-pre-lung-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
python contrastive.py --mode test --task disease --log_dir spec-split-pre-lung-large --data ../data
python contrastive.py --mode train --task disease --log_dir spec-split-pre-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task disease --log_dir spec-split-pre-lung-large --data ../data

python contrastive.py --mode train --task disease --log_dir spec-split-pre-lung-large --data ../data --evaluator linear --train_prop .01
python contrastive.py --mode test --task disease --log_dir spec-split-pre-lung-large --data ../data
python contrastive.py --mode train --task disease --log_dir spec-split-pre-lung-large --data ../data --evaluator linear --train_prop .1
python contrastive.py --mode test --task disease --log_dir spec-split-pre-lung-large --data ../data
python contrastive.py --mode train --task disease --log_dir spec-split-pre-lung-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task disease --log_dir spec-split-pre-lung-large --data ../data


# done
echo "Done"

