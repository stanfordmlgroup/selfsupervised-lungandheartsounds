#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="aihc lungheart exp 4 selfsuper"
#SBATCH --output=out/exp4-selfsupervised-%j.out

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

python contrastive.py --mode train --task heartchallenge --log_dir raw-pre-large --data ../heartchallenge --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heartchallenge --log_dir raw-pre-large --data ../heartchallenge

python contrastive.py --mode train --task heartchallenge --log_dir raw-pre-large --data ../heartchallenge --evaluator linear --train_prop 1.
python contrastive.py --mode test --task heartchallenge --log_dir raw-pre-large --data ../heartchallenge

python contrastive.py --mode train --task heartchallenge --log_dir spec-pre-large --data ../heartchallenge --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heartchallenge --log_dir spec-pre-large --data ../heartchallenge

python contrastive.py --mode train --task heartchallenge --log_dir spec-pre-large --data ../heartchallenge --evaluator linear --train_prop 1.
python contrastive.py --mode test --task heartchallenge --log_dir spec-pre-large --data ../heartchallenge

python contrastive.py --mode train --task heartchallenge --log_dir split-pre-large --data ../heartchallenge --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heartchallenge --log_dir split-pre-large --data ../heartchallenge

python contrastive.py --mode train --task heartchallenge --log_dir split-pre-large --data ../heartchallenge --evaluator linear --train_prop 1.
python contrastive.py --mode test --task heartchallenge --log_dir split-pre-large --data ../heartchallenge

python contrastive.py --mode train --task heartchallenge --log_dir spec-split-pre-large --data ../heartchallenge --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heartchallenge --log_dir spec-split-pre-large --data ../heartchallenge

python contrastive.py --mode train --task heartchallenge --log_dir spec-split-pre-large --data ../heartchallenge --evaluator linear --train_prop 1.
python contrastive.py --mode test --task heartchallenge --log_dir spec-split-pre-large --data ../heartchallenge

python contrastive.py --mode train --task heartchallenge --log_dir raw-pre-lung-large --data ../heartchallenge --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heartchallenge --log_dir raw-pre-lung-large --data ../heartchallenge

python contrastive.py --mode train --task heartchallenge --log_dir raw-pre-lung-large --data ../heartchallenge --evaluator linear --train_prop 1.
python contrastive.py --mode test --task heartchallenge --log_dir raw-pre-lung-large --data ../heartchallenge

python contrastive.py --mode train --task heartchallenge --log_dir spec-pre-lung-large --data ../heartchallenge --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heartchallenge --log_dir spec-pre-lung-large --data ../heartchallenge

python contrastive.py --mode train --task heartchallenge --log_dir spec-pre-lung-large --data ../heartchallenge --evaluator linear --train_prop 1.
python contrastive.py --mode test --task heartchallenge --log_dir spec-pre-lung-large --data ../heartchallenge

python contrastive.py --mode train --task heartchallenge --log_dir split-pre-lung-large --data ../heartchallenge --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heartchallenge --log_dir split-pre-lung-large --data ../heartchallenge

python contrastive.py --mode train --task heartchallenge --log_dir split-pre-lung-large --data ../heartchallenge --evaluator linear --train_prop 1.
python contrastive.py --mode test --task heartchallenge --log_dir split-pre-lung-large --data ../heartchallenge

python contrastive.py --mode train --task heartchallenge --log_dir spec-split-pre-lung-large --data ../heartchallenge --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heartchallenge --log_dir spec-split-pre-lung-large --data ../heartchallenge

python contrastive.py --mode train --task heartchallenge --log_dir spec-split-pre-lung-large --data ../heartchallenge --evaluator linear --train_prop 1.
python contrastive.py --mode test --task heartchallenge --log_dir spec-split-pre-lung-large --data ../heartchallenge

python contrastive.py --mode train --task crackle --log_dir raw-pre-crackle-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task crackle --log_dir raw-pre-crackle-large --data ../data

python contrastive.py --mode train --task crackle --log_dir raw-pre-crackle-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task crackle --log_dir raw-pre-crackle-large --data ../data

python contrastive.py --mode train --task crackle --log_dir spec-pre-crackle-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task crackle --log_dir spec-pre-crackle-large --data ../data

python contrastive.py --mode train --task crackle --log_dir spec-pre-crackle-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task crackle --log_dir spec-pre-crackle-large --data ../data

python contrastive.py --mode train --task crackle --log_dir split-pre-crackle-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task crackle --log_dir split-pre-crackle-large --data ../data

python contrastive.py --mode train --task crackle --log_dir split-pre-crackle-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task crackle --log_dir split-pre-crackle-large --data ../data

python contrastive.py --mode train --task crackle --log_dir spec-split-pre-crackle-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task crackle --log_dir spec-split-pre-crackle-large --data ../data

python contrastive.py --mode train --task crackle --log_dir spec-split-pre-crackle-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task crackle --log_dir spec-split-pre-crackle-large --data ../data

python contrastive.py --mode train --task crackle --log_dir raw-pre-crackle-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task crackle --log_dir raw-pre-crackle-lung-large --data ../data

python contrastive.py --mode train --task crackle --log_dir raw-pre-crackle-lung-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task crackle --log_dir raw-pre-crackle-lung-large --data ../data

python contrastive.py --mode train --task crackle --log_dir spec-pre-crackle-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task crackle --log_dir spec-pre-crackle-lung-large --data ../data

python contrastive.py --mode train --task crackle --log_dir spec-pre-crackle-lung-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task crackle --log_dir spec-pre-crackle-lung-large --data ../data

python contrastive.py --mode train --task crackle --log_dir split-pre-crackle-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task crackle --log_dir split-pre-crackle-lung-large --data ../data

python contrastive.py --mode train --task crackle --log_dir split-pre-crackle-lung-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task crackle --log_dir split-pre-crackle-lung-large --data ../data

python contrastive.py --mode train --task crackle --log_dir spec-split-pre-crackle-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task crackle --log_dir spec-split-pre-crackle-lung-large --data ../data

python contrastive.py --mode train --task crackle --log_dir spec-split-pre-crackle-lung-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task crackle --log_dir spec-split-pre-crackle-lung-large --data ../data


python contrastive.py --mode train --task wheeze --log_dir raw-pre-wheeze-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task wheeze --log_dir raw-pre-wheeze-large --data ../data

python contrastive.py --mode train --task wheeze --log_dir raw-pre-wheeze-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task wheeze --log_dir raw-pre-wheeze-large --data ../data

python contrastive.py --mode train --task wheeze --log_dir spec-pre-wheeze-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task wheeze --log_dir spec-pre-wheeze-large --data ../data

python contrastive.py --mode train --task wheeze --log_dir spec-pre-wheeze-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task wheeze --log_dir spec-pre-wheeze-large --data ../data

python contrastive.py --mode train --task wheeze --log_dir split-pre-wheeze-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task wheeze --log_dir split-pre-wheeze-large --data ../data

python contrastive.py --mode train --task wheeze --log_dir split-pre-wheeze-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task wheeze --log_dir split-pre-wheeze-large --data ../data

python contrastive.py --mode train --task wheeze --log_dir spec-split-pre-wheeze-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task wheeze --log_dir spec-split-pre-wheeze-large --data ../data

python contrastive.py --mode train --task wheeze --log_dir spec-split-pre-wheeze-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task wheeze --log_dir spec-split-pre-wheeze-large --data ../data

python contrastive.py --mode train --task wheeze --log_dir raw-pre-wheeze-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task wheeze --log_dir raw-pre-wheeze-lung-large --data ../data

python contrastive.py --mode train --task wheeze --log_dir raw-pre-wheeze-lung-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task wheeze --log_dir raw-pre-wheeze-lung-large --data ../data

python contrastive.py --mode train --task wheeze --log_dir spec-pre-wheeze-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task wheeze --log_dir spec-pre-wheeze-lung-large --data ../data

python contrastive.py --mode train --task wheeze --log_dir spec-pre-wheeze-lung-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task wheeze --log_dir spec-pre-wheeze-lung-large --data ../data

python contrastive.py --mode train --task wheeze --log_dir split-pre-wheeze-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task wheeze --log_dir split-pre-wheeze-lung-large --data ../data

python contrastive.py --mode train --task wheeze --log_dir split-pre-wheeze-lung-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task wheeze --log_dir split-pre-wheeze-lung-large --data ../data

python contrastive.py --mode train --task wheeze --log_dir spec-split-pre-wheeze-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task wheeze --log_dir spec-split-pre-wheeze-lung-large --data ../data

python contrastive.py --mode train --task wheeze --log_dir spec-split-pre-wheeze-lung-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task wheeze --log_dir spec-split-pre-wheeze-lung-large --data ../data

# done
echo "Done"

