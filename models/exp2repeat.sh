#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=144:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="exp 2 repeat aihc lung"
#SBATCH --output=out/exp2-repeat-%j.out

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
source ~/.bashrc
conda activate lungsounds

for i in 1 2 3 4 5 6 7 8 9 10
do
	python contrastive.py --mode train --task disease --log_dir repeat/supervised-small --data ../data --train_prop .01 --epochs 25
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-small --data ../data
	python contrastive.py --mode train --task disease --log_dir repeat/supervised-raw-small --data ../data --augment raw --train_prop .01 --epochs 25
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-raw-small --data ../data
	python contrastive.py --mode train --task disease --log_dir repeat/supervised-spec-small --data ../data --augment spec --train_prop .01 --epochs 25
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-spec-small --data ../data
	python contrastive.py --mode train --task disease --log_dir repeat/supervised-split-small --data ../data --augment split --train_prop .01 --epochs 25
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-split-small --data ../data
	python contrastive.py --mode train --task disease --log_dir repeat/supervised-spec-split-small --data ../data --augment spec+split --train_prop .01 --epochs 25
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-spec-split-small --data ../data
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	python contrastive.py --mode train --task disease --log_dir repeat/supervised-small --data ../data --train_prop .01 --epochs 25 --full_data True
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-small --data ../data
	python contrastive.py --mode train --task disease --log_dir repeat/supervised-raw-small --data ../data --augment raw --train_prop .01 --epochs 25 --full_data True
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-raw-small --data ../data
	python contrastive.py --mode train --task disease --log_dir repeat/supervised-spec-small --data ../data --augment spec --train_prop .01 --epochs 25 --full_data True
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-spec-small --data ../data
	python contrastive.py --mode train --task disease --log_dir repeat/supervised-split-small --data ../data --augment split --train_prop .01 --epochs 25 --full_data True
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-split-small --data ../data
	python contrastive.py --mode train --task disease --log_dir repeat/supervised-spec-split-small --data ../data --augment spec+split --train_prop .01 --epochs 25 --full_data True
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-spec-split-small --data ../data
done

for i in 1 2 3 4 5
do
	python contrastive.py --mode train --task disease --log_dir repeat/supervised-medium --data ../data --train_prop .1 --epochs 25
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-medium --data ../data
	python contrastive.py --mode train --task disease --log_dir repeat/supervised-raw-medium --data ../data --augment raw --train_prop .1 --epochs 25
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-raw-medium --data ../data
	python contrastive.py --mode train --task disease --log_dir repeat/supervised-spec-medium --data ../data --augment spec --train_prop .1 --epochs 25
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-spec-medium --data ../data
	python contrastive.py --mode train --task disease --log_dir repeat/supervised-split-medium --data ../data --augment split --train_prop .1 --epochs 25
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-split-medium --data ../data
	python contrastive.py --mode train --task disease --log_dir repeat/supervised-spec-split-medium --data ../data --augment spec+split --train_prop .1 --epochs 25
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-spec-split-medium --data ../data

	python contrastive.py --mode train --task disease --log_dir repeat/supervised-medium --data ../data --train_prop .1 --epochs 25 --full_data True
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-medium --data ../data
	python contrastive.py --mode train --task disease --log_dir repeat/supervised-raw-medium --data ../data --augment raw --train_prop .1 --epochs 25 --full_data True
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-raw-medium --data ../data
	python contrastive.py --mode train --task disease --log_dir repeat/supervised-spec-medium --data ../data --augment spec --train_prop .1 --epochs 25 --full_data True
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-spec-medium --data ../data
	python contrastive.py --mode train --task disease --log_dir repeat/supervised-split-medium --data ../data --augment split --train_prop .1 --epochs 25 --full_data True
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-split-medium --data ../data
	python contrastive.py --mode train --task disease --log_dir repeat/supervised-spec-split-medium --data ../data --augment spec+split --train_prop .1 --epochs 25 --full_data True
	python contrastive.py --mode test --task disease --log_dir repeat/supervised-spec-split-medium --data ../data
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	python contrastive.py --mode train --task disease --log_dir repeat/raw-pre-lung-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
	python contrastive.py --mode test --task disease --log_dir repeat/raw-pre-lung-large --data ../data --evaluator fine-tune
	python contrastive.py --mode train --task disease --log_dir repeat/spec-pre-lung-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
	python contrastive.py --mode test --task disease --log_dir repeat/spec-pre-lung-large --data ../data --evaluator fine-tune
	python contrastive.py --mode train --task disease --log_dir repeat/split-pre-lung-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
	python contrastive.py --mode test --task disease --log_dir repeat/split-pre-lung-large --data ../data --evaluator fine-tune
	python contrastive.py --mode train --task disease --log_dir repeat/spec-split-pre-lung-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
	python contrastive.py --mode test --task disease --log_dir repeat/spec-split-pre-lung-large --data ../data --evaluator fine-tune

	python contrastive.py --mode train --task disease --log_dir repeat/raw-pre-lung-large --data ../data --evaluator linear --train_prop .01 --epoch 1000
	python contrastive.py --mode test --task disease --log_dir repeat/raw-pre-lung-large --data ../data --evaluator linear
	python contrastive.py --mode train --task disease --log_dir repeat/spec-pre-lung-large --data ../data --evaluator linear --train_prop .01 --epoch 1000
	python contrastive.py --mode test --task disease --log_dir repeat/spec-pre-lung-large --data ../data --evaluator linear
	python contrastive.py --mode train --task disease --log_dir repeat/split-pre-lung-large --data ../data --evaluator linear --train_prop .01 --epoch 1000
	python contrastive.py --mode test --task disease --log_dir repeat/split-pre-lung-large --data ../data --evaluator linear
	python contrastive.py --mode train --task disease --log_dir repeat/spec-split-pre-lung-large --data ../data --evaluator linear --train_prop .01 --epoch 1000
	python contrastive.py --mode test --task disease --log_dir repeat/spec-split-pre-lung-large --data ../data --evaluator linear
done

for i in range 1 2 3 4 5 
do
	python contrastive.py --mode train --task disease --log_dir repeat/raw-pre-lung-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
	python contrastive.py --mode test --task disease --log_dir repeat/raw-pre-lung-large --data ../data --evaluator fine-tune
	python contrastive.py --mode train --task disease --log_dir repeat/spec-pre-lung-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
	python contrastive.py --mode test --task disease --log_dir repeat/spec-pre-lung-large --data ../data --evaluator fine-tune
	python contrastive.py --mode train --task disease --log_dir repeat/split-pre-lung-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
	python contrastive.py --mode test --task disease --log_dir repeat/split-pre-lung-large --data ../data --evaluator fine-tune
	python contrastive.py --mode train --task disease --log_dir repeat/spec-split-pre-lung-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
	python contrastive.py --mode test --task disease --log_dir repeat/spec-split-pre-lung-large --data ../data --evaluator fine-tune

	python contrastive.py --mode train --task disease --log_dir repeat/raw-pre-lung-large --data ../data --evaluator linear --train_prop .1 --epoch 1000
	python contrastive.py --mode test --task disease --log_dir repeat/raw-pre-lung-large --data ../data --evaluator linear
	python contrastive.py --mode train --task disease --log_dir repeat/spec-pre-lung-large --data ../data --evaluator linear --train_prop .1 --epoch 1000
	python contrastive.py --mode test --task disease --log_dir repeat/spec-pre-lung-large --data ../data --evaluator linear
	python contrastive.py --mode train --task disease --log_dir repeat/split-pre-lung-large --data ../data --evaluator linear --train_prop .1 --epoch 1000
	python contrastive.py --mode test --task disease --log_dir repeat/split-pre-lung-large --data ../data --evaluator linear
	python contrastive.py --mode train --task disease --log_dir repeat/spec-split-pre-lung-large --data ../data --evaluator linear --train_prop .1 --epoch 1000
	python contrastive.py --mode test --task disease --log_dir repeat/spec-split-pre-lung-large --data ../data --evaluator linear
done
python contrastive.py --mode train --task disease --log_dir repeat/raw-pre-lung-large --data ../data --evaluator linear --train_prop 1 --epoch 1000
python contrastive.py --mode test --task disease --log_dir repeat/raw-pre-lung-large --data ../data --evaluator linear
python contrastive.py --mode train --task disease --log_dir repeat/spec-pre-lung-large --data ../data --evaluator linear --train_prop 1 --epoch 1000
python contrastive.py --mode test --task disease --log_dir repeat/spec-pre-lung-large --data ../data --evaluator linear
python contrastive.py --mode train --task disease --log_dir repeat/split-pre-lung-large --data ../data --evaluator linear --train_prop 1 --epoch 1000
python contrastive.py --mode test --task disease --log_dir repeat/split-pre-lung-large --data ../data --evaluator linear
python contrastive.py --mode train --task disease --log_dir repeat/spec-split-pre-lung-large --data ../data --evaluator linear --train_prop 1 --epoch 1000
python contrastive.py --mode test --task disease --log_dir repeat/spec-split-pre-lung-large --data ../data --evaluator linear
# done
echo "Done"

