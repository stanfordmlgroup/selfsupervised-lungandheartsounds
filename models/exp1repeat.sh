#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=144:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="exp 1 repeat aihc lung"
#SBATCH --output=out/exp1-repeat-%j.out

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
source ~/.bashrc
conda activate lungsounds

#for i in 1 2 3 4 5 6 7 8 9 10
#do
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-small --data ../heart --train_prop .01 --epochs 25
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-small --data ../heart
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-raw-small --data ../heart --augment raw --train_prop .01 --epochs 25
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-raw-small --data ../heart
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-spec-small --data ../heart --augment spec --train_prop .01 --epochs 25
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-spec-small --data ../heart
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-split-small --data ../heart --augment split --train_prop .01 --epochs 25
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-split-small --data ../heart
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-spec-split-small --data ../heart --augment spec+split --train_prop .01 --epochs 25
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-spec-split-small --data ../heart
#done
#
#for i in 1 2 3 4 5 6 7 8 9 10
#do
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-small --data ../heart --train_prop .01 --epochs 25 --full_data True
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-small --data ../heart
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-raw-small --data ../heart --augment raw --train_prop .01 --epochs 25 --full_data True
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-raw-small --data ../heart
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-spec-small --data ../heart --augment spec --train_prop .01 --epochs 25 --full_data True
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-spec-small --data ../heart
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-split-small --data ../heart --augment split --train_prop .01 --epochs 25 --full_data True
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-split-small --data ../heart
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-spec-split-small --data ../heart --augment spec+split --train_prop .01 --epochs 25 --full_data True
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-spec-split-small --data ../heart
#done
#
#for i in 1 2 3 4 5
#do
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-medium --data ../heart --train_prop .1 --epochs 25
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-medium --data ../heart
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-raw-medium --data ../heart --augment raw --train_prop .1 --epochs 25
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-raw-medium --data ../heart
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-spec-medium --data ../heart --augment spec --train_prop .1 --epochs 25
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-spec-medium --data ../heart
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-split-medium --data ../heart --augment split --train_prop .1 --epochs 25
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-split-medium --data ../heart
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-spec-split-medium --data ../heart --augment spec+split --train_prop .1 --epochs 25
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-spec-split-medium --data ../heart
#
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-medium --data ../heart --train_prop .1 --epochs 25 --full_data True
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-medium --data ../heart
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-raw-medium --data ../heart --augment raw --train_prop .1 --epochs 25 --full_data True
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-raw-medium --data ../heart
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-spec-medium --data ../heart --augment spec --train_prop .1 --epochs 25 --full_data True
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-spec-medium --data ../heart
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-split-medium --data ../heart --augment split --train_prop .1 --epochs 25 --full_data True
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-split-medium --data ../heart
#	python contrastive.py --mode train --task heart --log_dir repeat/supervised-spec-split-medium --data ../heart --augment spec+split --train_prop .1 --epochs 25 --full_data True
#	python contrastive.py --mode test --task heart --log_dir repeat/supervised-spec-split-medium --data ../heart
#done

for i in 1 2 3 4 5 6 7 8 9 10
do
	python contrastive.py --mode train --task heart --log_dir repeat/raw-pre-large --data ../heart --evaluator fine-tune --train_prop .01 --epoch 25
	python contrastive.py --mode test --task heart --log_dir repeat/raw-pre-large --data ../heart --evaluator fine-tune
	python contrastive.py --mode train --task heart --log_dir repeat/spec-pre-large --data ../heart --evaluator fine-tune --train_prop .01 --epoch 25
	python contrastive.py --mode test --task heart --log_dir repeat/spec-pre-large --data ../heart --evaluator fine-tune
	python contrastive.py --mode train --task heart --log_dir repeat/split-pre-large --data ../heart --evaluator fine-tune --train_prop .01 --epoch 25
	python contrastive.py --mode test --task heart --log_dir repeat/split-pre-large --data ../heart --evaluator fine-tune
	python contrastive.py --mode train --task heart --log_dir repeat/spec-split-pre-large --data ../heart --evaluator fine-tune --train_prop .01 --epoch 25
	python contrastive.py --mode test --task heart --log_dir repeat/spec-split-pre-large --data ../heart --evaluator fine-tune

	python contrastive.py --mode train --task heart --log_dir repeat/raw-pre-large --data ../heart --evaluator linear --train_prop .01 --epoch 1000
	python contrastive.py --mode test --task heart --log_dir repeat/raw-pre-large --data ../heart --evaluator linear
	python contrastive.py --mode train --task heart --log_dir repeat/spec-pre-large --data ../heart --evaluator linear --train_prop .01 --epoch 1000
	python contrastive.py --mode test --task heart --log_dir repeat/spec-pre-large --data ../heart --evaluator linear
	python contrastive.py --mode train --task heart --log_dir repeat/split-pre-large --data ../heart --evaluator linear --train_prop .01 --epoch 1000
	python contrastive.py --mode test --task heart --log_dir repeat/split-pre-large --data ../heart --evaluator linear
	python contrastive.py --mode train --task heart --log_dir repeat/spec-split-pre-large --data ../heart --evaluator linear --train_prop .01 --epoch 1000
	python contrastive.py --mode test --task heart --log_dir repeat/spec-split-pre-large --data ../heart --evaluator linear
done

for i in range 1 2 3 4 5 
do
	python contrastive.py --mode train --task heart --log_dir repeat/raw-pre-large --data ../heart --evaluator fine-tune --train_prop .1 --epoch 25
	python contrastive.py --mode test --task heart --log_dir repeat/raw-pre-large --data ../heart --evaluator fine-tune
	python contrastive.py --mode train --task heart --log_dir repeat/spec-pre-large --data ../heart --evaluator fine-tune --train_prop .1 --epoch 25
	python contrastive.py --mode test --task heart --log_dir repeat/spec-pre-large --data ../heart --evaluator fine-tune
	python contrastive.py --mode train --task heart --log_dir repeat/split-pre-large --data ../heart --evaluator fine-tune --train_prop .1 --epoch 25
	python contrastive.py --mode test --task heart --log_dir repeat/split-pre-large --data ../heart --evaluator fine-tune
	python contrastive.py --mode train --task heart --log_dir repeat/spec-split-pre-large --data ../heart --evaluator fine-tune --train_prop .1 --epoch 25
	python contrastive.py --mode test --task heart --log_dir repeat/spec-split-pre-large --data ../heart --evaluator fine-tune

	python contrastive.py --mode train --task heart --log_dir repeat/raw-pre-large --data ../heart --evaluator linear --train_prop .1 --epoch 1000
	python contrastive.py --mode test --task heart --log_dir repeat/raw-pre-large --data ../heart --evaluator linear
	python contrastive.py --mode train --task heart --log_dir repeat/spec-pre-large --data ../heart --evaluator linear --train_prop .1 --epoch 1000
	python contrastive.py --mode test --task heart --log_dir repeat/spec-pre-large --data ../heart --evaluator linear
	python contrastive.py --mode train --task heart --log_dir repeat/split-pre-large --data ../heart --evaluator linear --train_prop .1 --epoch 1000
	python contrastive.py --mode test --task heart --log_dir repeat/split-pre-large --data ../heart --evaluator linear
	python contrastive.py --mode train --task heart --log_dir repeat/spec-split-pre-large --data ../heart --evaluator linear --train_prop .1 --epoch 1000
	python contrastive.py --mode test --task heart --log_dir repeat/spec-split-pre-large --data ../heart --evaluator linear
done
python contrastive.py --mode train --task heart --log_dir repeat/raw-pre-large --data ../heart --evaluator linear --train_prop 1 --epoch 1000
python contrastive.py --mode test --task heart --log_dir repeat/raw-pre-large --data ../heart --evaluator linear
python contrastive.py --mode train --task heart --log_dir repeat/spec-pre-large --data ../heart --evaluator linear --train_prop 1 --epoch 1000
python contrastive.py --mode test --task heart --log_dir repeat/spec-pre-large --data ../heart --evaluator linear
python contrastive.py --mode train --task heart --log_dir repeat/split-pre-large --data ../heart --evaluator linear --train_prop 1 --epoch 1000
python contrastive.py --mode test --task heart --log_dir repeat/split-pre-large --data ../heart --evaluator linear
python contrastive.py --mode train --task heart --log_dir repeat/spec-split-pre-large --data ../heart --evaluator linear --train_prop 1 --epoch 1000
python contrastive.py --mode test --task heart --log_dir repeat/spec-split-pre-large --data ../heart --evaluator linear
# done
echo "Done"

