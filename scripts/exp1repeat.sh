#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=144:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="exp 1 1_24 aihc lung"
#SBATCH --output=out/exp1-1_24-%j.out

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
#	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-small --data ../heart --train_prop .01 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-small --data ../heart --evaluator fine-tune
##	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-raw-small --data ../heart --augment raw --train_prop .01 --epochs 25 --evaluator fine-tune
##	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-raw-small --data ../heart --evaluator fine-tune
#	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-spec-small --data ../heart --augment spec --train_prop .01 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-spec-small --data ../heart --evaluator fine-tune
#  python contrastive.py --mode train --task heart --log_dir 1_24/supervised-time-small --data ../heart --augment time --train_prop .01 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-time-small --data ../heart --evaluator fine-tune
#  python contrastive.py --mode train --task heart --log_dir 1_24/supervised-freq-small --data ../heart --augment freq --train_prop .01 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-freq-small --data ../heart --evaluator fine-tune
#	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-split-small --data ../heart --augment split --train_prop .01 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-split-small --data ../heart --evaluator fine-tune
#	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-spec-split-small --data ../heart --augment spec+split --train_prop .01 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-spec-split-small --data ../heart --evaluator fine-tune
#
#	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-small-full --data ../heart --train_prop .01 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-small-full --data ../heart --evaluator fine-tune
##	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-raw-small --data ../heart --augment raw --train_prop .01 --epochs 25 --full_data True --evaluator fine-tune
##	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-raw-small --data ../heart --evaluator fine-tune
#	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-spec-small-full --data ../heart --augment spec --train_prop .01 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-spec-small-full --data ../heart --evaluator fine-tune
#  python contrastive.py --mode train --task heart --log_dir 1_24/supervised-time-small-full --data ../heart --augment time --train_prop .01 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-time-small-full --data ../heart --evaluator fine-tune
#  python contrastive.py --mode train --task heart --log_dir 1_24/supervised-freq-small-full --data ../heart --augment freq --train_prop .01 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-freq-small-full --data ../heart --evaluator fine-tune
#	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-split-small-full --data ../heart --augment split --train_prop .01 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-split-small-full --data ../heart --evaluator fine-tune
#	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-spec-split-small-full --data ../heart --augment spec+split --train_prop .01 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-spec-split-small-full --data ../heart --evaluator fine-tune
#done
#
#for i in 1 2 3 4 5
#do
#	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-medium --data ../heart --train_prop .1 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-medium --data ../heart --evaluator fine-tune
##	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-raw-medium --data ../heart --augment raw --train_prop .1 --epochs 25 --evaluator fine-tune
##	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-raw-medium --data ../heart --evaluator fine-tune
#	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-spec-medium --data ../heart --augment spec --train_prop .1 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-spec-medium --data ../heart --evaluator fine-tune
#  python contrastive.py --mode train --task heart --log_dir 1_24/supervised-time-medium --data ../heart --augment time --train_prop .1 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-time-medium --data ../heart --evaluator fine-tune
#	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-freq-medium --data ../heart --augment freq --train_prop .1 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-freq-medium --data ../heart --evaluator fine-tune
#	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-split-medium --data ../heart --augment split --train_prop .1 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-split-medium --data ../heart --evaluator fine-tune
#	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-spec-split-medium --data ../heart --augment spec+split --train_prop .1 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-spec-split-medium --data ../heart --evaluator fine-tune
#
#	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-medium-full --data ../heart --train_prop .1 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-medium-full --data ../heart --evaluator fine-tune
##	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-raw-medium-full --data ../heart --augment raw --train_prop .1 --epochs 25 --full_data True --evaluator fine-tune
##	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-raw-medium-full --data ../heart --evaluator fine-tune
#	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-spec-medium-full --data ../heart --augment spec --train_prop .1 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-spec-medium-full --data ../heart --evaluator fine-tune
#  python contrastive.py --mode train --task heart --log_dir 1_24/supervised-time-medium-full --data ../heart --augment time --train_prop .1 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-time-medium-full --data ../heart --evaluator fine-tune
#  python contrastive.py --mode train --task heart --log_dir 1_24/supervised-freq-medium-full --data ../heart --augment freq --train_prop .1 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-freq-medium-full --data ../heart --evaluator fine-tune
#	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-split-medium-full --data ../heart --augment split --train_prop .1 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-split-medium-full --data ../heart --evaluator fine-tune
#	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-spec-split-medium-full --data ../heart --augment spec+split --train_prop .1 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-spec-split-medium-full --data ../heart --evaluator fine-tune
#done

#python contrastive.py --mode train --task heart --log_dir 1_24/supervised-large --data ../heart --train_prop 1 --epochs 25 --evaluator fine-tune
#python contrastive.py --mode test --task heart --log_dir 1_24/supervised-large --data ../heart --evaluator fine-tune
##	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-raw-medium --data ../heart --augment raw --train_prop 1 --epochs 25 --evaluator fine-tune
##	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-raw-large --data ../heart --evaluator fine-tune
#python contrastive.py --mode train --task heart --log_dir 1_24/supervised-spec-large --data ../heart --augment spec --train_prop 1 --epochs 25 --evaluator fine-tune
#python contrastive.py --mode test --task heart --log_dir 1_24/supervised-spec-large --data ../heart --evaluator fine-tune
#python contrastive.py --mode train --task heart --log_dir 1_24/supervised-time-large --data ../heart --augment time --train_prop 1 --epochs 25 --evaluator fine-tune
#python contrastive.py --mode test --task heart --log_dir 1_24/supervised-time-large --data ../heart --evaluator fine-tune
#python contrastive.py --mode train --task heart --log_dir 1_24/supervised-freq-large --data ../heart --augment freq --train_prop 1 --epochs 25 --evaluator fine-tune
#python contrastive.py --mode test --task heart --log_dir 1_24/supervised-freq-large --data ../heart --evaluator fine-tune
#python contrastive.py --mode train --task heart --log_dir 1_24/supervised-split-large --data ../heart --augment split --train_prop 1 --epochs 25 --evaluator fine-tune
#python contrastive.py --mode test --task heart --log_dir 1_24/supervised-split-large --data ../heart --evaluator fine-tune
#python contrastive.py --mode train --task heart --log_dir 1_24/supervised-spec-split-large --data ../heart --augment spec+split --train_prop 1 --epochs 25 --evaluator fine-tune
#python contrastive.py --mode test --task heart --log_dir 1_24/supervised-spec-split-large --data ../heart --evaluator fine-tune
#
#python contrastive.py --mode train --task heart --log_dir 1_24/supervised-large-full --data ../heart --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#python contrastive.py --mode test --task heart --log_dir 1_24/supervised-large-full --data ../heart --evaluator fine-tune
##	python contrastive.py --mode train --task heart --log_dir 1_24/supervised-raw-large-full --data ../heart --augment raw --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
##	python contrastive.py --mode test --task heart --log_dir 1_24/supervised-raw-large-full --data ../heart --evaluator fine-tune
#python contrastive.py --mode train --task heart --log_dir 1_24/supervised-spec-large-full --data ../heart --augment spec --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#python contrastive.py --mode test --task heart --log_dir 1_24/supervised-spec-large-full --data ../heart --evaluator fine-tune
#python contrastive.py --mode train --task heart --log_dir 1_24/supervised-time-large-full --data ../heart --augment time --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#python contrastive.py --mode test --task heart --log_dir 1_24/supervised-time-large-full --data ../heart --evaluator fine-tune
#python contrastive.py --mode train --task heart --log_dir 1_24/supervised-freq-large-full --data ../heart --augment freq --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#python contrastive.py --mode test --task heart --log_dir 1_24/supervised-freq-large-full --data ../heart --evaluator fine-tune
#python contrastive.py --mode train --task heart --log_dir 1_24/supervised-split-large-full --data ../heart --augment split --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#python contrastive.py --mode test --task heart --log_dir 1_24/supervised-split-large-full --data ../heart --evaluator fine-tune
#python contrastive.py --mode train --task heart --log_dir 1_24/supervised-spec-split-large-full --data ../heart --augment spec+split --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#python contrastive.py --mode test --task heart --log_dir 1_24/supervised-spec-split-large-full --data ../heart --evaluator fine-tune

python contrastive.py --mode pretrain --task heart --log_dir 1_24/spec-pre-large --data ../heart --augment spec --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task heart --log_dir 1_24/split-pre-large --data ../heart --augment split --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task heart --log_dir 1_24/spec-split-pre-large --data ../heart --augment spec+split --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task heart --log_dir 1_24/time-pre-large --data ../heart --augment time --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task heart --log_dir 1_24/freq-pre-large --data ../heart --augment freq --train_prop 1.0 --epoch 10

for i in 1 2 3 4 5 6 7 8 9 10
do
#	python contrastive.py --mode train --task heart --log_dir 1_24/raw-pre-large --data ../heart --evaluator fine-tune --train_prop .01 --epoch 25
#	python contrastive.py --mode test --task heart --log_dir 1_24/raw-pre-large --data ../heart --evaluator fine-tune
	python contrastive.py --mode train --task heart --log_dir 1_24/spec-pre-large --data ../heart --evaluator fine-tune --train_prop .01 --epoch 25
	python contrastive.py --mode test --task heart --log_dir 1_24/spec-pre-large --data ../heart --evaluator fine-tune
	python contrastive.py --mode train --task heart --log_dir 1_24/split-pre-large --data ../heart --evaluator fine-tune --train_prop .01 --epoch 25
	python contrastive.py --mode test --task heart --log_dir 1_24/split-pre-large --data ../heart --evaluator fine-tune
	python contrastive.py --mode train --task heart --log_dir 1_24/spec-split-pre-large --data ../heart --evaluator fine-tune --train_prop .01 --epoch 25
	python contrastive.py --mode test --task heart --log_dir 1_24/spec-split-pre-large --data ../heart --evaluator fine-tune
	python contrastive.py --mode train --task heart --log_dir 1_24/time-pre-large --data ../heart --evaluator fine-tune --train_prop .01 --epoch 25
	python contrastive.py --mode test --task heart --log_dir 1_24/time-pre-large --data ../heart --evaluator fine-tune
  python contrastive.py --mode train --task heart --log_dir 1_24/freq-pre-large --data ../heart --evaluator fine-tune --train_prop .01 --epoch 25
	python contrastive.py --mode test --task heart --log_dir 1_24/freq-pre-large --data ../heart --evaluator fine-tune

#	python contrastive.py --mode train --task heart --log_dir 1_24/raw-pre-large --data ../heart --evaluator linear --train_prop .01 --epoch 1000
#	python contrastive.py --mode test --task heart --log_dir 1_24/raw-pre-large --data ../heart --evaluator linear
	python contrastive.py --mode train --task heart --log_dir 1_24/spec-pre-large --data ../heart --evaluator linear --train_prop .01 --epoch 1000
	python contrastive.py --mode test --task heart --log_dir 1_24/spec-pre-large --data ../heart --evaluator linear
	python contrastive.py --mode train --task heart --log_dir 1_24/split-pre-large --data ../heart --evaluator linear --train_prop .01 --epoch 1000
	python contrastive.py --mode test --task heart --log_dir 1_24/split-pre-large --data ../heart --evaluator linear
	python contrastive.py --mode train --task heart --log_dir 1_24/spec-split-pre-large --data ../heart --evaluator linear --train_prop .01 --epoch 1000
	python contrastive.py --mode test --task heart --log_dir 1_24/spec-split-pre-large --data ../heart --evaluator linear
  python contrastive.py --mode train --task heart --log_dir 1_24/time-pre-large --data ../heart --evaluator linear --train_prop .01 --epoch 1000
	python contrastive.py --mode test --task heart --log_dir 1_24/time-pre-large --data ../heart --evaluator linear
  python contrastive.py --mode train --task heart --log_dir 1_24/freq-pre-large --data ../heart --evaluator linear --train_prop .01 --epoch 1000
	python contrastive.py --mode test --task heart --log_dir 1_24/freq-pre-large --data ../heart --evaluator linear

done

for i in range 1 2 3 4 5
do
#	python contrastive.py --mode train --task heart --log_dir 1_24/raw-pre-large --data ../heart --evaluator fine-tune --train_prop .1 --epoch 25
#	python contrastive.py --mode test --task heart --log_dir 1_24/raw-pre-large --data ../heart --evaluator fine-tune
	python contrastive.py --mode train --task heart --log_dir 1_24/spec-pre-large --data ../heart --evaluator fine-tune --train_prop .1 --epoch 25
	python contrastive.py --mode test --task heart --log_dir 1_24/spec-pre-large --data ../heart --evaluator fine-tune
	python contrastive.py --mode train --task heart --log_dir 1_24/split-pre-large --data ../heart --evaluator fine-tune --train_prop .1 --epoch 25
	python contrastive.py --mode test --task heart --log_dir 1_24/split-pre-large --data ../heart --evaluator fine-tune
	python contrastive.py --mode train --task heart --log_dir 1_24/spec-split-pre-large --data ../heart --evaluator fine-tune --train_prop .1 --epoch 25
	python contrastive.py --mode test --task heart --log_dir 1_24/spec-split-pre-large --data ../heart --evaluator fine-tune
	python contrastive.py --mode train --task heart --log_dir 1_24/freq-pre-large --data ../heart --evaluator fine-tune --train_prop .1 --epoch 25
	python contrastive.py --mode test --task heart --log_dir 1_24/freq-pre-large --data ../heart --evaluator fine-tune
  python contrastive.py --mode train --task heart --log_dir 1_24/time-pre-large --data ../heart --evaluator fine-tune --train_prop .1 --epoch 25
	python contrastive.py --mode test --task heart --log_dir 1_24/time-pre-large --data ../heart --evaluator fine-tune

#	python contrastive.py --mode train --task heart --log_dir 1_24/raw-pre-large --data ../heart --evaluator linear --train_prop .1 --epoch 1000
#	python contrastive.py --mode test --task heart --log_dir 1_24/raw-pre-large --data ../heart --evaluator linear
	python contrastive.py --mode train --task heart --log_dir 1_24/spec-pre-large --data ../heart --evaluator linear --train_prop .1 --epoch 1000
	python contrastive.py --mode test --task heart --log_dir 1_24/spec-pre-large --data ../heart --evaluator linear
	python contrastive.py --mode train --task heart --log_dir 1_24/split-pre-large --data ../heart --evaluator linear --train_prop .1 --epoch 1000
	python contrastive.py --mode test --task heart --log_dir 1_24/split-pre-large --data ../heart --evaluator linear
	python contrastive.py --mode train --task heart --log_dir 1_24/spec-split-pre-large --data ../heart --evaluator linear --train_prop .1 --epoch 1000
	python contrastive.py --mode test --task heart --log_dir 1_24/spec-split-pre-large --data ../heart --evaluator linear
  python contrastive.py --mode train --task heart --log_dir 1_24/freq-pre-large --data ../heart --evaluator linear --train_prop .1 --epoch 1000
	python contrastive.py --mode test --task heart --log_dir 1_24/freq-pre-large --data ../heart --evaluator linear
  python contrastive.py --mode train --task heart --log_dir 1_24/time-pre-large --data ../heart --evaluator linear --train_prop .1 --epoch 1000
	python contrastive.py --mode test --task heart --log_dir 1_24/time-pre-large --data ../heart --evaluator linear
done

#python contrastive.py --mode train --task heart --log_dir 1_24/raw-pre-large --data ../heart --evaluator fine-tune --train_prop 1 --epoch 25
#python contrastive.py --mode test --task heart --log_dir 1_24/raw-pre-large --data ../heart --evaluator fine-tune
python contrastive.py --mode train --task heart --log_dir 1_24/spec-pre-large --data ../heart --evaluator fine-tune --train_prop 1 --epoch 25
python contrastive.py --mode test --task heart --log_dir 1_24/spec-pre-large --data ../heart --evaluator fine-tune
python contrastive.py --mode train --task heart --log_dir 1_24/split-pre-large --data ../heart --evaluator fine-tune --train_prop 1 --epoch 25
python contrastive.py --mode test --task heart --log_dir 1_24/split-pre-large --data ../heart --evaluator fine-tune
python contrastive.py --mode train --task heart --log_dir 1_24/spec-split-pre-large --data ../heart --evaluator fine-tune --train_prop 1 --epoch 25
python contrastive.py --mode test --task heart --log_dir 1_24/spec-split-pre-large --data ../heart --evaluator fine-tune
python contrastive.py --mode train --task heart --log_dir 1_24/freq-pre-large --data ../heart --evaluator fine-tune --train_prop 1 --epoch 25
python contrastive.py --mode test --task heart --log_dir 1_24/freq-pre-large --data ../heart --evaluator fine-tune
python contrastive.py --mode train --task heart --log_dir 1_24/time-pre-large --data ../heart --evaluator fine-tune --train_prop 1 --epoch 25
python contrastive.py --mode test --task heart --log_dir 1_24/time-pre-large --data ../heart --evaluator fine-tune

#python contrastive.py --mode train --task heart --log_dir 1_24/raw-pre-large --data ../heart --evaluator linear --train_prop 1 --epoch 1000
#python contrastive.py --mode test --task heart --log_dir 1_24/raw-pre-large --data ../heart --evaluator linear
python contrastive.py --mode train --task heart --log_dir 1_24/spec-pre-large --data ../heart --evaluator linear --train_prop 1 --epoch 1000
python contrastive.py --mode test --task heart --log_dir 1_24/spec-pre-large --data ../heart --evaluator linear
python contrastive.py --mode train --task heart --log_dir 1_24/split-pre-large --data ../heart --evaluator linear --train_prop 1 --epoch 1000
python contrastive.py --mode test --task heart --log_dir 1_24/split-pre-large --data ../heart --evaluator linear
python contrastive.py --mode train --task heart --log_dir 1_24/spec-split-pre-large --data ../heart --evaluator linear --train_prop 1 --epoch 1000
python contrastive.py --mode test --task heart --log_dir 1_24/spec-split-pre-large --data ../heart --evaluator linear
python contrastive.py --mode train --task heart --log_dir 1_24/freq-pre-large --data ../heart --evaluator linear --train_prop 1 --epoch 1000
python contrastive.py --mode test --task heart --log_dir 1_24/freq-pre-large --data ../heart --evaluator linear
python contrastive.py --mode train --task heart --log_dir 1_24/time-pre-large --data ../heart --evaluator linear --train_prop 1 --epoch 1000
python contrastive.py --mode test --task heart --log_dir 1_24/time-pre-large --data ../heart --evaluator linear

# done
echo "Done"

