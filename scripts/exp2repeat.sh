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
cd ../models
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-small --data ../data --train_prop .01 --epochs 25 --evaluator fine-tune
	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-small --data ../data --evaluator fine-tune
#	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-raw-small --data ../data --augment raw --train_prop .01 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-raw-small --data ../data --evaluator fine-tune
#	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-spec-small --data ../data --augment spec --train_prop .01 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-spec-small --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task disease --log_dir 2_1/supervised-time-small --data ../data --augment time --train_prop .01 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-time-small --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task disease --log_dir 2_1/supervised-freq-small --data ../data --augment freq --train_prop .01 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-freq-small --data ../data --evaluator fine-tune
#	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-split-small --data ../data --augment split --train_prop .01 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-split-small --data ../data --evaluator fine-tune
#	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-spec-split-small --data ../data --augment spec+split --train_prop .01 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-spec-split-small --data ../data --evaluator fine-tune

	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-small-full --data ../data --train_prop .01 --epochs 25 --full_data True --evaluator fine-tune
	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-small-full --data ../data --evaluator fine-tune
#	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-raw-small --data ../data --augment raw --train_prop .01 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-raw-small --data ../data --evaluator fine-tune
#	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-spec-small-full --data ../data --augment spec --train_prop .01 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-spec-small-full --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task disease --log_dir 2_1/supervised-time-small-full --data ../data --augment time --train_prop .01 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-time-small-full --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task disease --log_dir 2_1/supervised-freq-small-full --data ../data --augment freq --train_prop .01 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-freq-small-full --data ../data --evaluator fine-tune
#	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-split-small-full --data ../data --augment split --train_prop .01 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-split-small-full --data ../data --evaluator fine-tune
#	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-spec-split-small-full --data ../data --augment spec+split --train_prop .01 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-spec-split-small-full --data ../data --evaluator fine-tune
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-medium --data ../data --train_prop .1 --epochs 25 --evaluator fine-tune
	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-medium --data ../data --evaluator fine-tune
#	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-raw-medium --data ../data --augment raw --train_prop .1 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-raw-medium --data ../data --evaluator fine-tune
#	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-spec-medium --data ../data --augment spec --train_prop .1 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-spec-medium --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task disease --log_dir 2_1/supervised-time-medium --data ../data --augment time --train_prop .1 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-time-medium --data ../data --evaluator fine-tune
#	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-freq-medium --data ../data --augment freq --train_prop .1 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-freq-medium --data ../data --evaluator fine-tune
#	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-split-medium --data ../data --augment split --train_prop .1 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-split-medium --data ../data --evaluator fine-tune
#	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-spec-split-medium --data ../data --augment spec+split --train_prop .1 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-spec-split-medium --data ../data --evaluator fine-tune

	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-medium-full --data ../data --train_prop .1 --epochs 25 --full_data True --evaluator fine-tune
	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-medium-full --data ../data --evaluator fine-tune
#	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-raw-medium-full --data ../data --augment raw --train_prop .1 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-raw-medium-full --data ../data --evaluator fine-tune
#	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-spec-medium-full --data ../data --augment spec --train_prop .1 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-spec-medium-full --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task disease --log_dir 2_1/supervised-time-medium-full --data ../data --augment time --train_prop .1 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-time-medium-full --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task disease --log_dir 2_1/supervised-freq-medium-full --data ../data --augment freq --train_prop .1 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-freq-medium-full --data ../data --evaluator fine-tune
#	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-split-medium-full --data ../data --augment split --train_prop .1 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-split-medium-full --data ../data --evaluator fine-tune
#	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-spec-split-medium-full --data ../data --augment spec+split --train_prop .1 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-spec-split-medium-full --data ../data --evaluator fine-tune
done

for i in 1 2 3 4 5
do
  python contrastive.py --mode train --task disease --log_dir 2_1/supervised-large --data ../data --train_prop 1 --epochs 25 --evaluator fine-tune
  python contrastive.py --mode test --task disease --log_dir 2_1/supervised-large --data ../data --evaluator fine-tune
  #	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-raw-medium --data ../data --augment raw --train_prop 1 --epochs 25 --evaluator fine-tune
  #	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-raw-large --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task disease --log_dir 2_1/supervised-spec-large --data ../data --augment spec --train_prop 1 --epochs 25 --evaluator fine-tune
#  python contrastive.py --mode test --task disease --log_dir 2_1/supervised-spec-large --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task disease --log_dir 2_1/supervised-time-large --data ../data --augment time --train_prop 1 --epochs 25 --evaluator fine-tune
#  python contrastive.py --mode test --task disease --log_dir 2_1/supervised-time-large --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task disease --log_dir 2_1/supervised-freq-large --data ../data --augment freq --train_prop 1 --epochs 25 --evaluator fine-tune
#  python contrastive.py --mode test --task disease --log_dir 2_1/supervised-freq-large --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task disease --log_dir 2_1/supervised-split-large --data ../data --augment split --train_prop 1 --epochs 25 --evaluator fine-tune
#  python contrastive.py --mode test --task disease --log_dir 2_1/supervised-split-large --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task disease --log_dir 2_1/supervised-spec-split-large --data ../data --augment spec+split --train_prop 1 --epochs 25 --evaluator fine-tune
#  python contrastive.py --mode test --task disease --log_dir 2_1/supervised-spec-split-large --data ../data --evaluator fine-tune
  
  python contrastive.py --mode train --task disease --log_dir 2_1/supervised-large-full --data ../data --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
  python contrastive.py --mode test --task disease --log_dir 2_1/supervised-large-full --data ../data --evaluator fine-tune
  #	python contrastive.py --mode train --task disease --log_dir 2_1/supervised-raw-large-full --data ../data --augment raw --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
  #	python contrastive.py --mode test --task disease --log_dir 2_1/supervised-raw-large-full --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task disease --log_dir 2_1/supervised-spec-large-full --data ../data --augment spec --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#  python contrastive.py --mode test --task disease --log_dir 2_1/supervised-spec-large-full --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task disease --log_dir 2_1/supervised-time-large-full --data ../data --augment time --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#  python contrastive.py --mode test --task disease --log_dir 2_1/supervised-time-large-full --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task disease --log_dir 2_1/supervised-freq-large-full --data ../data --augment freq --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#  python contrastive.py --mode test --task disease --log_dir 2_1/supervised-freq-large-full --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task disease --log_dir 2_1/supervised-split-large-full --data ../data --augment split --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#  python contrastive.py --mode test --task disease --log_dir 2_1/supervised-split-large-full --data ../data --evaluator fine-tune
#  python contrastive.py --mode train --task disease --log_dir 2_1/supervised-spec-split-large-full --data ../data --augment spec+split --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#  python contrastive.py --mode test --task disease --log_dir 2_1/supervised-spec-split-large-full --data ../data --evaluator fine-tune
done
  
python contrastive.py --mode pretrain --task disease --log_dir 2_1/spec-pre-large --data ../data --augment spec --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task disease --log_dir 2_1/split-pre-large --data ../data --augment split --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task disease --log_dir 2_1/spec-split-pre-large --data ../data --augment spec+split --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task disease --log_dir 2_1/time-pre-large --data ../data --augment time --train_prop 1.0 --epoch 10
python contrastive.py --mode pretrain --task disease --log_dir 2_1/freq-pre-large --data ../data --augment freq --train_prop 1.0 --epoch 10

for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
#	python contrastive.py --mode train --task disease --log_dir 2_1/raw-pre-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
#	python contrastive.py --mode test --task disease --log_dir 2_1/raw-pre-large --data ../data --evaluator fine-tune
	python contrastive.py --mode train --task disease --log_dir 2_1/spec-pre-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
	python contrastive.py --mode test --task disease --log_dir 2_1/spec-pre-large --data ../data --evaluator fine-tune
	python contrastive.py --mode train --task disease --log_dir 2_1/split-pre-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
	python contrastive.py --mode test --task disease --log_dir 2_1/split-pre-large --data ../data --evaluator fine-tune
	python contrastive.py --mode train --task disease --log_dir 2_1/spec-split-pre-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
	python contrastive.py --mode test --task disease --log_dir 2_1/spec-split-pre-large --data ../data --evaluator fine-tune
	python contrastive.py --mode train --task disease --log_dir 2_1/time-pre-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
	python contrastive.py --mode test --task disease --log_dir 2_1/time-pre-large --data ../data --evaluator fine-tune
  python contrastive.py --mode train --task disease --log_dir 2_1/freq-pre-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
	python contrastive.py --mode test --task disease --log_dir 2_1/freq-pre-large --data ../data --evaluator fine-tune

#	python contrastive.py --mode train --task disease --log_dir 2_1/raw-pre-large --data ../data --evaluator linear --train_prop .01 --epoch 1000
#	python contrastive.py --mode test --task disease --log_dir 2_1/raw-pre-large --data ../data --evaluator linear
	python contrastive.py --mode train --task disease --log_dir 2_1/spec-pre-large --data ../data --evaluator linear --train_prop .01 --epoch 1000
	python contrastive.py --mode test --task disease --log_dir 2_1/spec-pre-large --data ../data --evaluator linear
	python contrastive.py --mode train --task disease --log_dir 2_1/split-pre-large --data ../data --evaluator linear --train_prop .01 --epoch 1000
	python contrastive.py --mode test --task disease --log_dir 2_1/split-pre-large --data ../data --evaluator linear
	python contrastive.py --mode train --task disease --log_dir 2_1/spec-split-pre-large --data ../data --evaluator linear --train_prop .01 --epoch 1000
	python contrastive.py --mode test --task disease --log_dir 2_1/spec-split-pre-large --data ../data --evaluator linear
  python contrastive.py --mode train --task disease --log_dir 2_1/time-pre-large --data ../data --evaluator linear --train_prop .01 --epoch 1000
	python contrastive.py --mode test --task disease --log_dir 2_1/time-pre-large --data ../data --evaluator linear
  python contrastive.py --mode train --task disease --log_dir 2_1/freq-pre-large --data ../data --evaluator linear --train_prop .01 --epoch 1000
	python contrastive.py --mode test --task disease --log_dir 2_1/freq-pre-large --data ../data --evaluator linear

done

for i in 1 2 3 4 5 6 7 8 9 10
do
#	python contrastive.py --mode train --task disease --log_dir 2_1/raw-pre-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
#	python contrastive.py --mode test --task disease --log_dir 2_1/raw-pre-large --data ../data --evaluator fine-tune
	python contrastive.py --mode train --task disease --log_dir 2_1/spec-pre-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
	python contrastive.py --mode test --task disease --log_dir 2_1/spec-pre-large --data ../data --evaluator fine-tune
	python contrastive.py --mode train --task disease --log_dir 2_1/split-pre-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
	python contrastive.py --mode test --task disease --log_dir 2_1/split-pre-large --data ../data --evaluator fine-tune
	python contrastive.py --mode train --task disease --log_dir 2_1/spec-split-pre-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
	python contrastive.py --mode test --task disease --log_dir 2_1/spec-split-pre-large --data ../data --evaluator fine-tune
	python contrastive.py --mode train --task disease --log_dir 2_1/freq-pre-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
	python contrastive.py --mode test --task disease --log_dir 2_1/freq-pre-large --data ../data --evaluator fine-tune
  python contrastive.py --mode train --task disease --log_dir 2_1/time-pre-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
	python contrastive.py --mode test --task disease --log_dir 2_1/time-pre-large --data ../data --evaluator fine-tune

#	python contrastive.py --mode train --task disease --log_dir 2_1/raw-pre-large --data ../data --evaluator linear --train_prop .1 --epoch 1000
#	python contrastive.py --mode test --task disease --log_dir 2_1/raw-pre-large --data ../data --evaluator linear
	python contrastive.py --mode train --task disease --log_dir 2_1/spec-pre-large --data ../data --evaluator linear --train_prop .1 --epoch 1000
	python contrastive.py --mode test --task disease --log_dir 2_1/spec-pre-large --data ../data --evaluator linear
	python contrastive.py --mode train --task disease --log_dir 2_1/split-pre-large --data ../data --evaluator linear --train_prop .1 --epoch 1000
	python contrastive.py --mode test --task disease --log_dir 2_1/split-pre-large --data ../data --evaluator linear
	python contrastive.py --mode train --task disease --log_dir 2_1/spec-split-pre-large --data ../data --evaluator linear --train_prop .1 --epoch 1000
	python contrastive.py --mode test --task disease --log_dir 2_1/spec-split-pre-large --data ../data --evaluator linear
  python contrastive.py --mode train --task disease --log_dir 2_1/freq-pre-large --data ../data --evaluator linear --train_prop .1 --epoch 1000
	python contrastive.py --mode test --task disease --log_dir 2_1/freq-pre-large --data ../data --evaluator linear
  python contrastive.py --mode train --task disease --log_dir 2_1/time-pre-large --data ../data --evaluator linear --train_prop .1 --epoch 1000
	python contrastive.py --mode test --task disease --log_dir 2_1/time-pre-large --data ../data --evaluator linear
done

for i in 1 2 3 4 5
do
  #python contrastive.py --mode train --task disease --log_dir 2_1/raw-pre-large --data ../data --evaluator fine-tune --train_prop 1 --epoch 25
  #python contrastive.py --mode test --task disease --log_dir 2_1/raw-pre-large --data ../data --evaluator fine-tune
  python contrastive.py --mode train --task disease --log_dir 2_1/spec-pre-large --data ../data --evaluator fine-tune --train_prop 1 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-pre-large --data ../data --evaluator fine-tune
  python contrastive.py --mode train --task disease --log_dir 2_1/split-pre-large --data ../data --evaluator fine-tune --train_prop 1 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/split-pre-large --data ../data --evaluator fine-tune
  python contrastive.py --mode train --task disease --log_dir 2_1/spec-split-pre-large --data ../data --evaluator fine-tune --train_prop 1 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-split-pre-large --data ../data --evaluator fine-tune
  python contrastive.py --mode train --task disease --log_dir 2_1/freq-pre-large --data ../data --evaluator fine-tune --train_prop 1 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/freq-pre-large --data ../data --evaluator fine-tune
  python contrastive.py --mode train --task disease --log_dir 2_1/time-pre-large --data ../data --evaluator fine-tune --train_prop 1 --epoch 25
  python contrastive.py --mode test --task disease --log_dir 2_1/time-pre-large --data ../data --evaluator fine-tune
  
  #python contrastive.py --mode train --task disease --log_dir 2_1/raw-pre-large --data ../data --evaluator linear --train_prop 1 --epoch 1000
  #python contrastive.py --mode test --task disease --log_dir 2_1/raw-pre-large --data ../data --evaluator linear
  python contrastive.py --mode train --task disease --log_dir 2_1/spec-pre-large --data ../data --evaluator linear --train_prop 1 --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-pre-large --data ../data --evaluator linear
  python contrastive.py --mode train --task disease --log_dir 2_1/split-pre-large --data ../data --evaluator linear --train_prop 1 --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/split-pre-large --data ../data --evaluator linear
  python contrastive.py --mode train --task disease --log_dir 2_1/spec-split-pre-large --data ../data --evaluator linear --train_prop 1 --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/spec-split-pre-large --data ../data --evaluator linear
  python contrastive.py --mode train --task disease --log_dir 2_1/freq-pre-large --data ../data --evaluator linear --train_prop 1 --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/freq-pre-large --data ../data --evaluator linear
  python contrastive.py --mode train --task disease --log_dir 2_1/time-pre-large --data ../data --evaluator linear --train_prop 1 --epoch 1000
  python contrastive.py --mode test --task disease --log_dir 2_1/time-pre-large --data ../data --evaluator linear
done 

# done
echo "Done"

