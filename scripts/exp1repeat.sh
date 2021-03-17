#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=144:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="exp 1 aihc lung"
#SBATCH --output=out/exp1-%j.out

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate lungsounds
cd ../models

#for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
#do
#	python contrastive.py --mode train --task heart --log_dir 3_16/supervised-small --data ../heart --train_prop .1 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 3_16/supervised-small --data ../heart --evaluator fine-tune --model_num $i
#
#	python contrastive.py --mode train --task heart --log_dir 3_16/supervised-small-full --data ../heart --train_prop .1 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 3_16/supervised-small-full --data ../heart --evaluator fine-tune --model_num $i
#done
#
#for i in 0 1 2 3 4 5 6 7 8 9
#do
#	python contrastive.py --mode train --task heart --log_dir 3_16/supervised-medium --data ../heart --train_prop .5 --epochs 25 --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 3_16/supervised-medium --data ../heart --evaluator fine-tune --model_num $i
#
#	python contrastive.py --mode train --task heart --log_dir 3_16/supervised-medium-full --data ../heart --train_prop .5 --epochs 25 --full_data True --evaluator fine-tune
#	python contrastive.py --mode test --task heart --log_dir 3_16/supervised-medium-full --data ../heart --evaluator fine-tune --model_num $i
#done
#
#for i in 0 1 2 3 4
#do
#  python contrastive.py --mode train --task heart --log_dir 3_16/supervised-large --data ../heart --train_prop 1 --epochs 25 --evaluator fine-tune
#  python contrastive.py --mode test --task heart --log_dir 3_16/supervised-large --data ../heart --evaluator fine-tune --model_num $i
#
#  python contrastive.py --mode train --task heart --log_dir 3_16/supervised-large-full --data ../heart --train_prop 1 --epochs 25 --full_data True --evaluator fine-tune
#  python contrastive.py --mode test --task heart --log_dir 3_16/supervised-large-full --data ../heart --evaluator fine-tune --model_num $i
#done

#python contrastive.py --mode pretrain --task heart --log_dir 3_16/spec-pre-large --data ../heart --augment spec --train_prop 1.0 --epoch 10
#python contrastive.py --mode pretrain --task heart --log_dir 3_16/split-pre-large --data ../heart --augment split --train_prop 1.0 --epoch 10
#python contrastive.py --mode pretrain --task heart --log_dir 3_16/spec-split-pre-large --data ../heart --augment spec+split --train_prop 1.0 --epoch 10
#python contrastive.py --mode pretrain --task heart --log_dir 3_16/time-pre-large --data ../heart --augment time --train_prop 1.0 --epoch 10
#python contrastive.py --mode pretrain --task heart --log_dir 3_16/freq-pre-large --data ../heart --augment freq --train_prop 1.0 --epoch 10
#
#wait
#cd ../scripts
#sbatch Exp4HC.sh &
#sbatch Exp4Symp.sh &
#cd ../models

#for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
#do
#	python contrastive.py --mode train --task heart --log_dir 3_16/spec-pre-large --data ../heart --evaluator fine-tune --train_prop .1 --epoch 25  --model_num $i
#	python contrastive.py --mode test --task heart --log_dir 3_16/spec-pre-large --data ../heart --evaluator fine-tune --model_num $i
#	python contrastive.py --mode train --task heart --log_dir 3_16/split-pre-large --data ../heart --evaluator fine-tune --train_prop .1 --epoch 25 --model_num $i
#	python contrastive.py --mode test --task heart --log_dir 3_16/split-pre-large --data ../heart --evaluator fine-tune --model_num $i
#	python contrastive.py --mode train --task heart --log_dir 3_16/spec-split-pre-large --data ../heart --evaluator fine-tune --train_prop .1 --epoch 25 --model_num $i
#	python contrastive.py --mode test --task heart --log_dir 3_16/spec-split-pre-large --data ../heart --evaluator fine-tune --model_num $i
#	python contrastive.py --mode train --task heart --log_dir 3_16/time-pre-large --data ../heart --evaluator fine-tune --train_prop .1 --epoch 25 --model_num $i
#	python contrastive.py --mode test --task heart --log_dir 3_16/time-pre-large --data ../heart --evaluator fine-tune --model_num $i
# python contrastive.py --mode train --task heart --log_dir 3_16/freq-pre-large --data ../heart --evaluator fine-tune --train_prop .1 --epoch 25 --model_num $i
#	python contrastive.py --mode test --task heart --log_dir 3_16/freq-pre-large --data ../heart --evaluator fine-tune --model_num $i
#done
#for i in 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
#do
#	python contrastive.py --mode train --task heart --log_dir 3_16/spec-pre-large --data ../heart --evaluator linear --train_prop .1 --epoch 5000 --model_num $i
#	python contrastive.py --mode test --task heart --log_dir 3_16/spec-pre-large --data ../heart --evaluator linear --model_num $i
#	python contrastive.py --mode train --task heart --log_dir 3_16/split-pre-large --data ../heart --evaluator linear --train_prop .1 --epoch 5000 --model_num $i
#	python contrastive.py --mode test --task heart --log_dir 3_16/split-pre-large --data ../heart --evaluator linear --model_num $i
#	python contrastive.py --mode train --task heart --log_dir 3_16/spec-split-pre-large --data ../heart --evaluator linear --train_prop .1 --epoch 5000 --model_num $i
#	python contrastive.py --mode test --task heart --log_dir 3_16/spec-split-pre-large --data ../heart --evaluator linear --model_num $i
# python contrastive.py --mode train --task heart --log_dir 3_16/time-pre-large --data ../heart --evaluator linear --train_prop .1 --epoch 5000 --model_num $i
#	python contrastive.py --mode test --task heart --log_dir 3_16/time-pre-large --data ../heart --evaluator linear --model_num $i
# python contrastive.py --mode train --task heart --log_dir 3_16/freq-pre-large --data ../heart --evaluator linear --train_prop .1 --epoch 5000 --model_num $i
#	python contrastive.py --mode test --task heart --log_dir 3_16/freq-pre-large --data ../heart --evaluator linear --model_num $i
#
#done

for i in 40 41 42 43 44 45 46 47 48 49
do
	python contrastive.py --mode train --task heart --log_dir 3_16/spec-pre-large --data ../heart --evaluator fine-tune --train_prop .5 --epoch 25 --model_num $i
	python contrastive.py --mode test --task heart --log_dir 3_16/spec-pre-large --data ../heart --evaluator fine-tune --model_num $i
	python contrastive.py --mode train --task heart --log_dir 3_16/split-pre-large --data ../heart --evaluator fine-tune --train_prop .5 --epoch 25 --model_num $i
	python contrastive.py --mode test --task heart --log_dir 3_16/split-pre-large --data ../heart --evaluator fine-tune --model_num $i
	python contrastive.py --mode train --task heart --log_dir 3_16/spec-split-pre-large --data ../heart --evaluator fine-tune --train_prop .5 --epoch 25 --model_num $i
	python contrastive.py --mode test --task heart --log_dir 3_16/spec-split-pre-large --data ../heart --evaluator fine-tune --model_num $i
	python contrastive.py --mode train --task heart --log_dir 3_16/freq-pre-large --data ../heart --evaluator fine-tune --train_prop .5 --epoch 25 --model_num $i
	python contrastive.py --mode test --task heart --log_dir 3_16/freq-pre-large --data ../heart --evaluator fine-tune --model_num $i
  python contrastive.py --mode train --task heart --log_dir 3_16/time-pre-large --data ../heart --evaluator fine-tune --train_prop .5 --epoch 25 --model_num $i
	python contrastive.py --mode test --task heart --log_dir 3_16/time-pre-large --data ../heart --evaluator fine-tune --model_num $i
done
for i in 50 51 52 53 54 55 56 57 58 59
do
	python contrastive.py --mode train --task heart --log_dir 3_16/spec-pre-large --data ../heart --evaluator linear --train_prop .5 --epoch 5000 --model_num $i
	python contrastive.py --mode test --task heart --log_dir 3_16/spec-pre-large --data ../heart --evaluator linear --model_num $i
	python contrastive.py --mode train --task heart --log_dir 3_16/split-pre-large --data ../heart --evaluator linear --train_prop .5 --epoch 5000 --model_num $i
	python contrastive.py --mode test --task heart --log_dir 3_16/split-pre-large --data ../heart --evaluator linear --model_num $i
	python contrastive.py --mode train --task heart --log_dir 3_16/spec-split-pre-large --data ../heart --evaluator linear --train_prop .5 --epoch 5000 --model_num $i
	python contrastive.py --mode test --task heart --log_dir 3_16/spec-split-pre-large --data ../heart --evaluator linear --model_num $i
  python contrastive.py --mode train --task heart --log_dir 3_16/freq-pre-large --data ../heart --evaluator linear --train_prop .5 --epoch 5000 --model_num $i
	python contrastive.py --mode test --task heart --log_dir 3_16/freq-pre-large --data ../heart --evaluator linear --model_num $i
  python contrastive.py --mode train --task heart --log_dir 3_16/time-pre-large --data ../heart --evaluator linear --train_prop .5 --epoch 5000 --model_num $i
	python contrastive.py --mode test --task heart --log_dir 3_16/time-pre-large --data ../heart --evaluator linear --model_num $i
done

for i in 60 61 62 63 64
do
  python contrastive.py --mode train --task heart --log_dir 3_16/spec-pre-large --data ../heart --evaluator fine-tune --train_prop 1 --epoch 25 --model_num $i
  python contrastive.py --mode test --task heart --log_dir 3_16/spec-pre-large --data ../heart --evaluator fine-tune --model_num $i
  python contrastive.py --mode train --task heart --log_dir 3_16/split-pre-large --data ../heart --evaluator fine-tune --train_prop 1 --epoch 25 --model_num $i
  python contrastive.py --mode test --task heart --log_dir 3_16/split-pre-large --data ../heart --evaluator fine-tune --model_num $i
  python contrastive.py --mode train --task heart --log_dir 3_16/spec-split-pre-large --data ../heart --evaluator fine-tune --train_prop 1 --epoch 25 --model_num $i
  python contrastive.py --mode test --task heart --log_dir 3_16/spec-split-pre-large --data ../heart --evaluator fine-tune --model_num $i
  python contrastive.py --mode train --task heart --log_dir 3_16/freq-pre-large --data ../heart --evaluator fine-tune --train_prop 1 --epoch 25 --model_num $i
  python contrastive.py --mode test --task heart --log_dir 3_16/freq-pre-large --data ../heart --evaluator fine-tune --model_num $i
  python contrastive.py --mode train --task heart --log_dir 3_16/time-pre-large --data ../heart --evaluator fine-tune --train_prop 1 --epoch 25 --model_num $i
  python contrastive.py --mode test --task heart --log_dir 3_16/time-pre-large --data ../heart --evaluator fine-tune  --model_num $i
done

for i in 65 66 67 68 69
do
  python contrastive.py --mode train --task heart --log_dir 3_16/spec-pre-large --data ../heart --evaluator linear --train_prop 1 --epoch 5000 --model_num $i
  python contrastive.py --mode test --task heart --log_dir 3_16/spec-pre-large --data ../heart --evaluator linear --model_num $i
  python contrastive.py --mode train --task heart --log_dir 3_16/split-pre-large --data ../heart --evaluator linear --train_prop 1 --epoch 5000 --model_num $i
  python contrastive.py --mode test --task heart --log_dir 3_16/split-pre-large --data ../heart --evaluator linear --model_num $i
  python contrastive.py --mode train --task heart --log_dir 3_16/spec-split-pre-large --data ../heart --evaluator linear --train_prop 1 --epoch 5000 --model_num $i
  python contrastive.py --mode test --task heart --log_dir 3_16/spec-split-pre-large --data ../heart --evaluator linear --model_num $i
  python contrastive.py --mode train --task heart --log_dir 3_16/freq-pre-large --data ../heart --evaluator linear --train_prop 1 --epoch 5000 --model_num $i
  python contrastive.py --mode test --task heart --log_dir 3_16/freq-pre-large --data ../heart --evaluator linear --model_num $i
  python contrastive.py --mode train --task heart --log_dir 3_16/time-pre-large --data ../heart --evaluator linear --train_prop 1 --epoch 5000 --model_num $i
  python contrastive.py --mode test --task heart --log_dir 3_16/time-pre-large --data ../heart --evaluator linear --model_num $i
done 


wait
#cd ../scripts
#sbatch Exp5Heart.sh &
#sbatch Exp5Lung.sh &
#cd ../models
# done
echo "Done"

