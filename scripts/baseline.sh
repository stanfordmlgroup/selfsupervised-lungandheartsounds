#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=144:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="baseline aihc lung"
#SBATCH --output=out/baseline-repeat-%j.out

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
source ~/.bashrc
conda activate lungsounds
cd ../models ||exit

for i in range 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
do
  python contrastive.py --mode train --task disease --log_dir 3_27/ baseline --data ../data  --train_prop .1 --epoch 5000 --evaluator linear
  python contrastive.py --mode test --task disease --log_dir 3_27/ baseline --data ../data --evaluator linear --model_num $i
done

for i in range 20 21 22 23 24 25 26 27 28 29
do
  python contrastive.py --mode train --task disease --log_dir 3_27/ baseline --data ../data  --train_prop .5 --epoch 5000 --evaluator linear
  python contrastive.py --mode test --task disease --log_dir 3_27/ baseline --data ../data --evaluator linear --model_num $i
done
for i in range 30 31 32 33 34
do
  python contrastive.py --mode train --task disease --log_dir 3_27/ baseline --data ../data  --train_prop 1.0 --epoch 5000 --evaluator linear
  python contrastive.py --mode test --task disease --log_dir 3_27/ baseline --data ../data --evaluator linear --model_num $i
done

for i in range 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
do
  python contrastive.py --mode train --task heart --log_dir 3_27/ baseline --data ../heart  --train_prop .1 --epoch 5000 --evaluator linear
  python contrastive.py --mode test --task heart --log_dir 3_27/ baseline --data ../heart --evaluator linear --model_num $i
done

for i in range 20 21 22 23 24 25 26 27 28 29
do
  python contrastive.py --mode train --task heart --log_dir 3_27/ baseline --data ../heart  --train_prop .5 --epoch 5000 --evaluator linear
  python contrastive.py --mode test --task heart --log_dir 3_27/ baseline --data ../heart --evaluator linear --model_num $i
done

for i in range 30 31 32 33 34
do
  python contrastive.py --mode train --task heart --log_dir 3_27/ baseline --data ../heart  --train_prop 1.0 --epoch 5000 --evaluator linear
  python contrastive.py --mode test --task heart --log_dir 3_27/ baseline --data ../heart --evaluator linear --model_num $i
done

#done
echo "Done."