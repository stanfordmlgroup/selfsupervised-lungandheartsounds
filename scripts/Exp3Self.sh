#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="aihc lungheart exp 3 selfsuper"
#SBATCH --output=out/exp3-selfsupervised-%j.out

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

# python contrastive.py --mode pretrain --task demo --log_dir demo0-pre-lung-large --data ../data --train_prop 1.0 --epoch 10 --exp 0

python contrastive.py --mode train --task demo --log_dir demo0-pre-lung-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
python contrastive.py --mode test --task demo --log_dir demo0-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo0-pre-lung-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
python contrastive.py --mode test --task demo --log_dir demo0-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo0-pre-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task demo --log_dir demo0-pre-lung-large --data ../data

python contrastive.py --mode train --task demo --log_dir demo0-pre-lung-large --data ../data --evaluator linear --train_prop .01
python contrastive.py --mode test --task demo --log_dir demo0-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo0-pre-lung-large --data ../data --evaluator linear --train_prop .1
python contrastive.py --mode test --task demo --log_dir demo0-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo0-pre-lung-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task demo --log_dir demo0-pre-lung-large --data ../data

# python contrastive.py --mode pretrain --task demo --log_dir demo1-pre-lung-large --data ../data --train_prop 1.0 --epoch 10 --exp 1

python contrastive.py --mode train --task demo --log_dir demo1-pre-lung-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
python contrastive.py --mode test --task demo --log_dir demo1-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo1-pre-lung-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
python contrastive.py --mode test --task demo --log_dir demo1-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo1-pre-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task demo --log_dir demo1-pre-lung-large --data ../data

python contrastive.py --mode train --task demo --log_dir demo1-pre-lung-large --data ../data --evaluator linear --train_prop .01
python contrastive.py --mode test --task demo --log_dir demo1-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo1-pre-lung-large --data ../data --evaluator linear --train_prop .1
python contrastive.py --mode test --task demo --log_dir demo1-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo1-pre-lung-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task demo --log_dir demo1-pre-lung-large --data ../data

# python contrastive.py --mode pretrain --task demo --log_dir demo2-pre-lung-large --data ../data --train_prop 1.0 --epoch 10 --exp 2

python contrastive.py --mode train --task demo --log_dir demo2-pre-lung-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
python contrastive.py --mode test --task demo --log_dir demo2-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo2-pre-lung-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
python contrastive.py --mode test --task demo --log_dir demo2-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo2-pre-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task demo --log_dir demo2-pre-lung-large --data ../data

python contrastive.py --mode train --task demo --log_dir demo2-pre-lung-large --data ../data --evaluator linear --train_prop .01
python contrastive.py --mode test --task demo --log_dir demo2-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo2-pre-lung-large --data ../data --evaluator linear --train_prop .1
python contrastive.py --mode test --task demo --log_dir demo2-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo2-pre-lung-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task demo --log_dir demo2-pre-lung-large --data ../data

# python contrastive.py --mode pretrain --task demo --log_dir demo3-pre-lung-large --data ../data --train_prop 1.0 --epoch 10 --exp 3

# python contrastive.py --mode train --task demo --log_dir demo3-pre-lung-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
# python contrastive.py --mode test --task demo --log_dir demo3-pre-lung-large --data ../data
# python contrastive.py --mode train --task demo --log_dir demo3-pre-lung-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
# python contrastive.py --mode test --task demo --log_dir demo3-pre-lung-large --data ../data
# python contrastive.py --mode train --task demo --log_dir demo3-pre-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
# python contrastive.py --mode test --task demo --log_dir demo3-pre-lung-large --data ../data

# python contrastive.py --mode train --task demo --log_dir demo3-pre-lung-large --data ../data --evaluator linear --train_prop .01
# python contrastive.py --mode test --task demo --log_dir demo3-pre-lung-large --data ../data
# python contrastive.py --mode train --task demo --log_dir demo3-pre-lung-large --data ../data --evaluator linear --train_prop .1
# python contrastive.py --mode test --task demo --log_dir demo3-pre-lung-large --data ../data
# python contrastive.py --mode train --task demo --log_dir demo3-pre-lung-large --data ../data --evaluator linear --train_prop 1.
# python contrastive.py --mode test --task demo --log_dir demo3-pre-lung-large --data ../data

# python contrastive.py --mode pretrain --task demo --log_dir demo4-pre-lung-large --data ../data --train_prop 1.0 --epoch 10 --exp 4

python contrastive.py --mode train --task demo --log_dir demo4-pre-lung-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
python contrastive.py --mode test --task demo --log_dir demo4-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo4-pre-lung-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
python contrastive.py --mode test --task demo --log_dir demo4-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo4-pre-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task demo --log_dir demo4-pre-lung-large --data ../data

python contrastive.py --mode train --task demo --log_dir demo4-pre-lung-large --data ../data --evaluator linear --train_prop .01
python contrastive.py --mode test --task demo --log_dir demo4-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo4-pre-lung-large --data ../data --evaluator linear --train_prop .1
python contrastive.py --mode test --task demo --log_dir demo4-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo4-pre-lung-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task demo --log_dir demo4-pre-lung-large --data ../data

# python contrastive.py --mode pretrain --task demo --log_dir demo5-pre-lung-large --data ../data --train_prop 1.0 --epoch 10 --exp 5

python contrastive.py --mode train --task demo --log_dir demo5-pre-lung-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
python contrastive.py --mode test --task demo --log_dir demo5-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo5-pre-lung-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
python contrastive.py --mode test --task demo --log_dir demo5-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo5-pre-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task demo --log_dir demo5-pre-lung-large --data ../data

python contrastive.py --mode train --task demo --log_dir demo5-pre-lung-large --data ../data --evaluator linear --train_prop .01
python contrastive.py --mode test --task demo --log_dir demo5-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo5-pre-lung-large --data ../data --evaluator linear --train_prop .1
python contrastive.py --mode test --task demo --log_dir demo5-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo5-pre-lung-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task demo --log_dir demo5-pre-lung-large --data ../data

# python contrastive.py --mode pretrain --task demo --log_dir demo6-pre-lung-large --data ../data --train_prop 1.0 --epoch 10 --exp 6

python contrastive.py --mode train --task demo --log_dir demo6-pre-lung-large --data ../data --evaluator fine-tune --train_prop .01 --epoch 25
python contrastive.py --mode test --task demo --log_dir demo6-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo6-pre-lung-large --data ../data --evaluator fine-tune --train_prop .1 --epoch 25
python contrastive.py --mode test --task demo --log_dir demo6-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo6-pre-lung-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task demo --log_dir demo6-pre-lung-large --data ../data

python contrastive.py --mode train --task demo --log_dir demo6-pre-lung-large --data ../data --evaluator linear --train_prop .01
python contrastive.py --mode test --task demo --log_dir demo6-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo6-pre-lung-large --data ../data --evaluator linear --train_prop .1
python contrastive.py --mode test --task demo --log_dir demo6-pre-lung-large --data ../data
python contrastive.py --mode train --task demo --log_dir demo6-pre-lung-large --data ../data --evaluator linear --train_prop 1.
python contrastive.py --mode test --task demo --log_dir demo6-pre-lung-large --data ../data

# done
echo "Done"
