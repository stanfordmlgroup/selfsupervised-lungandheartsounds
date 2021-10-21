#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="aihc lungheart exp 1.5 selfsuper"
#SBATCH --output=out/exp1-5-selfsupervised-%j.out

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

python contrastive.py --mode pretrain --task heart --log_dir pre-small --data ../heart --augment raw --train_prop .01 --epoch 10

python contrastive.py --mode train --task heart --log_dir pre-small --data ../heart --evaluator fine-tune --train_prop .01 --epoch 25
python contrastive.py --mode test --task heart --log_dir pre-small --data ../heart
python contrastive.py --mode train --task heart --log_dir pre-small --data ../heart --evaluator fine-tune --train_prop .1 --epoch 25
python contrastive.py --mode test --task heart --log_dir pre-small --data ../heart
python contrastive.py --mode train --task heart --log_dir pre-small --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heart --log_dir pre-small --data ../heart

python contrastive.py --mode pretrain --task heart --log_dir pre-medium --data ../heart --augment raw --train_prop .1 --epoch 10

python contrastive.py --mode train --task heart --log_dir pre-medium --data ../heart --evaluator fine-tune --train_prop .01 --epoch 25
python contrastive.py --mode test --task heart --log_dir pre-medium --data ../heart
python contrastive.py --mode train --task heart --log_dir pre-medium --data ../heart --evaluator fine-tune --train_prop .1 --epoch 25
python contrastive.py --mode test --task heart --log_dir pre-medium --data ../heart
python contrastive.py --mode train --task heart --log_dir pre-medium --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heart --log_dir pre-medium --data ../heart

# python contrastive.py --mode pretrain --task heart --log_dir pre-large --data ../heart --augment raw --train_prop 1.0 --epoch 10

# python contrastive.py --mode train --task heart --log_dir pre-large --data ../heart --evaluator fine-tune --train_prop .01 --epoch 25
# python contrastive.py --mode test --task heart --log_dir pre-large --data ../heart
# python contrastive.py --mode train --task heart --log_dir pre-large --data ../heart --evaluator fine-tune --train_prop .1 --epoch 25
# python contrastive.py --mode test --task heart --log_dir pre-large --data ../heart
# python contrastive.py --mode train --task heart --log_dir pre-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
# python contrastive.py --mode test --task heart --log_dir pre-large --data ../heart

# done
echo "Done"

