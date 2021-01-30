#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=144:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="aihc lungheart exp 2 supervised"
#SBATCH --output=out/exp2-supervised-%j.out

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

python contrastive.py --mode train --task disease --log_dir supervised-disease-small --data ../data --train_prop .01 --epochs 25
python contrastive.py --mode test --task disease --log_dir supervised-disease-small --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-raw-small --data ../data --augment raw --train_prop .01 --epochs 25
python contrastive.py --mode test --task disease --log_dir supervised-disease-raw-small --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-spec-small --data ../data --augment spec --train_prop .01 --epochs 25
python contrastive.py --mode test --task disease --log_dir supervised-disease-spec-small --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-split-small --data ../data --augment split --train_prop .01 --epochs 25
python contrastive.py --mode test --task disease --log_dir supervised-disease-split-small --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-spec-split-small --data ../data --augment spec+split --train_prop .01 --epochs 25
python contrastive.py --mode test --task disease --log_dir supervised-disease-spec-split-small --data ../data

python contrastive.py --mode train --task disease --log_dir supervised-disease-small --data ../data --train_prop .01 --epochs 25 --full_data True
python contrastive.py --mode test --task disease --log_dir supervised-disease-small --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-raw-small --data ../data --augment raw --train_prop .01 --epochs 25 --full_data True
python contrastive.py --mode test --task disease --log_dir supervised-disease-raw-small --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-spec-small --data ../data --augment spec --train_prop .01 --epochs 25 --full_data True
python contrastive.py --mode test --task disease --log_dir supervised-disease-spec-small --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-split-small --data ../data --augment split --train_prop .01 --epochs 25 --full_data True
python contrastive.py --mode test --task disease --log_dir supervised-disease-split-small --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-spec-split-small --data ../data --augment spec+split --train_prop .01 --epochs 25 --full_data True
python contrastive.py --mode test --task disease --log_dir supervised-disease-spec-split-small --data ../data

python contrastive.py --mode train --task disease --log_dir supervised-disease-medium --data ../data --train_prop .1 --epochs 25
python contrastive.py --mode test --task disease --log_dir supervised-disease-medium --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-raw-medium --data ../data --augment raw --train_prop .1 --epochs 25
python contrastive.py --mode test --task disease --log_dir supervised-disease-raw-medium --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-spec-medium --data ../data --augment spec --train_prop .1 --epochs 25
python contrastive.py --mode test --task disease --log_dir supervised-disease-spec-medium --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-split-medium --data ../data --augment split --train_prop .1 --epochs 25
python contrastive.py --mode test --task disease --log_dir supervised-disease-split-medium --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-spec-split-medium --data ../data --augment spec+split --train_prop .1 --epochs 25
python contrastive.py --mode test --task disease --log_dir supervised-disease-spec-split-medium --data ../data

python contrastive.py --mode train --task disease --log_dir supervised-disease-medium --data ../data --train_prop .1 --epochs 25 --full_data True
python contrastive.py --mode test --task disease --log_dir supervised-disease-medium --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-raw-medium --data ../data --augment raw --train_prop .1 --epochs 25 --full_data True
python contrastive.py --mode test --task disease --log_dir supervised-disease-raw-medium --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-spec-medium --data ../data --augment spec --train_prop .1 --epochs 25 --full_data True
python contrastive.py --mode test --task disease --log_dir supervised-disease-spec-medium --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-split-medium --data ../data --augment split --train_prop .1 --epochs 25 --full_data True
python contrastive.py --mode test --task disease --log_dir supervised-disease-split-medium --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-spec-split-medium --data ../data --augment spec+split --train_prop .1 --epochs 25 --full_data True
python contrastive.py --mode test --task disease --log_dir supervised-disease-spec-split-medium --data ../data

python contrastive.py --mode train --task disease --log_dir supervised-disease-large --data ../data --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task disease --log_dir supervised-disease-large --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-raw-large --data ../data --augment raw --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task disease --log_dir supervised-disease-raw-large --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-spec-large --data ../data --augment spec --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task disease --log_dir supervised-disease-spec-large --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-split-large --data ../data --augment split --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task disease --log_dir supervised-disease-split-large --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-spec-split-large --data ../data --augment spec+split --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task disease --log_dir supervised-disease-spec-split-large --data ../data

python contrastive.py --mode train --task disease --log_dir supervised-disease-large --data ../data --train_prop 1.0 --epochs 25 --full_data True
python contrastive.py --mode test --task disease --log_dir supervised-disease-large --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-raw-large --data ../data --augment raw --train_prop 1.0 --epochs 25 --full_data True
python contrastive.py --mode test --task disease --log_dir supervised-disease-raw-large --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-spec-large --data ../data --augment spec --train_prop 1.0 --epochs 25 --full_data True
python contrastive.py --mode test --task disease --log_dir supervised-disease-spec-large --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-split-large --data ../data --augment split --train_prop 1.0 --epochs 25 --full_data True
python contrastive.py --mode test --task disease --log_dir supervised-disease-split-large --data ../data
python contrastive.py --mode train --task disease --log_dir supervised-disease-spec-split-large --data ../data --augment spec+split --train_prop 1.0 --epochs 25 --full_data True
python contrastive.py --mode test --task disease --log_dir supervised-disease-spec-split-large --data ../data

# done
echo "Done"

