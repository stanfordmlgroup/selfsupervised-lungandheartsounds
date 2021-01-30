#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=144:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="aihc lungheart exp 1 supervised"
#SBATCH --output=out/exp1-supervised-%j.out

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

# python contrastive.py --mode train --task heart --log_dir supervised-small --data ../heart --train_prop .01 --epochs 25
# python contrastive.py --mode test --task heart --log_dir supervised-small --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-raw-small --data ../heart --augment raw --train_prop .01 --epochs 25
# python contrastive.py --mode test --task heart --log_dir supervised-raw-small --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-spec-small --data ../heart --augment spec --train_prop .01 --epochs 25
# python contrastive.py --mode test --task heart --log_dir supervised-spec-small --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-split-small --data ../heart --augment split --train_prop .01 --epochs 25
# python contrastive.py --mode test --task heart --log_dir supervised-split-small --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-spec-split-small --data ../heart --augment spec+split --train_prop .01 --epochs 25
# python contrastive.py --mode test --task heart --log_dir supervised-spec-split-small --data ../heart

# python contrastive.py --mode train --task heart --log_dir supervised-small --data ../heart --train_prop .01 --epochs 25 --full_data True
# python contrastive.py --mode test --task heart --log_dir supervised-small --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-raw-small --data ../heart --augment raw --train_prop .01 --epochs 25 --full_data True
# python contrastive.py --mode test --task heart --log_dir supervised-raw-small --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-spec-small --data ../heart --augment spec --train_prop .01 --epochs 25 --full_data True
# python contrastive.py --mode test --task heart --log_dir supervised-spec-small --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-split-small --data ../heart --augment split --train_prop .01 --epochs 25 --full_data True
# python contrastive.py --mode test --task heart --log_dir supervised-split-small --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-spec-split-small --data ../heart --augment spec+split --train_prop .01 --epochs 25 --full_data True
# python contrastive.py --mode test --task heart --log_dir supervised-spec-split-small --data ../heart

# python contrastive.py --mode train --task heart --log_dir supervised-medium --data ../heart --train_prop .1 --epochs 25
# python contrastive.py --mode test --task heart --log_dir supervised-medium --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-raw-medium --data ../heart --augment raw --train_prop .1 --epochs 25
# python contrastive.py --mode test --task heart --log_dir supervised-raw-medium --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-spec-medium --data ../heart --augment spec --train_prop .1 --epochs 25
# python contrastive.py --mode test --task heart --log_dir supervised-spec-medium --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-split-medium --data ../heart --augment split --train_prop .1 --epochs 25
# python contrastive.py --mode test --task heart --log_dir supervised-split-medium --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-spec-split-medium --data ../heart --augment spec+split --train_prop .1 --epochs 25
# python contrastive.py --mode test --task heart --log_dir supervised-spec-split-medium --data ../heart

# python contrastive.py --mode train --task heart --log_dir supervised-medium --data ../heart --train_prop .1 --epochs 25 --full_data True
# python contrastive.py --mode test --task heart --log_dir supervised-medium --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-raw-medium --data ../heart --augment raw --train_prop .1 --epochs 25 --full_data True
# python contrastive.py --mode test --task heart --log_dir supervised-raw-medium --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-spec-medium --data ../heart --augment spec --train_prop .1 --epochs 25 --full_data True
# python contrastive.py --mode test --task heart --log_dir supervised-spec-medium --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-split-medium --data ../heart --augment split --train_prop .1 --epochs 25 --full_data True
# python contrastive.py --mode test --task heart --log_dir supervised-split-medium --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-spec-split-medium --data ../heart --augment spec+split --train_prop .1 --epochs 25 --full_data True
# python contrastive.py --mode test --task heart --log_dir supervised-spec-split-medium --data ../heart

# python contrastive.py --mode train --task heart --log_dir supervised-large --data ../heart --train_prop 1.0 --epochs 25
# python contrastive.py --mode test --task heart --log_dir supervised-large --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-raw-large --data ../heart --augment raw --train_prop 1.0 --epochs 25
# python contrastive.py --mode test --task heart --log_dir supervised-raw-large --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-spec-large --data ../heart --augment spec --train_prop 1.0 --epochs 25
# python contrastive.py --mode test --task heart --log_dir supervised-spec-large --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-split-large --data ../heart --augment split --train_prop 1.0 --epochs 25
# python contrastive.py --mode test --task heart --log_dir supervised-split-large --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-spec-split-large --data ../heart --augment spec+split --train_prop 1.0 --epochs 25
# python contrastive.py --mode test --task heart --log_dir supervised-spec-split-large --data ../heart

# python contrastive.py --mode train --task heart --log_dir supervised-large --data ../heart --train_prop 1.0 --epochs 25 --full_data True
# python contrastive.py --mode test --task heart --log_dir supervised-large --data ../heart
python contrastive.py --mode train --task heart --log_dir supervised-raw-large --data ../heart --augment raw --train_prop 1.0 --epochs 25 --full_data True 
python contrastive.py --mode test --task heart --log_dir supervised-raw-large --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-spec-large --data ../heart --augment spec --train_prop 1.0 --epochs 25 --full_data True
# python contrastive.py --mode test --task heart --log_dir supervised-spec-large --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-split-large --data ../heart --augment split --train_prop 1.0 --epochs 25 --full_data True
# python contrastive.py --mode test --task heart --log_dir supervised-split-large --data ../heart
# python contrastive.py --mode train --task heart --log_dir supervised-spec-split-large --data ../heart --augment spec+split --train_prop 1.0 --epochs 25 --full_data True
# python contrastive.py --mode test --task heart --log_dir supervised-spec-split-large --data ../heart

# done
echo "Done"

