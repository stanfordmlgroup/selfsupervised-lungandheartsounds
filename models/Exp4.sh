#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=144:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="aihc lungheart exp 3 supervised"
#SBATCH --output=out/exp3-supervised-%j.out

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

python contrastive.py --mode train --task crackle --log_dir supervised-crackle-large --data ../data --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task crackle --log_dir supervised-crackle-large --data ../data
python contrastive.py --mode train --task crackle --log_dir supervised-crackle-raw-large --data ../data --augment raw --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task crackle --log_dir supervised-crackle-raw-large --data ../data
python contrastive.py --mode train --task crackle --log_dir supervised-crackle-spec-large --data ../data --augment spec --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task crackle --log_dir supervised-crackle-spec-large --data ../data
python contrastive.py --mode train --task crackle --log_dir supervised-crackle-split-large --data ../data --augment split --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task crackle --log_dir supervised-crackle-split-large --data ../data
python contrastive.py --mode train --task crackle --log_dir supervised-crackle-spec-split-large --data ../data --augment spec+split --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task crackle --log_dir supervised-crackle-spec-split-large --data ../data

python contrastive.py --mode train --task crackle --log_dir supervised-crackle-large --data ../data --train_prop 1.0 --epochs 25 --full_data True
python contrastive.py --mode test --task crackle --log_dir supervised-crackle-large --data ../data
python contrastive.py --mode train --task crackle --log_dir supervised-crackle-raw-large --data ../data --augment raw --train_prop 1.0 --epochs 25 --full_data True
python contrastive.py --mode test --task crackle --log_dir supervised-crackle-raw-large --data ../data
python contrastive.py --mode train --task crackle --log_dir supervised-crackle-spec-large --data ../data --augment spec --train_prop 1.0 --epochs 25 --full_data True
python contrastive.py --mode test --task crackle --log_dir supervised-crackle-spec-large --data ../data
python contrastive.py --mode train --task crackle --log_dir supervised-crackle-split-large --data ../data --augment split --train_prop 1.0 --epochs 25 --full_data True
python contrastive.py --mode test --task crackle --log_dir supervised-crackle-split-large --data ../data
python contrastive.py --mode train --task crackle --log_dir supervised-crackle-spec-split-large --data ../data --augment spec+split --train_prop 1.0 --epochs 25 --full_data True
python contrastive.py --mode test --task crackle --log_dir supervised-crackle-spec-split-large --data ../data

python contrastive.py --mode train --task wheeze --log_dir supervised-wheeze-large --data ../data --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task wheeze --log_dir supervised-wheeze-large --data ../data
python contrastive.py --mode train --task wheeze --log_dir supervised-wheeze-raw-large --data ../data --augment raw --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task wheeze --log_dir supervised-wheeze-raw-large --data ../data
python contrastive.py --mode train --task wheeze --log_dir supervised-wheeze-spec-large --data ../data --augment spec --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task wheeze --log_dir supervised-wheeze-spec-large --data ../data
python contrastive.py --mode train --task wheeze --log_dir supervised-wheeze-split-large --data ../data --augment split --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task wheeze --log_dir supervised-wheeze-split-large --data ../data
python contrastive.py --mode train --task wheeze --log_dir supervised-wheeze-spec-split-large --data ../data --augment spec+split --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task wheeze --log_dir supervised-wheeze-spec-split-large --data ../data

python contrastive.py --mode train --task wheeze --log_dir supervised-wheeze-large --data ../data --train_prop 1.0 --epochs 25 --full_data True
python contrastive.py --mode test --task wheeze --log_dir supervised-wheeze-large --data ../data
python contrastive.py --mode train --task wheeze --log_dir supervised-wheeze-raw-large --data ../data --augment raw --train_prop 1.0 --epochs 25 --full_data True
python contrastive.py --mode test --task wheeze --log_dir supervised-wheeze-raw-large --data ../data
python contrastive.py --mode train --task wheeze --log_dir supervised-wheeze-spec-large --data ../data --augment spec --train_prop 1.0 --epochs 25 --full_data True
python contrastive.py --mode test --task wheeze --log_dir supervised-wheeze-spec-large --data ../data
python contrastive.py --mode train --task wheeze --log_dir supervised-wheeze-split-large --data ../data --augment split --train_prop 1.0 --epochs 25 --full_data True
python contrastive.py --mode test --task wheeze --log_dir supervised-wheeze-split-large --data ../data
python contrastive.py --mode train --task wheeze --log_dir supervised-wheeze-spec-split-large --data ../data --augment spec+split --train_prop 1.0 --epochs 25 --full_data True
python contrastive.py --mode test --task wheeze --log_dir supervised-wheeze-spec-split-large --data ../data

python contrastive.py --mode train --task heartchallenge --log_dir supervised-heartchallenge-large --data ../heartchallenge --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task heartchallenge --log_dir supervised-heartchallenge-large --data ../heartchallenge
python contrastive.py --mode train --task heartchallenge --log_dir supervised-heartchallenge-raw-large --data ../heartchallenge --augment raw --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task heartchallenge --log_dir supervised-heartchallenge-raw-large --data ../heartchallenge
python contrastive.py --mode train --task heartchallenge --log_dir supervised-heartchallenge-spec-large --data ../heartchallenge --augment spec --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task heartchallenge --log_dir supervised-heartchallenge-spec-large --data ../heartchallenge
python contrastive.py --mode train --task heartchallenge --log_dir supervised-heartchallenge-split-large --data ../heartchallenge --augment split --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task heartchallenge --log_dir supervised-heartchallenge-split-large --data ../heartchallenge
python contrastive.py --mode train --task heartchallenge --log_dir supervised-heartchallenge-spec-split-large --data ../heartchallenge --augment spec+split --train_prop 1.0 --epochs 25
python contrastive.py --mode test --task heartchallenge --log_dir supervised-heartchallenge-spec-split-large --data ../heartchallenge

# done
echo "Done"

