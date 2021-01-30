@echo off
cd /D "%~dp0"
# python contrastive.py --mode train --task disease --log_dir raw-pre-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
# python contrastive.py --mode test --task disease --log_dir raw-pre-large --data ../data
# python contrastive.py --mode train --task disease --log_dir raw-pre-large --data ../data --evaluator linear --train_prop 1.
# python contrastive.py --mode test --task disease --log_dir raw-pre-large --data ../data
# python contrastive.py --mode train --task disease --log_dir spec-pre-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
# python contrastive.py --mode test --task disease --log_dir spec-pre-large --data ../data
# python contrastive.py --mode train --task disease --log_dir spec-pre-large --data ../data --evaluator linear --train_prop 1.
# python contrastive.py --mode test --task disease --log_dir spec-pre-large --data ../data
# python contrastive.py --mode train --task disease --log_dir split-pre-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
# python contrastive.py --mode test --task disease --log_dir split-pre-large --data ../data
# python contrastive.py --mode train --task disease --log_dir split-pre-large --data ../data --evaluator linear --train_prop 1.
# python contrastive.py --mode test --task disease --log_dir split-pre-large --data ../data
# python contrastive.py --mode train --task disease --log_dir spec-split-pre-large --data ../data --evaluator fine-tune --train_prop 1.0 --epoch 25
# python contrastive.py --mode test --task disease --log_dir spec-split-pre-large --data ../data
# python contrastive.py --mode train --task disease --log_dir spec-split-pre-large --data ../data --evaluator linear --train_prop 1.
# python contrastive.py --mode test --task disease --log_dir spec-split-pre-large --data ../data
python contrastive.py --mode train --task heart --log_dir raw-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heart --log_dir raw-pre-lung-large --data ../heart
python contrastive.py --mode train --task heart --log_dir raw-pre-lung-large --data ../heart --evaluator linear --train_prop 1.
python contrastive.py --mode test --task heart --log_dir raw-pre-lung-large --data ../heart
python contrastive.py --mode train --task heart --log_dir spec-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heart --log_dir spec-pre-lung-large --data ../heart
python contrastive.py --mode train --task heart --log_dir spec-pre-lung-large --data ../heart --evaluator linear --train_prop 1.
python contrastive.py --mode test --task heart --log_dir spec-pre-lung-large --data ../heart
python contrastive.py --mode train --task heart --log_dir split-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heart --log_dir split-pre-lung-large --data ../heart
python contrastive.py --mode train --task heart --log_dir split-pre-lung-large --data ../heart --evaluator linear --train_prop 1.
python contrastive.py --mode test --task heart --log_dir split-pre-lung-large --data ../heart
python contrastive.py --mode train --task heart --log_dir spec-split-pre-lung-large --data ../heart --evaluator fine-tune --train_prop 1.0 --epoch 25
python contrastive.py --mode test --task heart --log_dir spec-split-pre-lung-large --data ../heart
python contrastive.py --mode train --task heart --log_dir spec-split-pre-lung-large --data ../heart --evaluator linear --train_prop 1.
python contrastive.py --mode test --task heart --log_dir spec-split-pre-lung-large --data ../heart
popd
endlocal
# done
ECHO "Done"
