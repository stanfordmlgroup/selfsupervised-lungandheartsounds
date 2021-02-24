cd ../models ||exit

for i in range 1 2 3 4 5 6 7 8 9 10
do
  python contrastive.py --mode test --task crackle --log_dir rand/crackle --data ../data  --train_prop .01 --epoch 0
  python contrastive.py --mode test --task crackle --log_dir rand/crackle --data ../data  --train_prop 1.0 --epoch 1000 --evaluator linear --model_num 2
  python contrastive.py --mode test --task crackle --log_dir rand/crackle --data ../data  --train_prop 1.0 --epoch 1000 --evaluator linear --model_num 2
done