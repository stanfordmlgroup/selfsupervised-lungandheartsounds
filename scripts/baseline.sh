cd ../models ||exit

#for i in range 0 1 2 3 4 5 6 7 8 9 10
#do
#  python contrastive.py --mode pretrain --task crackle --log_dir rand/crackle --data ../data  --train_prop .01 --epoch 0
#  python contrastive.py --mode train --task crackle --log_dir rand/crackle --data ../data  --train_prop 1.0 --epoch 1000 --evaluator linear
#  python contrastive.py --mode test --task crackle --log_dir rand/crackle --data ../data --evaluator linear --model_num $i
#done
#
#for i in range 0 1 2 3 4 5 6 7 8 9 10
#do
#  python contrastive.py --mode pretrain --task wheeze --log_dir rand/wheeze --data ../data  --train_prop .01 --epoch 0
#  python contrastive.py --mode train --task wheeze --log_dir rand/wheeze --data ../data  --train_prop 1.0 --epoch 1000 --evaluator linear
#  python contrastive.py --mode test --task wheeze --log_dir rand/wheeze --data ../data --evaluator linear --model_num $i
#done
#
#for i in range 0 1 2 3 4 5 6 7 8 9 10
#do
#  python contrastive.py --mode pretrain --task disease --log_dir rand/disease --data ../data  --train_prop .01 --epoch 0
#  python contrastive.py --mode train --task disease --log_dir rand/disease --data ../data  --train_prop 1.0 --epoch 1000 --evaluator linear
#  python contrastive.py --mode test --task disease --log_dir rand/disease --data ../data --evaluator linear --model_num $i
#done
#
for i in range 0 1 2 3 4 5 6 7 8 9 10
do
  python contrastive.py --mode pretrain --task heart --log_dir rand/heart --data ../heart  --train_prop .01 --epoch 0
  python contrastive.py --mode train --task heart --log_dir rand/heart --data ../heart  --train_prop 1.0 --epoch 1000 --evaluator linear
  python contrastive.py --mode test --task heart --log_dir rand/heart --data ../heart --evaluator linear --model_num $i
done

#for i in range 0 1 2 3 4 5 6 7 8 9 10
#do
#  python contrastive.py --mode pretrain --task heartchallenge --log_dir rand/heartchallenge --data ../heartchallenge  --train_prop 1 --epoch 0
#  python contrastive.py --mode train --task heartchallenge --log_dir rand/heartchallenge --data ../heartchallenge  --train_prop 1.0 --epoch 1000 --evaluator linear
#  python contrastive.py --mode test --task heartchallenge --log_dir rand/heartchallenge --data ../heartchallenge --evaluator linear --model_num $i
#done
