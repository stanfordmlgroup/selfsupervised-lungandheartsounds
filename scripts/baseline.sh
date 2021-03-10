cd ../models ||exit

#for i in range 0 1 2 3 4 5 6 7 8 9 10
#do
#  python contrastive.py --mode pretrain --task crackle --log_dir baseline/crackle --data ../data  --train_prop .01 --epoch 0
#  python contrastive.py --mode train --task crackle --log_dir baseline/crackle --data ../data  --train_prop 1.0 --epoch 5000 --evaluator linear
#  python contrastive.py --mode test --task crackle --log_dir baseline/crackle --data ../data --evaluator linear --model_num $i
#done
#
#for i in range 0 1 2 3 4 5 6 7 8 9 10
#do
#  python contrastive.py --mode pretrain --task wheeze --log_dir baseline/wheeze --data ../data  --train_prop .01 --epoch 0
#  python contrastive.py --mode train --task wheeze --log_dir baseline/wheeze --data ../data  --train_prop 1.0 --epoch 5000 --evaluator linear
#  python contrastive.py --mode test --task wheeze --log_dir baseline/wheeze --data ../data --evaluator linear --model_num $i
#done



for i in range 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
do
  python contrastive.py --mode pretrain --task disease --log_dir baseline/disease --data ../data  --train_prop .01 --epoch 0
  python contrastive.py --mode train --task disease --log_dir baseline/disease --data ../data  --train_prop .1 --epoch 5000 --evaluator linear
  python contrastive.py --mode test --task disease --log_dir baseline/disease --data ../data --evaluator linear --model_num $i
done

for i in range 20 21 22 23 24 25 26 27 28 29
do
  python contrastive.py --mode pretrain --task disease --log_dir baseline/disease --data ../data  --train_prop .01 --epoch 0
  python contrastive.py --mode train --task disease --log_dir baseline/disease --data ../data  --train_prop .5 --epoch 5000 --evaluator linear
  python contrastive.py --mode test --task disease --log_dir baseline/disease --data ../data --evaluator linear --model_num $i
done
for i in range 30 31 32 33 34
do
  python contrastive.py --mode pretrain --task disease --log_dir baseline/disease --data ../data  --train_prop .01 --epoch 0
  python contrastive.py --mode train --task disease --log_dir baseline/disease --data ../data  --train_prop 1.0 --epoch 5000 --evaluator linear
  python contrastive.py --mode test --task disease --log_dir baseline/disease --data ../data --evaluator linear --model_num $i
done

for i in range 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
do
  python contrastive.py --mode pretrain --task heart --log_dir baseline/heart --data ../heart  --train_prop .01 --epoch 0
  python contrastive.py --mode train --task heart --log_dir baseline/heart --data ../heart  --train_prop .1 --epoch 5000 --evaluator linear
  python contrastive.py --mode test --task heart --log_dir baseline/heart --data ../heart --evaluator linear --model_num $i
done

for i in range 20 21 22 23 24 25 26 27 28 29
do
  python contrastive.py --mode pretrain --task heart --log_dir baseline/heart --data ../heart  --train_prop .01 --epoch 0
  python contrastive.py --mode train --task heart --log_dir baseline/heart --data ../heart  --train_prop .5 --epoch 5000 --evaluator linear
  python contrastive.py --mode test --task heart --log_dir baseline/heart --data ../heart --evaluator linear --model_num $i
done

for i in range 30 31 32 33 34
do
  python contrastive.py --mode pretrain --task heart --log_dir baseline/heart --data ../heart  --train_prop .01 --epoch 0
  python contrastive.py --mode train --task heart --log_dir baseline/heart --data ../heart  --train_prop 1.0 --epoch 5000 --evaluator linear
  python contrastive.py --mode test --task heart --log_dir baseline/heart --data ../heart --evaluator linear --model_num $i
done

#
#for i in range 0 1 2 3 4 5 6 7 8 9 10
#do
#  python contrastive.py --mode pretrain --task heartchallenge --log_dir baseline/heartchallenge --data ../heartchallenge  --train_prop 1 --epoch 0
#  python contrastive.py --mode train --task heartchallenge --log_dir baseline/heartchallenge --data ../heartchallenge  --train_prop 1.0 --epoch 5000 --evaluator linear
#  python contrastive.py --mode test --task heartchallenge --log_dir baseline/heartchallenge --data ../heartchallenge --evaluator linear --model_num $i
#done
