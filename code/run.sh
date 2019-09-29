python3 1.1.py --order 9 --point_num 100

python3 1.1.py --order 3 --point_num 10
python3 1.1.py --order 4 --point_num 10
python3 1.1.py --order 5 --point_num 10
python3 1.1.py --order 9 --point_num 10

python3 1.1.py --order 9 --point_num 15
python3 1.1.py --order 9 --point_num 100
python3 1.1.py --order 9 --point_num 200
python3 1.1.py --order 9 --point_num 500

python3 1.1.py --order 0 --point_num 10
python3 1.1.py --order 1 --point_num 10
python3 1.1.py --order 3 --point_num 10
python3 1.1.py --order 9 --point_num 10

python3 1.1.py --order 9 --point_num 10 --use_reg --lam -18
python3 1.1.py --order 9 --point_num 10 --use_reg --lam 0
python3 1.1.py --order 9 --point_num 10 --use_reg --lam -9
python3 1.1.py --order 9 --point_num 10 --use_reg --lam -3

python3 1.2.py --lr 0.1 --max_iter 20
python3 1.2.py --lr 0.01 --max_iter 200

python3 1.2.py --lr 0.1 --max_iter 20 --optimizer sgd --bs 50
python3 1.2.py --lr 0.1 --max_iter 20 --optimizer sgd --bs 200


python3 1.2.py --lr 0.1 --max_iter 20 --optimizer sgd --bs 400
python3 1.2.py --lr 0.2 --max_iter 10 --optimizer sgd --bs 400
python3 1.2.py --lr 0.5 --max_iter 10 --optimizer sgd --bs 400
python3 1.2.py --lr 0.2 --max_iter 10
python3 1.2.py --lr 0.001 --max_iter 2000
