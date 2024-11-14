# chunk_size: [20,30,40,50,60,70,80]
# num_token: [150,200,250]
# stdmult: [1.0,1.5]

# chunk_size=20
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 20 --num_token 100 --phase train_test --gpu 0 --use_gpu --save chk20_nt100
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 20 --num_token 150 --phase train_test --gpu 0 --use_gpu --save chk20_nt150
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 20 --num_token 200 --phase train_test --gpu 0 --use_gpu --save chk20_nt200
# chunk_size=30
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 30 --num_token 100 --phase train_test --gpu 0 --use_gpu --save chk30_nt100
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 30 --num_token 150 --phase train_test --gpu 0 --use_gpu --save chk30_nt150
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 30 --num_token 200 --phase train_test --gpu 0 --use_gpu --save chk30_nt200
# chunk_size=40
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 40 --num_token 100 --phase train_test --gpu 0 --use_gpu --save chk40_nt100
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 40 --num_token 150 --phase train_test --gpu 0 --use_gpu --save chk40_nt150
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 40 --num_token 200 --phase train_test --gpu 0 --use_gpu --save chk40_nt200
# chunk_size=50
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 50 --num_token 100 --phase train_test --gpu 0 --use_gpu --save chk50_nt100
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 50 --num_token 150 --phase train_test --gpu 0 --use_gpu --save chk50_nt150
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 50 --num_token 200 --phase train_test --gpu 0 --use_gpu --save chk50_nt200
# chunk_size=60
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 60 --num_token 100 --phase train_test --gpu 0 --use_gpu --save chk60_nt100
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 60 --num_token 150 --phase train_test --gpu 0 --use_gpu --save chk60_nt150
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 60 --num_token 200 --phase train_test --gpu 0 --use_gpu --save chk60_nt200
# chunk_size=70
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 70 --num_token 100 --phase train_test --gpu 0 --use_gpu --save chk70_nt100
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 70 --num_token 150 --phase train_test --gpu 0 --use_gpu --save chk70_nt150
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 70 --num_token 200 --phase train_test --gpu 0 --use_gpu --save chk70_nt200
# chunk_size=80
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 80 --num_token 100 --phase train_test --gpu 0 --use_gpu --save chk80_nt100
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 80 --num_token 150 --phase train_test --gpu 0 --use_gpu --save chk80_nt150
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 80 --num_token 200 --phase train_test --gpu 0 --use_gpu --save chk80_nt200