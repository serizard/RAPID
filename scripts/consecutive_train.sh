# chunk_size: [30,40,50,60,70]
# num_token: [150,200,250]

# chunk_size=30
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 30 --num_token 120 --phase train_test --gpu 0 --use_gpu --save chk30_nt120
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 30 --num_token 150 --phase train_test --gpu 0 --use_gpu --save chk30_nt150
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 30 --num_token 180 --phase train_test --gpu 0 --use_gpu --save chk30_nt180
# chunk_size=40
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 40 --num_token 120 --phase train_test --gpu 0 --use_gpu --save chk40_nt120
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 40 --num_token 150 --phase train_test --gpu 0 --use_gpu --save chk40_nt150
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 40 --num_token 180 --phase train_test --gpu 0 --use_gpu --save chk40_nt180
# chunk_size=50
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 50 --num_token 120 --phase train_test --gpu 0 --use_gpu --save chk50_nt120
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 50 --num_token 150 --phase train_test --gpu 0 --use_gpu --save chk50_nt150
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 50 --num_token 180 --phase train_test --gpu 0 --use_gpu --save chk50_nt180
# chunk_size=60
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 60 --num_token 120 --phase train_test --gpu 0 --use_gpu --save chk60_nt120
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 60 --num_token 150 --phase train_test --gpu 0 --use_gpu --save chk60_nt150
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 60 --num_token 180 --phase train_test --gpu 0 --use_gpu --save chk60_nt180
# chunk_size=70
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 70 --num_token 120 --phase train_test --gpu 0 --use_gpu --save chk70_nt120
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 70 --num_token 150 --phase train_test --gpu 0 --use_gpu --save chk70_nt150
CUDA_VISIBLE_DEVICES=0 python main.py --y_col type_label --update --edge_weight --graphuse --train_gender both --chunk_size 70 --num_token 180 --phase train_test --gpu 0 --use_gpu --save chk70_nt180
