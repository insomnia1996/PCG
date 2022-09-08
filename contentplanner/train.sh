CUDA_VISIBLE_DEVICES=0 python train.py\
    --model_name roberta-base\
    --crf_low_rank 64\
    --crf_beam_size 16\
    --batch_size_per_gpu 1\
    --gradient_accumulation_steps 16\
    --total_steps 40000\
    --print_every 100\
    --save_every 2000\
    --learning_rate 2e-5\
    --mle_loss_weight 0.5\
    --save_path_prefix ./ckpt/

