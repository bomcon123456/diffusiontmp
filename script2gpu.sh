#! /bin/bash
accelerate launch --num_processes=2 --gpu_ids="0,1" --mixed_precision="fp16"   main.py \
  --train_data_dir="/lustre/scratch/client/guardpro/trungdt21/research/face_gen/truncated-diffusion-probabilistic-models/data/celeba/img_align_celeba_cropped" \
  --resolution=64 --random_flip \
  --output_dir="celeba64_trunc499_v2" \
  --train_batch_size=128 \
  --num_truncated=499 \
  --num_epochs=500 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention \
  --save_model_epochs=30 \
  --logger="tensorboard"
