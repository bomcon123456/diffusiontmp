#! /bin/bash
accelerate launch --mixed_precision="fp16"   main.py \
  --dataset_name="cifar10" \
  --resolution=32 --random_flip \
  --output_dir="test-cifar10" \
  --train_batch_size=256 \
  --num_epochs=500 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention \
  --save_model_epochs=30 \
  --logger="tensorboard"