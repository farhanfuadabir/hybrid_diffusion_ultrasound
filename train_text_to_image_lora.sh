accelerate launch diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="data/BUSI_flattened" \
  --image_column="image" \
  --caption_column="text" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=3000 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --output_dir="./lora_output" \
  --rank=4 \
  --mixed_precision="fp16"


# accelerate launch diffusers/examples/text_to_image/train_text_to_image_lora.py \
#   --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#   --train_data_dir="./data/BUSI" \
#   --caption_column="text" \
#   --resolution=512 \
#   --train_batch_size=4 \
#   --gradient_accumulation_steps=1 \
#   --max_train_steps=3000 \
#   --learning_rate=1e-4 \
#   --lr_scheduler="constant" \
#   --output_dir="./lora_output" \
#   --rank=4 \
#   --mixed_precision="fp16"


# accelerate launch diffusers/examples/text_to_image/train_text_to_image_lora.py \
#   --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
#   --dataset_name="json" \
#   --train_data_dir="data/BUSI/captions.jsonl" \
#   --image_column="image" \
#   --caption_column="text" \
#   --resolution=512 \
#   --train_batch_size=4 \
#   --gradient_accumulation_steps=1 \
#   --max_train_steps=3000 \
#   --learning_rate=1e-4 \
#   --lr_scheduler=constant \
#   --output_dir=./lora_output \
#   --rank=4 \
#   --mixed_precision=fp16

# accelerate launch train_text_to_image.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_name=$dataset_name \
#   --use_ema \
#   --resolution=512 --center_crop --random_flip \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --gradient_checkpointing \
#   --mixed_precision="fp16" \
#   --max_train_steps=15000 \
#   --learning_rate=1e-05 \
#   --max_grad_norm=1 \
#   --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --output_dir="lora_output" 