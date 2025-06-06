accelerate launch diffusers/examples/textual_inversion/textual_inversion.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="../../../data/BUSI_flattened" \
  --placeholder_token="<ultrasound_tumor>" \
  --initializer_token="ultrasound" \
  --learnable_property="object" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=2000 \
  --learning_rate=5e-4 \
  --output_dir="../../../textual_inversion_output" \
  --mixed_precision="fp16"
