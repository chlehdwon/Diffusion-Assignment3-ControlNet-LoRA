export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./runs/small_animal"   # output directory of each run

accelerate launch train.py \
--seed=0 \
--pretrained_model_name_or_path=$MODEL_DIR \
--output_dir=$OUTPUT_DIR \
--dataset_name=JFoz/dog-poses-controlnet-dataset \
--resolution=512 \
--learning_rate=1e-5 \
--validation_image "./data/conditioning_image_small1.jpg" "./data/conditioning_image_small2.jpg" \
--validation_prompt "a puppy jumping up a tree" "a golden retriever dog jumps in the air" \
--train_batch_size=1 \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--set_grads_to_none \
--use_8bit_adam \
--checkpoints_total_limit 2 \
--validation_steps 100 \
--report_to "tensorboard" \
--image_column "original_image" \
--caption_column "caption"