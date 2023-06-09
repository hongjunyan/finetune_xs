deepspeed --num_gpus=2 finetune_language_model.py \
--deepspeed ds_config_zero2.json \
--model_name_or_path NinedayWang/PolyCoder-160M \
--output_dir xs-poly-160M \
--dataset_name xscript \
--context_length 512 \
--do_train \
--fp16 \
--overwrite_cache \
--num_train_epochs 10 \
--logging_steps 3000 \
--warmup_steps 3000 \
--weight_decay 1e-01 \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 8 \
--learning_rate 5e-04