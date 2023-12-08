python exemplar_generator/fintune/sft.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_path exemplar_generator/fintune/dataset/ielts_writing_dataset_without_graph.json \
    --load_in_4bit \
    --use_peft \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \