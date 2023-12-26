CUDA_VISIBLE_DEVICES=0 python exemplar_generator/fintune/sft.py \
    --model_name meta-llama/Llama-2-13b-chat-hf \
    --dataset_path exemplar_generator/fintune/dataset/ielts_writing_dataset_without_graph.json \
    --load_in_4bit \
    --use_peft \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --output_dir exemplar_generator/fintune/output/without_graph_2 \