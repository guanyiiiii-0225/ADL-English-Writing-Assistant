CUDA_VISIBLE_DEVICES=0 python inference.py \
    --base_model_name meta-llama/Llama-2-13b-chat-hf \
    --peft_path ./output/without_graph_2 \
    --test_data_path ./dataset/test_dataset.json \
    --output_path ./prediction.json