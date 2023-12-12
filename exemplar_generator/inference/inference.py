from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import json


def get_args_parser():
    parser = argparse.ArgumentParser("Llama-2", add_help=False)
    parser.add_argument(
        "--model_name",
        default="meta-llama/Llama-2-13b-chat-hf",
        type=str,
        required=True,
    )
    parser.add_argument("--input_file", default="", type=str, required=True)
    parser.add_argument("--output_file", default="", type=str, required=True)

    return parser


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config, device_map="auto"
    )
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    with open(args.input_file, "r") as input_file:
        input = json.load(input_file)

    output = []
    for test in tqdm(input):
        system_prompt = "You are an artificial intelligence English writing assistant. The following are English composition test questions. You will provide answers that are useful, safe, detailed, and polite. Please directly provide good writing examples for the following English essay questions without any other irrelevant output. The narrative of the essay conforms to the answer instructions, the content is complete, and the organization is coherent; the grammar, sentence structure, word usage, and spelling are all good."
        user_message = test["question"]
        text = (
            f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]\n"
        )

        sequences = pipeline(
            text,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=1024,
        )

        result = test
        result["answer"] = sequences[0]["generated_text"].replace(text, "")
        output.append(result)

    with open(args.output_file, "w") as output_file:
        json.dump(output, output_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Llama-2", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_file:
        Path(os.path.dirname(args.output_file)).mkdir(parents=True, exist_ok=True)
    main(args)
