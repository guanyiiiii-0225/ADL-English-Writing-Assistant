from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from datasets import load_dataset
import torch
import argparse
import json
from jsonformer import Jsonformer
import re

torch.set_printoptions(threshold=10_000)

def get_system_prompt():
    criteria = {
        "Content (out of 5 points)": {
            "5 or 4": "Clear subject(s) and backed by specific, complete related details.",
            "3": "The subject is not clear or emphasized, and some related descriptions are not fully developed.",
            "2 or 1": "The subject is unclear, most related descriptions are undeveloped or unrelated to the main point.",
            "0": "Off-topic or not written at all.",
        },
        "Organization (out of 5 points)": {
            "5 or 4": "Key points are clear, with an introduction, development, and conclusion. The transitions are used properly.",
            "3": "Key points are not well arranged, and the use of development examples and transitions is inappropriate.",
            "2 or 1": "Key points are unclear, and there's a lack of coherence.",
            "0": "The entire text lacks organization, or did not follow the prompts for writing.",
        },
        "Grammar, Sentence Structure (out of 4 points)": {
            "4": "Almost no grammatical errors in the text, and the sentence structure varies.",
            "3": "Few grammatical errors that do not affect the expression of the text's meaning.",
            "2 or 1": "Many grammar errors that significantly affect the expression of the text's meaning.",
            "0": "Severe grammar errors throughout the text, making the meaning unclear.",
        },
        "Vocabulary, Spelling (out of 4 points)": {
            "4": "Precise and appropriate word choice, with almost no grammatical errors.",
            "3": "Monotonous/repetitive vocabulary, occasional inappropriate word choice, few spelling errors that do not affect the expression of the text's meaning.",
            "2 or 1": "Many errors in word choice and spelling that significantly affect the text's meaning.",
            "0": "Only wrote or plagiarized fragmented words related to the topic.",
        },
        "Format (out of 2 points)": {
            "2": "The format, punctuation, and capitalization are nearly flawless.",
            "1": "Errors in the format, punctuation, and capitalization, but they do not affect the expression of the text's meaning.",
            "0": "Violates basic writing format or conventions, with numerous errors in punctuation, capitalization, and so on.",
        },
    }
    criteria_string = json.dumps(criteria, ensure_ascii=False,  indent=4)
    criteria_formatted = re.sub('[\"{},]', "", criteria_string)
    prompt = f"You are an English essay grader in an examination.\nThe scoring items in this exam are 'Content', 'Organization', 'Grammar, Sentence Structure', 'Vocabulary, Spelling', and 'Format'. Each item has different scores, and the scoring standards are as follows: {criteria_formatted}.\n"

    return prompt

def get_example_prompt(problem: str, content: str, output: str):
    prompt = f"[INST]The following is the topic of this exam: {problem}\nThe following is the essay content to be graded:\n{content}.\n\nPlease read this essay and score each grading item: [/INST]\n{output}</s>\n"
    # print(prompt)
    return prompt

def get_user_prompt(problem: str, content: str):
    prompt = f"[INST]The following is the topic of this exam: {problem}\nThe following is the essay content to be graded:\n{content}.\n\nPlease read this essay and score each grading item: [/INST]\n"
    return prompt

def get_prompt(problem, content, examples):
    example_indices = [0, 1, 2]
    prompt = f"<s>[INST] <<SYS>>{get_system_prompt()}<</SYS>>[/INST]{''.join([get_example_prompt(examples[ind]['problem'], examples[ind]['content'], examples[ind]['output']) for ind in example_indices])}{get_user_prompt(problem, content)}"
    print(prompt)
    return prompt

def inference_raw(pipe, tokenizer, problem, content, examples):
    prompt = get_prompt(problem, content, examples)
    sequences = pipe(
        prompt,
        do_sample=True,
        top_p=0.99,
        temperature=1e-8,
        num_return_sequences=1,
        # eos_token_id=tokenizer.eos_token_id,
        # pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=1024
    )

    output = sequences[0]["generated_text"].rsplit('[/INST]')[-1].strip()
    
    return output

# def inference(model, tokenizer, problem, content, examples):
#     json_schema = {
#         "type": "object",
#         "properties": {
#             "內容 (滿分 5 分)": {
#                 "type": "object",
#                 "properties": {
#                     "分數": {"type": "number"},
#                     "評語": {"type": "string"},
#                 }
#             },
#             "組織 (滿分 5 分)":  {
#                 "type": "object",
#                 "properties": {
#                     "分數": {"type": "number"},
#                     "評語": {"type": "string"},
#                 }
#             },
#             "文法、句構 (滿分 4 分)":  {
#                 "type": "object",
#                 "properties": {
#                     "分數": {"type": "number"},
#                     "評語": {"type": "string"},
#                 }
#             },
#             "字彙、拼字 (滿分 4 分)":  {
#                 "type": "object",
#                 "properties": {
#                     "分數": {"type": "number"},
#                     "評語": {"type": "string"},
#                 }
#             },
#             "體例 (滿分 2 分)":  {
#                 "type": "object",
#                 "properties": {
#                     "分數": {"type": "number"},
#                     "評語": {"type": "string"},
#                 }
#             },
#         },
#     }

#     prompt = get_prompt(problem, content, examples)
#     jsonformer = Jsonformer(model, tokenizer, json_schema, prompt, temperature=0.01)
#     generated_data = jsonformer()

#     return generated_data


def grade(test_data_path, example_data_path, output_path):
    model_name = "togethercomputer/LLaMA-2-7B-32K"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    # generation_config = GenerationConfig(
    #     do_sample=True,
    #     max_new_tokens=512,
    #     top_p=0.99,
    #     temperature=1e-8,
    # )

    dataset = load_dataset("json", data_files={"test": test_data_path, "example": example_data_path})

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant_config, torch_dtype=torch.bfloat16
    )

    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    answers = []

    for x in dataset["example"]:
        # output = inference(model, tokenizer, x["problem"], x["content"], dataset["example"])
        output = inference_raw(pipe, tokenizer, x["problem"], x["content"], dataset["example"])
        print(output)
        print('')
        answers.append({
            "problem": x["problem"],
            "content": x["content"],
            "output": output
        })
    
    of = open(output_path, 'w', encoding='utf8')
    json.dump(answers, of, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        required=True,
        help="Path to test data.",
    ) 
    parser.add_argument(
        "--example_data_path",
        type=str,
        default="",
        required=True,
        help="Path to example data.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        required=True,
        help="Path to output.",
    )

    args = parser.parse_args()
    test_data_path = args.test_data_path
    example_data_path = args.example_data_path
    output_path = args.output_path

    grade(test_data_path, example_data_path, output_path)
