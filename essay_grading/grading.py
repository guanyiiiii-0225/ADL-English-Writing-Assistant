from transformers import (
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


def get_prompt(problem: str, content: str):
    criteria = {
        "內容 (滿分 5 分)": {
            "5 或 4 分": "主題(句)清楚切題，並有具體、完整的相關細節支持。",
            "3 分": "主題不夠清楚或突顯，部分相關敘述發展不全。",
            "2 或 1 分": "主題不明，大部分相關敘述發展不全或與主題無關。",
            "0 分": "文不對題或沒寫。",
        },
        "組織 (滿分 5 分)": {
            "5 或 4 分": "重點分明，有開頭、發展、結尾，前後連貫，轉承語使用得當。",
            "3 分": "重點安排不妥，前後發展比例與轉承語使用欠妥。",
            "2 或 1 分": "重點不明、前後不連貫。",
            "0 分": "全文毫無組織或未按提示寫作。",
        },
        "文法、句構 (滿分 4 分)": {
            "4 分": "全文幾無文法錯誤，文句結構富變化。",
            "3 分": "文法錯誤少，且未影響文意之表達。",
            "2 或 1 分": "文法錯誤多，且明顯影響文意之表達。",
            "0 分": "全文文法錯誤嚴重，導致文意不明。",
        },
        "字彙、拼字 (滿分 4 分)": {
            "4 分": "用字精確、得宜，且幾無文法錯誤。",
            "3 分": "字詞單調、重複，用字偶有不當，少許拼字錯誤，但不影響文意之表達。",
            "2 或 1 分": "用字、拼字錯誤多，明顯影響文意之表達。",
            "0 分": "只寫出或抄襲與題意有關的零碎字詞。",
        },
        "體例 (滿分 2 分)": {
            "2 分": "格式、標點、大小寫幾無錯誤。",
            "1 分": "格式、標點、大小寫等有錯誤，但不影響文意之表達。",
            "0 分": "違背基本的寫作體例或格式，標點、大小寫等錯誤甚多。",
        },
    }
    prompt = f"你是一場考試中的英語作文評分者，這場考試的評分項目有「內容」、「組織」、「文法、句構」、「字彙、拼字」、「體例」五項，每個項目有不同的配分，評分標準如下：{json.dumps(criteria, ensure_ascii=False,  indent=4)}\n以下是這場考試的主題：{problem}\n以下是要進行評分的作文內容：\n{content}。\n\n請閱讀此作文，根據各評分項目的標準與配分給分。\n"

    print(prompt)

    return prompt

def inference_raw(model, tokenizer, problem, content):
    prompt = get_prompt(problem, content)
    tokenized_instruction = tokenizer(prompt, padding=True, return_tensors='pt').to('cuda')
    output = model.generate(**tokenized_instruction, 
                            generation_config=GenerationConfig(
                                do_sample=True,
                                max_new_tokens=512,
                                top_p=0.99,
                                temperature=0.5,
                            )
                        )

    output = tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, "").strip()
    
    return output

def inference(model, tokenizer, problem, content):
    json_schema = {
        "type": "object",
        "properties": {
            "分數": {
                "type": "object",
                "properties": {
                    "內容 (滿分 5 分)": {"type": "number"},
                    "組織 (滿分 5 分)": {"type": "number"},
                    "文法、句構 (滿分 4 分)": {"type": "number"},
                    "字彙、拼字 (滿分 4 分)": {"type": "number"},
                    "體例 (滿分 2 分)": {"type": "number"},
                },
            },
            # "評語": {"type": "string"},
        },
    }

    prompt = get_prompt(problem, content)
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt, temperature=1e-8)
    generated_data = jsonformer()

    return generated_data


def grade(test_data_path, output_path):
    model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"

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

    dataset = load_dataset("json", data_files={"test": test_data_path})

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant_config, torch_dtype=torch.bfloat16
    )

    model.eval()

    answers = []

    for x in dataset["test"]:
        output = inference(model, tokenizer, x["problem"], x["content"])
        print(output)
        answers.append(output)
    
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
        "--output_path",
        type=str,
        default="",
        required=True,
        help="Path to output.",
    )

    args = parser.parse_args()
    test_data_path = args.test_data_path
    output_path = args.output_path

    grade(test_data_path, output_path)
