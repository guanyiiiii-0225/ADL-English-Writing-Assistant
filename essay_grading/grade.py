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
        "內容 (5分)": {
            "優": "主題(句)清楚切題，並有具體、完整的相關細節支持。(5-4 分)",
            "可": "主題不夠清楚或突顯，部分相關敘述發展不全。(3 分)",
            "差": "主題不明，大部分相關敘述發展不全或與主題無關。(2-1 分)",
            "劣": "文不對題或沒寫(凡文不對題或沒寫者，其他各項均以零分計算)。(0 分)",
        },
        "組織 (5分)": {
            "優": "重點分明，有開頭、發展、結尾，前後連貫，轉承語使用得當。(5-4 分)",
            "可": "重點安排不妥，前後發展比例與轉承語使用欠妥。(3 分)",
            "差": "重點不明、前後不連貫。(2-1 分)",
            "劣": "全文毫無組織或未按提示寫作。(0 分)",
        },
        "文法、句構 (4分)": {
            "優": "全文幾無文法錯誤，文句結構富變化。(4 分)",
            "可": "文法錯誤少，且未影響文意之表達。(3 分)",
            "差": "文法錯誤多，且明顯影響文意之表達。(2-1 分)",
            "劣": "全文文法錯誤嚴重，導致文意不明。(0 分)",
        },
        "字彙、拼字 (4分)": {
            "優": "用字精確、得宜，且幾無文法錯誤。(4 分)",
            "可": "字詞單調、重複，用字偶有不當，少許拼字錯誤，但不影響文意之表達。(3 分)",
            "差": "用字、拼字錯誤多，明顯影響文意之表達。(2-1 分)",
            "劣": "只寫出或抄襲與題意有關的零碎字詞。(0 分)",
        },
        "體例 (2分)": {
            "優": "格式、標點、大小寫幾無錯誤。(2 分)",
            "差": "格式、標點、大小寫等有錯誤，但不影響文意之表達。(1 分)",
            "劣": "違背基本的寫作體例或格式，標點、大小寫等錯誤甚多。(0 分)",
        },
    }
    prompt = f"你是一場考試中的英語作文評分者，請根據以下的評分標準對作文評分，將每個評分項目的分數分別輸出，並給予簡短評語：{json.dumps(criteria)}。\n以下是考試的英語作文題目：{problem}\n以下是要進行評分的作文內容：{content}。"

    return prompt


def inference(model, tokenizer, problem, content):
    json_schema = {
        "type": "object",
        "properties": {
            "分數": {
                "type": "object",
                "properties": {
                    "內容 (5分)": {"type": "number"},
                    "組織 (5分)": {"type": "number"},
                    "文法、句構 (4分)": {"type": "number"},
                    "字彙、拼字 (4分)": {"type": "number"},
                    "體例 (2分)": {"type": "number"},
                },
            },
            "評語": {"type": "string"},
        },
    }

    prompt = get_prompt(problem, content)
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
    generated_data = jsonformer()

    return generated_data


def grade(test_data_path):
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

    for x in dataset["test"]:
        output = inference(model, tokenizer, x["problem"], x["content"])
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        required=True,
        help="Path to test data.",
    )

    args = parser.parse_args()
    test_data_path = args.test_data_path

    inference()
