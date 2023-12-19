from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch

model_name = "meta-llama/Llama-2-13b-chat-hf"


def generate_writing_without_finetune(guide):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    system_prompt = ""
    user_message = guide
    text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]"

    sequences = pipeline(
        text,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=2048,
    )

    return sequences[0]["generated_text"].replace(text, "")


if __name__ == "__main__":
    guide = "說明︰依提示寫一篇英文作文，文長至少120個單詞（words）。提示：近年來，很多大學鼓勵教授以英語講授專業課程，請寫一篇英文作文，說明你對這個現象的看法。文分兩段，第一段說明你是否認同這個趨勢並陳述理由；第二段說明如果你未來就讀的大學必修課是以英語授課，你將會如何因應或規劃。"
    result = generate_writing_without_finetune(guide)
    print(result)
