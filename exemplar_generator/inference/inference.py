from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch

model = "meta-llama/Llama-2-13b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.pad_token_id = tokenizer.eos_token_id

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model, quantization_config=bnb_config, device_map="auto"
)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
system_prompt = "You are an artificial intelligence English writing assistant. The following are English composition test questions. You will provide answers that are useful, safe, detailed, and polite. Please provide examples of good writing for the following English essay questions. The narrative of the essay conforms to the answer instructions, the content is complete, and the organization is coherent; the grammar, sentence structure, word usage, and spelling are all good."
user_message = "Instructions: Follow the prompts and write an English composition of at least 120 words. Tip: As a member of Taiwan, what makes you most proud of Taiwan? Please write an English essay with this title, talking about two aspects or things in Taiwan that make you most proud (for example: people, things, things, culture, systems, etc.). The first paragraph describes these two aspects or things and explains why they make you proud; the second paragraph explains how you think these Taiwanese characteristics can be introduced or marketed so that the world can better understand Taiwan."
text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]\n"

sequences = pipeline(
    text,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=1000,
)
for seq in sequences:
    print(seq['generated_text'].replace(text, ''))
