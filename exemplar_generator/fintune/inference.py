from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, AutoTokenizer, GenerationConfig
import json
from trl import is_xpu_available

tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    base_model_name: Optional[str] = field(default="facebook/opt-350m", metadata={"help": "the model name"})
    test_data_path: Optional[str] = field(
        default="./dataset/ielts_writing_dataset.json", metadata={"help": "the dataset path"}
    )
    peft_path: Optional[str] = field(
        default="output", metadata={"help": "Path to the saved PEFT checkpoint"}
    )
    output_path: Optional[str] = field(default="output", metadata={"help": "the output file path"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})

def formatting_prompts_func(example):
    inputs = []
    for question in example['question']:
        system_prompt = "You are an artificial intelligence English writing assistant. The following are English composition test questions. You will provide answers that are useful, safe, detailed, and polite. Please provide examples of good writing for the following English essay questions. The narrative of the essay conforms to the answer instructions, the content is complete, and the organization is coherent; the grammar, sentence structure, word usage, and spelling are all good."
        text = f"<s>[INST]<<SYS>> {system_prompt} <</SYS>> {question}[/INST]"
        inputs.append(text)
    example['input'] = inputs
    return example

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Step 1: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
    )
    # Copy the model to each device
    device_map = (
        {"": f"xpu:{Accelerator().local_process_index}"}
        if is_xpu_available()
        else {"": Accelerator().local_process_index}
    )
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None

model = AutoModelForCausalLM.from_pretrained(
    script_args.base_model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_auth_token=script_args.use_auth_token
)
tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name)

# Step 2: Load the dataset
extension = script_args.test_data_path.split(".")[-1]
data_files = {}
data_files["test"] = script_args.test_data_path
dataset = load_dataset(extension, data_files=data_files, split="test")
dataset = dataset.map(formatting_prompts_func, batched=True)

# Step 3: Load LoRA
model = PeftModel.from_pretrained(model, script_args.peft_path)

# Step 5: Inference on test dataset
outputs_list = []
for i in tqdm(range(len(dataset))):
    input = dataset[i]['input']
    tokenized_input = tokenizer(input, return_tensors='pt', max_length=script_args.seq_length).to('cuda')
    outputs = model.generate(
        **tokenized_input,
        generation_config=GenerationConfig(
            do_sample=True,
            max_new_tokens=512,
            top_p=0.99,
            temperature=1e-8,
        )
    )
    outputs_list.append({
        "question": dataset[i]['question'],
        "output": tokenizer.decode(outputs[0], skip_special_tokens=True).split('[/INST]')[1]
    })

print(outputs_list)

# Step 6: Save the outputs
with open(script_args.output_path, 'w') as f:
    json.dump(outputs_list, f, ensure_ascii=False)



