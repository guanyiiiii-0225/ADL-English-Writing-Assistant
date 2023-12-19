from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests

def generate_image_description(image_path, prompt="Can you describe this image in detail?"):
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.bfloat16)
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return generated_text

# Example usage:
image_path = "./103_4.png"
generated_description = generate_image_description(image_path)
print(generated_description)
