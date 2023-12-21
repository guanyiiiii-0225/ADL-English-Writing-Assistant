from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
from image_to_text.get_translate import translate_image_to_text
import gc

def generate_image_description(images_paths, prompt="Can you describe this image in detail?"):
    
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
    
    description_arr = []
    for image_path in images_paths:
        #image = Image.open(image_path).convert('RGB')
        # convert the image to text
        #image_description = "This is an image."
        with torch.no_grad():
            image = Image.open(image_path).convert("RGB")
            model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.bfloat16)
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)
            outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
            )
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            image_description = generated_text
            image.save('./upload.png','png')
            translate_text = translate_image_to_text('./upload.png')
            if len(translate_text.strip()) > 0: 
                image_description += 'The text in this image: ' + translate_text
            print(image_description)
            description_arr.append(image_description)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    


    return description_arr

