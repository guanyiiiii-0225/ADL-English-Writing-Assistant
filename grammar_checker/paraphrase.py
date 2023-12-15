import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import time

def paraphrase(input):
    start = time.time()
    model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')
    batch = tokenizer(input, return_tensors='pt', padding=True, truncation=True)
    generated_ids = model.generate(batch['input_ids'])
    generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(time.time() - start)
    return generated_sentence

if __name__ == '__main__':
    raw_text = ["I am cute and I am pretty.", "I am good and I am handsome.", "I am good and I am handsome.", "I am good and I am handsome."]
    output = paraphrase(raw_text)
    print(output[0])
    print(output[1])