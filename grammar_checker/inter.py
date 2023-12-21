from transformers import pipeline
import time
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import difflib

def find_diff(text1, text2, filename):
    text1 = text1.split('.')
    text1 = [(x + '.').strip() for x in text1]
    text1 = text1[:-1]

    text2 = text2.split('.')
    text2 = [(x + '.').strip() for x in text2]
    text2 = text2[:-1]


    differ = difflib.HtmlDiff()
    html_diff = differ.make_file(text1, text2)

    with open(filename, "w", encoding="utf-8") as file:
        file.write(html_diff)

def grammar_correction(input):
    start = time.time()
    corrector = pipeline(
                'text2text-generation',
                'pszemraj/flan-t5-large-grammar-synthesis',
                )
    results = corrector(input)
    print(time.time() - start)
    return results[0]['generated_text']

def paraphrase(input):
    start = time.time()
    model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')
    batch = tokenizer(input, return_tensors='pt', padding=True, truncation=True)
    batch = batch.to(device)
    generated_ids = model.generate(batch['input_ids'])
    generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(time.time() - start)
    return generated_sentence

def generate_html(raw_text):
    grammar_output = grammar_correction(raw_text)
    grammar_output = grammar_output.split('.')
    grammar_output = [x + '.' for x in grammar_output]
    grammar_output = grammar_output[:-1]
    print(grammar_output)
    paraphrase_output = paraphrase(grammar_output)
    print(paraphrase_output)
    p_out = ''
    for item in paraphrase_output:
        p_out = p_out + item
        if(p_out[-1] != '.'):
            p_out += '.'
    print(p_out)
    find_diff(raw_text, p_out, "./grammar_checker/1213_diff.html")