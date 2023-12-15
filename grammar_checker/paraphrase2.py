from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def paraphrase(input):
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model = model.to(device)
    # sentence = "This is something which i cannot understand at all"
    # sentence = 'In the modern world, technology plays an crucial role in our everyday life. It have changed the way we communicate, work, and even entertain ourselves. One of the most important invention is the Internet, which allow us to connect with people around the world.'

    text =  "paraphrase: " + input + " </s>"

    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=1
    )
    output_list = []
    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        output_list.append(line)
    
    return output_list