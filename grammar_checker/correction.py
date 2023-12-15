from transformers import pipeline
import time

def grammar_correction(input):
    start = time.time()
    corrector = pipeline(
                'text2text-generation',
                'pszemraj/flan-t5-large-grammar-synthesis',
                )
    results = corrector(input)
    print(time.time() - start)
    return results[0]['generated_text']

if __name__ == '__main__':
    raw_text = 'I hate wet and reiny days. It rained a lot in 1816.... a lot - like everyday; the weather in Europe was abnormally wet because it rained in Switzerland on 130 out of the 183 days from April to September. If I was Mary Shelley I might decide to write a book too. Afterall, it was the onnly thing you could do without TV or anything. She said that she "passed the summer of 1816 in the environs of Geneva...we occasionally amused ourselves with some German stories of ghosts... These tales excited in us a playful desire of imitation"  So, people were stuck inside and bored. Mary Shelley decided to write a book becuase it was so awful outside. I can totally see her point, you know? I guess I would write a novel if there was nothing else to do.'
    output = grammar_correction(raw_text)
    print(output)