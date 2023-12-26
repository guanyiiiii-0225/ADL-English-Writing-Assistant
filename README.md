# 2023-ADL-Final
## Project Structure/Folder Structure
```
___
 ├─ essay_grading: folder for Content Scoring
 ├─ exampler_generator: folder for Example Essay Generation
     ├─ finetune: folder for fine-tuning llama model
     └─ inference: folder for inference llama model without fine-tuning
 ├─ grammar_checker: folder for Grammar Correction and Paraphrase
 ├─ image_to_text: folder for Image-to-Text Conversion
 └─ streamlit.py: python file to set up streamlit site
```

## Prepare Enviroment 
### Create environment and install package
```
conda create --name adl-final python=3.11.0
conda activate adl-final
```

```
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

```
pip install -r requirements.txt
sudo apt install xclip
```
### Huggingface token login
```
huggingface-cli login
```
and enter your hugginface token

### Setup Chrome browser and chromedriver
Install Chrome browser and download corresponding chromedriver: https://chromedriver.chromium.org/


## Run streamlit
```
streamlit run streamlit.py --server.port 8080
```

go to http://localhost:8080/ and see the website!