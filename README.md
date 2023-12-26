# 2023-ADL-Final
## Project description
English composition presents a formidable challenge for students. In response, we present an English Writing Assistant, designed to improve high school students' English writing skills for the General Scholastic Ability Test. Addressing the challenge of English composition, the tool features two main functions: generating exemplary essays using the InstructBLIP Image-to-Text Conversion system integrated with the Llama language model, and critiquing student essays with the T5 model, the bart-paraphrase model, and the Chinese alpaca model. This innovative approach not only provides students with model essays but also offers personalized feedback, significantly enhancing their English writing abilities.

In the usage scenario we've designed, users can choose from two options: (1) Essay Generation Workflow, where they generate sample essays based on specified essay prompts; and (2) Essay Correction Workflow, where we offer grading and advice on essays composed by the users. The figure below illustrates the structural framework of our English Writing Assistant.

<img src='framework.png' width='60%'>

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
