# 2023-ADL-Final
## Install Chrome browser and download corresponding chromedriver
## Prepare Enviroment 
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

```
huggingface-cli login
```
and enter your hugginface token


## Run streamlit
```
streamlit run streamlit.py
```