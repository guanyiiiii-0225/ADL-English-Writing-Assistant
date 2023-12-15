from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
import time
import streamlit as st
import pandas as pd

edge_options = Options()
edge_options.add_argument('--headless')
driver = webdriver.Edge(options=edge_options)
driver.get('C:/Users/Linnn/Desktop/2023-ADL-Final/linnn/1213_diff.html')

td = driver.find_elements(By.CSS_SELECTOR, 'td.diff_header + td')
# print(len(td))

table_data = {
    'Your Essay:': [],
    'Modified Essay:': [],
}

for i in range(0, len(td), 2):
    e1 = td[i].get_attribute("innerHTML")
    e2 = td[i+1].get_attribute("innerHTML")
    table_data['Your Essay:'].append(e1)
    table_data['Modified Essay:'].append(e2)

# print(table_data)
st.set_page_config(layout="wide")

st.markdown('''<style type="text/css">
        table.diff {font-family:Courier; border:medium;}
        .diff_header {background-color:#e0e0e0}
        td.diff_header {text-align:right}
        .diff_next {background-color:#c0c0c0}
        .diff_add {background-color:#aaffaa}
        .diff_chg {background-color:#ffff77}
        .diff_sub {background-color:#ffaaaa}
    </style>''', unsafe_allow_html=True)

df = pd.DataFrame(table_data)
st.write(df.to_html(escape=False), unsafe_allow_html=True)