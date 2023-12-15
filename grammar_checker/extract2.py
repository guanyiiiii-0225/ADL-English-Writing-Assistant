from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
import time
import streamlit as st
import pandas as pd

def extract_from_html():
    edge_options = Options()
    edge_options.add_argument('--headless')
    driver = webdriver.Edge(options=edge_options)
    # Need absolute address in this function
    driver.get('C:/Users/Linnn/Desktop/2023-ADL-Final/grammar_checker/1213_diff.html')

    td = driver.find_elements(By.CSS_SELECTOR, 'td.diff_header + td')
    data = ""

    for i in range(1, len(td), 2):
        e = td[i].get_attribute("innerHTML")
        data += e
        data += " "

    return data