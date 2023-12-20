import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import cv2
import subprocess

'''
option = webdriver.ChromeOptions()
option.add_experimental_option('excludeSwitches', ['enable-automation']) 
option.add_argument('--incognito')
option.add_experimental_option("prefs", {"profile.content_settings.exceptions.clipboard": {"[*.]httpbin.org,*": {'last_modified': (time.time()*1000), 'setting': 2}}})
option.use_chromium = True
option.add_argument("--disable-blink-features=AutomationControlled")
option.add_argument("--enable-features=ClipboardContentSetting")
option.add_argument('-headless')

# 初始化 Chrome 瀏覽器
driver = webdriver.Chrome(options=option)
# 前往 Google 翻譯網站
driver.get("https://translate.google.com/?hl=zh-TW")
#driver.minimize_window()


# 點擊「圖片翻譯」按鈕
time.sleep(1) 
image_translate_button = driver.find_element(By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[1]/nav/div[2]/div/button/span')
image_translate_button.click()


# 上傳圖片
image_path = '110.jpg'
upload = driver.find_element(By.CSS_SELECTOR, 'input[accept*="png"]')
element_html = upload.get_attribute('outerHTML')
print("Element HTML:")
print(element_html)
time.sleep(1) 
upload.send_keys(os.path.abspath(image_path))


# 點擊「中文翻譯」按鈕
time.sleep(1) 
chinese_button = driver.find_element(By.CSS_SELECTOR, '#i58 > span.VfPpkd-YVzG2b')
chinese_button.click()

# 等待翻譯結果出現
time.sleep(1)  
copy_button = driver.find_element(By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[5]/c-wiz/div[2]/c-wiz/div/div[1]/div[2]/div[1]/button/span[2]')
copy_button.click()


def fetch_clipboard_contents():
    command = "xclip -selection clipboard -o"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, _ = process.communicate()
    return output.decode("utf-8").strip()

copied_text = fetch_clipboard_contents()
print(copied_text)

# 關閉瀏覽器
driver.quit()

'''

def translate_image_to_text(image_path):
    option = webdriver.ChromeOptions()
    option.add_experimental_option('excludeSwitches', ['enable-automation']) 
    option.add_argument('--incognito')
    option.add_experimental_option("prefs", {"profile.content_settings.exceptions.clipboard": {"[*.]httpbin.org,*": {'last_modified': (time.time()*1000), 'setting': 2}}})
    option.use_chromium = True
    option.add_argument("--disable-blink-features=AutomationControlled")
    option.add_argument("--enable-features=ClipboardContentSetting")
    #option.add_argument('-headless')

    # Initialize Chrome browser
    driver = webdriver.Chrome(options=option)
    driver.get("https://translate.google.com/?hl=zh-TW")
    
    # Click "Image Translate" button
    time.sleep(1) 
    image_translate_button = driver.find_element(By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[1]/nav/div[2]/div/button/span')
    image_translate_button.click()

    # Upload image
    upload = driver.find_element(By.CSS_SELECTOR, 'input[accept*="png"]')
    time.sleep(1) 
    print(image_path)
    upload.send_keys(os.path.abspath(image_path))

    # Click "Chinese Translate" button
    time.sleep(1) 
    chinese_button = driver.find_element(By.CSS_SELECTOR, '#i58 > span.VfPpkd-YVzG2b')
    chinese_button.click()

    # Wait for translation result to appear
    time.sleep(5)  
    copy_button = driver.find_element(By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[5]/c-wiz/div[2]/c-wiz/div/div[1]/div[2]/div[1]/button/span[2]')
    copy_button.click()

    # Fetch copied text from clipboard
    def fetch_clipboard_contents():
        command = "xclip -selection clipboard -o"
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        output, _ = process.communicate()
        return output.decode("utf-8").strip()

    copied_text = fetch_clipboard_contents()
    print(copied_text)

    # Close the browser
    driver.quit()
    
    return copied_text



