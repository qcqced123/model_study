import requests
import pandas as pd
import os, sys, json, time, gc, random

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
from typing import List
from tqdm.auto import tqdm
from multiprocessing import Pool
from api.crawler.target.tag_name import col2tag


def get_product_link(driver: webdriver.Chrome, curr) -> List[str]:
    """ get product detailed information from each sub-page
    this page will be called by with process of CPU
    """
    driver.get(curr)
    time.sleep(15)

    scroll_position = 0
    scroll_height = 1000
    while True:
        last_scroll_position = scroll_position
        driver.execute_script(f"window.scrollTo(0, {scroll_position + scroll_height});")
        time.sleep(1)
        scroll_position += scroll_height
        current_height = driver.execute_script("return document.body.scrollHeight")
        if scroll_position >= current_height or scroll_position == last_scroll_position:
            break

    product_links = driver.find_elements(By.CSS_SELECTOR, col2tag['product_link'])
    product_link_list = [tag.get_attribute('href') for tag in product_links]
    return product_link_list


def get_product_info(driver: webdriver.Chrome, url: str):
    driver.get(url)
    time.sleep(5)

    title = driver.find_element(By.CSS_SELECTOR, col2tag['product_name']).text
    current_price = driver.find_elements(By.CSS_SELECTOR, col2tag['red_current_price'])
    if not current_price:
        current_price = driver.find_element(By.CSS_SELECTOR, col2tag['black_current_price'])

    current_price = current_price[0].text if isinstance(current_price, list) else current_price.text

    before_price = driver.find_elements(By.CSS_SELECTOR, col2tag['before_price'])
    before_price = before_price[0].text if before_price else None

    try:
        blue_current_price = driver.find_element(By.CSS_SELECTOR, col2tag['blue_current_price'])
        if current_price and not before_price:
            before_price = current_price
            current_price = blue_current_price[0].text if isinstance(blue_current_price, list) else blue_current_price.text
    except: pass

    deals_state = driver.find_elements(By.CSS_SELECTOR, col2tag['deals_state'])
    if not deals_state:
        try:
            deals_state = driver.find_element(By.CSS_SELECTOR, col2tag['link_deals_state'])
        except: pass

    if deals_state:
        deals_state = deals_state[0].text if isinstance(deals_state, list) else deals_state.text

    else:
        deals_state = None

    return pd.DataFrame(
        [[title, current_price, before_price, deals_state]],
        columns=['product_name', 'current_price', 'before_price', 'deals_state']
    )


def main_loop():
    """
    implementations:
        1) get all links of promotion page: get_product_link()
        2) get the detailed information from each sub-page: get_product_info()
    """
    chrome_driver_path = "/Users/qcqced/chromedriver-mac-arm64/chromedriver"
    options = webdriver.ChromeOptions()
    options.add_argument("headless")

    s = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=s, options=options)

    url = "https://www.target.com/c/grocery-deals/-/N-k4uyq?type=products&moveTo=product-list-grid&Nao="
    links = []

    df = pd.DataFrame(columns=['product_name', 'current_price', 'before_price', 'deals_state'])
    for page in range(0, 1):
        pid = page * 24
        curr_url = url + str(pid)

        # convert to multi processing
        try:
            links.extend(get_product_link(driver, curr_url))

        except Exception as e:
            print(f"Error: {e}")
            break

    for link in tqdm(links):
        result = get_product_info(driver, link)
        df = pd.concat([df, result], axis=0, ignore_index=True)

    # with Pool(processes=4) as pool:
    #     results = pool.map(get_product_info, links)

    driver.quit()
    df.to_csv('target_deals.csv', index=False)
    return


if __name__ == '__main__':
    main_loop()
