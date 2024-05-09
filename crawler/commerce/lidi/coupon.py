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
from api.crawler.lidi.tag_name import col2tag


def main_loop():
    """
    implementations:
        1) get all links of promotion page: get_product_link()
        2) get all links of product page
        3) get all the detailed info of each product
    """
    chrome_driver_path = "/Users/qcqced/chromedriver-mac-arm64/chromedriver"
    s = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=s)

    url = "https://www.lidl.com/rewards-coupons"

    driver.get(url)
    time.sleep(10)

    # for scrolling down
    body = driver.find_element(By.TAG_NAME, 'body')
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        body.send_keys(Keys.END)
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            break
        last_height = new_height

    # get all the links of the promotion pages
    promotion_links = driver.find_elements(By.CSS_SELECTOR, col2tag['promotion_link'])
    promotion_list = [link.get_attribute('href') for link in promotion_links]

    # get all the links of the product pages
    product_list = []
    for link in promotion_list[0:5]:
        driver.get(link)
        time.sleep(2)
        try:
            product_list.append(driver.find_element(By.CSS_SELECTOR, col2tag['product_link']).get_attribute('href'))

        except:
            product_list.extend([element.get_attribute('href') for element in driver.find_elements(By.CSS_SELECTOR, col2tag['multi_product_link'])])

    print(product_list)
    print(len(product_list))

    # get all of the detailed information about each product
    for product_link in product_list[0:5]:
        driver.get(product_link)
        time.sleep(2)

        product_name = driver.find_element(By.CSS_SELECTOR, col2tag['product_name']).get_attribute('aria-label')
        before_price = driver.find_element(By.CSS_SELECTOR, col2tag['before_price']).text
        deals_state = driver.find_element(By.CSS_SELECTOR, col2tag['deals_state']).text
        expire_date = driver.find_element(By.CSS_SELECTOR, col2tag['expire_date']).text


if __name__ == '__main__':
    main_loop()


