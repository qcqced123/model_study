import requests
import pandas as pd
import os, sys, json, time, gc, random

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from typing import List
from tqdm.auto import tqdm
from multiprocessing import Pool
from api.crawler.amazon.run_file.run_api import asin_sub_loop
from api.crawler.amazon.today_deals.amazon_whole_foods_market import get_product_info


def crawl_link() -> List[str]:
    """ Crawl amazon coupons deals page, this function just get the next page link
    to crawl detailed product information

    Current time consuming is 5 minutes to get all the links now

    main logic is same as below:
        1) get the next page link
        2) load all of the information from dynamic loading page
        3) get the next link, and sales promotion information
    """
    url = "https://www.amazon.com/b?ie=UTF8&node=13213781011&pf_rd_p=2da01923-af16-4b6e-8ac4-721108132711&pf_rd_r=WPNWSTSZTA9771HT7TEX"
    chrome_driver_path = "/home/qcqced/chromedriver-linux64/chromedriver"

    options = webdriver.ChromeOptions()
    options.add_argument("headless")

    s = Service(chrome_driver_path)

    driver = webdriver.Chrome(service=s, options=options)
    driver.get(url)
    while True:
        try:
            show_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//*[@id="a-autoid-1"]'))
            )
            show_more_button.click()
            time.sleep(5)

        except:
            print("No more button to click")
            break

    selected_tags = driver.find_elements(By.XPATH, "//div[@data-claimurl]")
    promotion_link_list = ["https://www.amazon.com" + tag.get_attribute('data-claimurl') for tag in tqdm(selected_tags)]
    driver.quit()
    return promotion_link_list


def get_asin_list(url: str) -> str:
    """ get asin list from the promotion link page

    Args:
        url (str): promotion link url
    """
    sleep_time = random.uniform(0,2)
    time.sleep(sleep_time)

    chrome_driver_path = "/home/qcqced/chromedriver-linux64/chromedriver"
    options = webdriver.ChromeOptions()
    options.add_argument("headless")

    s = Service(chrome_driver_path)

    driver = webdriver.Chrome(service=s, options=options)
    driver.get(url)

    elements = driver.find_elements(By.CSS_SELECTOR, "li[name='productGrid'][data-asin]")
    asin = [element.get_attribute("data-asin") for element in elements]
    driver.quit()
    return asin


def main_loop() -> None:
    promotion_link_list = crawl_link()
    with Pool(processes=4) as pool:
        asin_list = pool.map(get_asin_list, promotion_link_list)

    get_product_info(asin_list)


if __name__ == '__main__':
    main_loop()

