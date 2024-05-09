import requests
import pandas as pd
import os, sys, json, time, gc

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait as wait
from typing import List
from tqdm.auto import tqdm
from multiprocessing import Pool
from api.crawler.amazon.run_file.run_api import asin_sub_loop


def crawl_asin() -> List[str]:
    """ Start crawling loop function, return asin list in amazon unique product code
    This method is used to get the ASIN code from the Amazon Whole Foods Market Deals Page
    """
    url = "https://www.amazon.com/fmc/alldeals?almBrandId=VUZHIFdob2xlIEZvb2Rz&ref=wf_dsk_sn_deals-4c0a5"
    chrome_driver_path = "/home/qcqced/chromedriver-linux64/chromedriver"
    s = Service(chrome_driver_path)
    options = webdriver.ChromeOptions()
    options.add_argument("headless")

    driver = webdriver.Chrome(service=s, options=options)
    driver.get(url)

    selected_tags = driver.find_elements(By.CSS_SELECTOR, 'div._subaru-desktop_gridDesktopStyle_gridCell__3gkoz')
    asin_list = [tag.get_attribute('id').split('-')[1] for tag in tqdm(selected_tags)]

    driver.quit()
    return asin_list


def get_product_info(asin_list: List[str]):
    """ get detailed product information by asin code

    Args:
        asin_list (List[str]): asin code list from crawl_asin function
    """
    with Pool(processes=4) as pool:
        results = pool.map(asin_sub_loop, asin_list)

    return results


def main_loop():
    asin_list = crawl_asin()
    results = get_product_info(asin_list)
    return results


if __name__ == '__main__':
    print(main_loop())
