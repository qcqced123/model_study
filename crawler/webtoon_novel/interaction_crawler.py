import pandas as pd
import os, sys, json, time, gc
import numpy as np
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait as wait

'''
[Test Demo]
This is Source Code for Scrapping Comments(Interactions)
1) Comments
    - User_Name
    - Posted Time
    - WebToon Title

2) Collect Range => 22.08.01 ~ 22.11.20

3) Target Platform => Naver WebToon

4) Design Style => Function & For-Loop

5) Variable Definition
    - base_url => 네이버 웹툰 메인 페이지 
    - Subpage => 웹튼 A의 상세 페이지
    - Episode_page => 웹튼 A의 i번째 회차 페이지
'''


# Stage 0. Function Definition
def main_crawl():

    # Step 1. Data Collection for sub-page url
    base_url = 'https://comic.naver.com/webtoon/weekday.nhn'
    html = requests.get(base_url).text
    soup = BeautifulSoup(html, 'html.parser')

    item_tag = soup.select('a.title')

    link_list = [link.get('href') for link in item_tag]

    print('finish')

    return link_list


def subpage_crawl(link_list):
    episode_link = {}

    # Step 3. Collect Target Episode Page URL
    # 11개 [512:532]
    for link in tqdm(link_list[523:532]):
        '''
        
        (1) link => '/webtoon/list?titleId=570503&weekday=thu'
        (2) 따라서 Full Type 형태의 링크로 전처리 필요
        
        '''

        sub_url = f'https://comic.naver.com{link}'
        html = requests.get(sub_url).text
        soup = BeautifulSoup(html, 'html.parser')
        '''
        
        1) Return Type => Dictionary
        2) Return Example = {'webtoon A' : [(Episode 1, Upload time), (Episode 2], Upload time),
                             'webtoon b' : [(Episode 1, Upload time), (Episode 2, Upload time)]}
        
        '''
        item_id = soup.select_one('div.comicinfo > div.detail > h2 > span.title').get_text()
    # Step 4. Collect Pagination
        '''
    
        1) Pagination URL => sub_url + '&page=페이지번호'
        2) 1주일에 1개 연재 => 10 ~ 20개의 컨텐츠
        3) 1주일에 2개 연재 => 20 ~ 40개의 컨텐츠    
        => 페이지 당 10개의 컨텐츠 표기, 따라서 4페이지 정도면 충분할 것으로 예상
        
        '''
        episode_list, upload_list = [], []

        for i in range(1, 5):

            pagination_url = f'{sub_url}&page={i}'
            html = requests.get(pagination_url).text
            soup = BeautifulSoup(html, 'html.parser')

            episode_tag = soup.select('td.title > a')
            upload = soup.find_all('td', {'class' : 'num'})

            episode_list += [link.get('href') for link in episode_tag]
            upload_list += [datetime.strptime(date.text, '%Y.%m.%d') for date in upload]

        upload_list = [str(date).split(' ')[0] for date in upload_list if date >= datetime(2022, 8, 1)]
        episode_link[item_id] = list(set(zip(episode_list, upload_list)))

    return episode_link


def iter_comment(url, key):
    user_id, posted_time, webtoon_name, comments, i = [], [], [], [], 1
    driver = webdriver.Chrome('./chromedriver.exe') # Driver 공통 실행
    adult_list = ['796867', '773522', '719508', '670143', '791253', '789434',
                  '797410', '802293', '756139', '793350']
    condition_id = url.split('=')[1]
    condition_id = condition_id.split('&')[0]

    if condition_id in adult_list:
        print("성인 관람 가능 웹툰 입니다.")

    else:
        driver.implicitly_wait(1)
        driver.get(url)
        driver.find_element(By.XPATH, '//*[@id="cbox_module_wai_u_cbox_sort_option_tab2"]/span[2]').click()
        driver.implicitly_wait(3)

        while i:
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            user_tag = soup.select('span.u_cbox_nick')
            time_tag = soup.select('span.u_cbox_date')
            comment_tag = soup.select('span.u_cbox_contents')

            user_id += [user_id.text for user_id in user_tag]
            posted_time += [str(write_time.get('data-value').split('+')[0]) for write_time in time_tag]
            comments += [comment.text for comment in comment_tag]
            try:
                if i % 10 != 0:
                    # wait(driver, timeout)
                    wait(driver, 10).until(EC.element_to_be_clickable((By.LINK_TEXT, f'{i + 1}'))).click()
                else:
                    wait(driver, 10).until(EC.element_to_be_clickable((By.CLASS_NAME, 'u_cbox_next'))).click()

                i += 1

            except Exception as e:
                print(e)
                i = False

            time.sleep(1)

    driver.close()
    webtoon_name = [key] * len(posted_time)

    return user_id, posted_time, webtoon_name, comments


def episode_page_crawl(episode_link):
    for key, values in episode_link.items():
        interaction_df = pd.DataFrame(columns=['user_id', 'posted_time', 'item_id'])
        user_id, posted_time, item_id, text = [], [], [], []

        for value in values:
            value = value[0]
            temp = value.split('?')[1]
            url_title = temp.split('&')[0]
            url_episode = temp.split('&')[1]
            episode_url = f'https://comic.naver.com/comment/comment.nhn?{url_title}&{url_episode}'

            user, write_time, name, comment = iter_comment(episode_url, key)
            user_id += user
            posted_time += write_time
            item_id += name
            text += comment

        interaction_df['user_id'] = user_id
        interaction_df['posted_time'] = posted_time
        interaction_df['item_id'] = item_id
        interaction_df['text'] = text
        print(interaction_df)
        interaction_df.to_csv(f'[{key}]Interaction Data.csv',
                              index=False,
                              encoding='utf-8-sig')


# Stage 1. Main Loop
if __name__ == "__main__":
    base_info = main_crawl()
    link = subpage_crawl(base_info)
    episode_page_crawl(link)
    '''
    interaction_df.to_csv('[test]Interaction_DataFrame.csv',
                          index=False,
                          encoding='utf-8-sig')
    '''