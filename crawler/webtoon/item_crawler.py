import pandas as pd
import os, sys, json, time, gc
import numpy as np
import requests
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm

'''

[Test Demo]
This is Source Code for Scrapping Contents Meta Data
1) Meta Data
    - WebToon_Name
    - Author
    - Genre
    - Rating
    - Thumbnail
    - Description
    
2) Collect Range => 22.08.01 ~ 22.11.20

3) Target Platform => Naver WebToon

4) Design Style => Bottom-Up Approach

'''


# Stage 0. Function Definition
def main_crawl():

    # Step 1. Data Collection for item_name, sub-page url
    base_url = 'https://comic.naver.com/webtoon/weekday.nhn'
    html = requests.get(base_url).text
    soup = BeautifulSoup(html, 'html.parser')

    item_tag = soup.select('a.title')
    img_tag = soup.select('div.thumb > a > img')  # <a> 하위에 정의된 <img> Tag 선택)

    name_list = [name.get('title') for name in item_tag]
    link_list = [link.get('href') for link in item_tag]
    img_list = [img.get('src') for img in img_tag]

    # Step 2. Make DataFrame & Collect Sub-Page Information
    item_df = pd.DataFrame(columns=['item_id', 'author', 'type', 'genre',
                                    'description', 'rating', 'thumbnail'], )
    item_df['item_id'] = name_list
    item_df['thumbnail'] = img_list

    print('finish')

    return name_list, link_list, img_list, item_df


def sub_crawl(link_list, df):

    author_list, type_list, genre_list, description_list = [], [], [], []
    for link in tqdm(link_list):
        # link => '/webtoon/list?titleId=570503&weekday=thu'
        # 따라서 Full Type 형태의 링크로 전처리 필요
        sub_url = f'https://comic.naver.com{link}'
        html = requests.get(sub_url).text
        soup = BeautifulSoup(html, 'html.parser')
        ''' 
        
        author_list => 작가가 두명 이상인 경우가 존재 => 맨 앞의 작가만 적재하거나 스토리 작가, 그림 작가 나눠서 적재하는 방식 고려
        (다만 누가 스토리 작가, 그림 작가인지 서술 규칙이 없는 것 같다. 그렇다고 죄다 찾아서 수기로 넣자니.. 그건 아닌거 같고)
        genre_list => ('형식', '장르') => 쉼표 기준 왼쪽은 Type, 오른쪽은 Genre
        
        '''
        description_list.append(soup.select_one('div.comicinfo > div.detail > p').get_text())
        author_list.append(soup.select_one('div.comicinfo > div.detail > h2 > span.wrt_nm').get_text())
        context_info = soup.select_one('div.comicinfo > div.detail > p.detail_info > span.genre').get_text()

        type_list.append(context_info.split(', ')[0])
        genre_list.append(context_info.split(', ')[1])

    df['author'] = author_list
    df['type'] = type_list
    df['genre'] = genre_list
    df['description'] = description_list
    df.to_excel('(test)WebToon_Meta Data.xlsx', index=False)

    return df


# Stage 1. Main Loop
if __name__ == "__main__":
    base_info = main_crawl()
    item_df = sub_crawl(base_info[1], base_info[3])
    print(item_df)

