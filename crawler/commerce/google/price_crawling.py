import requests
import pandas as pd

from bs4 import BeautifulSoup as bs
from typing import List, Dict, Union, Any, Callable
from api.metrics.metric import rouge_n, bleu, rouge_l
from api.db_utils.db_product_meta import db_get_product_name


def crawl_ingredients_info(headers: Dict[str, str], ingredient: str) -> List:
    """ crawl the product information from google shopping query result by BS4

    Args:
        headers: default setting for crawling google shopping query result
        ingredient: default str, any iterable object that contain product name, food name, ingredient name
    """
    product = []
    site = f"https://www.google.com/search?newwindow=1&sca_esv=590361239&hl=en&gl=us&tbm=shop&psb=1&q={ingredient}&tbs=mr:1,merchagg:g8299768%7Cg113872638%7Cg6296794%7Cg128518585%7Cg784994%7Cm140415710%7Cm8175035%7Cm111838607%7Cm117437161%7Cm126920480%7Cm5066339911%7Cm114193152%7Cm7388148%7Cm6296724%7Cm260799036%7Cm6570704%7Cm253292496%7Cm131615169%7Cm178357103%7Cm436356195%7Cm280215458%7Cm10046%7Cm10473868%7Cm139575325%7Cm1247713&sa=X&ved=0ahUKEwjD6sH76MCDAxVlrlYBHYraBjgQsysIvA4oLw&biw=1247&bih=1220&dpr=1"
    try:
        response = requests.get(site, headers=headers)
        soup = bs(response.text, 'html.parser')
        list = soup.find("div", {"class": "sh-pr__product-results-grid sh-pr__product-results"})
        product = list.findAll("div", {"class": "sh-dgr__content"})

    except Exception as e:
        print(e)

    return product


def build_product_info_dict(ingredient: str, product: str) -> Dict:
    """ build product information dictionary by using crawled data from google shopping query result

    Args:
        ingredient: default str
        product: default str, product information from google shopping query result
    """
    data = {}
    try:
        product_name = product.find("h3", {"class": "tAxDx"}).get_text()  # product name
        price = product.find("span", {"class": "a8Pemb OFFNJ"}).get_text()  # displayed price of product
        site_name = product.find("div", {"class": "aULzUe IuHnof"}).get_text()  # site name
        site1 = product.find("div", {"class": "mnIHsc"})
        site2 = site1.find("a", class_='shntl')
        site3 = site2.get("href")
        site4 = "https://google.com" + site3
        url = site4.replace("amp;", "")

        data = {
            'ingredient': ingredient,
            'product': product_name,
            'price': price,
            'site': site_name,
            'url': url
        }

    except Exception as e:
        print(e)

    return data


def sorting_func(product_list: List[Dict], sort_func: Callable) -> List:
    """ sorting the product list algorithm for google shopping query result

    Args:
        product_list: default List[Dict], product information list from google shopping query result
        sort_func: default setting ROUGE-N score, but there are other options
            1) BLEU
            2) ROUGE-N
            3) ROUGE-L
    """
    lowest_price = sorted(product_list, key=lambda x: (-sort_func(x['ingredient'], x['product']), float(x['price'].replace('$', ''))))
    return lowest_price


def google_shopping(ingredients: List[str], sort_func: Callable = rouge_n) -> List[Dict[str, Union[str, Any]]]:
    """ loop function for crawling google shopping query result by BS4

    Args:
        ingredients: default List[str1, str2, ... strN],
                     any iterable object that contain product name, food name, ingredient name

        sort_func: default setting ROUGE-N score, but there are other options
            1) bleu
            2) rouge_n
            3) rouge_l

    Implementation:
        1) crawl_ingredients_info: crawl the product information from google shopping query result by BS4
        2) build_product_info_dict: build product information dictionary by using crawled data from google shopping query result
        3) sorting_func: sorting the product list algorithm for google shopping query result
        4) remove duplicate site: get only one product from each unique site


    Notes:
        original setting for soring func is Only use 'price' ascending, but we add the new options
        because there is a issue for product name mis-matching between DB and query output

        So, we add the N-Gram base metrics for solving mis-match problem
        => lowest_price = sorted(product_list, key=lambda x: float(x['price'].replace('$', '')))
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US, en;q=0.5'
    }

    result = []
    products_data = []
    for ingredient in ingredients:
        products = crawl_ingredients_info(headers, ingredient)

        if not products:
            continue

        count = 0
        product_data = []
        for product in products:
            data = build_product_info_dict(ingredient, product)
            if data: product_data.append(data)
            count += 1

        products_data.append(product_data)
        for product_list in products_data:
            try:
                unique_site = set()
                lowest_price = sorting_func(product_list, sort_func)

                # get only one product from each unique site
                for product_dict in lowest_price:
                    site = product_dict.get('site')
                    if 'Walmart' in site:
                        site = 'Walmart'
                    if 'eBay' in site:
                        site = 'eBay'
                    if site and site not in unique_site:
                        result.append(product_dict)
                        unique_site.add(site)

            except Exception as e:
                print(e)

    return result


if __name__ == "__main__":
    query = ["\"Tunisian-Spiced Meatballs with Apricot Glaze, Roasted Carrots & Scallion Couscous\""]
    results = google_shopping(query)
    df = pd.DataFrame.from_dict(results)
    print(df)
    df.to_csv('test_df.csv', index=False, encoding='utf-8-sig')
