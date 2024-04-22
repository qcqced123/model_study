import sys
import bottlenose
from bs4 import BeautifulSoup


def get_product_name(asin: str) -> None:
    """ Using Amazon Open API named bottlenose,"""
    try:
        response = amazon.ItemLookup(ItemId=asin, ResponseGroup='ItemAttributes', SearchIndex='All')
        soup = BeautifulSoup(response, 'xml')
        return soup

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == '__main__':
    AMAZON_ACCESS_KEY_ID = sys.stdin.readline().rstrip()
    AMAZON_SECRET_KEY = sys.stdin.readline().rstrip()
    AMAZON_ASSOC_TAG = sys.stdin.readline().rstrip()

    amazon = bottlenose.api.Amazon(AMAZON_ACCESS_KEY_ID, AMAZON_SECRET_KEY, AMAZON_ASSOC_TAG)
    example_asin = 'B07K3D3RC3'

    product_name = get_product_name(example_asin)
    if product_name: print(f"The product name for ASIN {example_asin} is: {product_name}")
    else: print("Product not found.")
