import streamlit
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


""" Connect to Elastic Search Server """


class MacConfig:
    password = 'IJ_+q4tOV1vCl-Yt51U1'
    ca_certs = '/Users/qcqced/Desktop/ElasticSearch/elasticsearch-8.12.2/config/certs/http_ca.crt'


class LinuxConfig:
    password = 'u5oNftFCVGzDRGYFUXTy'
    ca_certs = '/home/qcqced/바탕화면/QA_System/elasticsearch-8.13.1/config/certs/http_ca.crt'


mac_cfg, linux_cfg = MacConfig(), LinuxConfig()

try:
    es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", linux_cfg.password),
    ca_certs=linux_cfg.ca_certs
    )
    print(es.ping())

except ConnectionError as e:
    print("Connection Error:", e)


def search(input_query: str):

    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    h = model.encode(input_query)
    query = {
        "field": "DescriptionVector",
        "query_vector": h,
        "k": 30,
        "num_candidates": 500
    }

    candidates = es.knn_search(
        index="all_products",
        knn=query,
        source=['"ProductName', 'Description']
    )
    results = candidates['hits']['hits']
    return results


def main():
    streamlit.title("Search Fashion Products")
    query = streamlit.text_input("Enter your query here")
    if streamlit.button("Search"):
        if query:
            results = search(query)
            streamlit.subheader("Search Results")
            for result in results:
                with streamlit.container():
                    if '_source' in result:
                        try:
                            streamlit.header(f"{result['_source']['ProductName']}")
                        except Exception as e:
                            print(e)
                        try:
                            streamlit.write(result['_source']['Description'])
                        except Exception as e:
                            print(e)

                    streamlit.divider()


if __name__ == '__main__':
    main()