import sys
import arxiv
import pandas as pd

from tqdm.auto import tqdm
from multiprocessing import Pool
from collections import Counter, defaultdict


def set_sorting(sorting: str = 'relevance') -> object:
    """
    Set the sorting criterion for the search results.

    if you pass argument sorting example below:
        relevance: arxiv.SortCriterion.Relevance
        latest_date: arxiv.SortCriterion.SubmittedDate

    Args:
        sorting: default str, sorting criterion for the search results,
                 Possible values are: 'relevance', 'lastUpdatedDate', 'submittedDate'

    Returns:

    """
    if sorting == 'relevance':
        return arxiv.SortCriterion.Relevance

    elif sorting == 'submittedDate':
        return arxiv.SortCriterion.SubmittedDate

    elif sorting == 'lastUpdatedDate':
        return arxiv.SortCriterion.LastUpdatedDate


def main_loop(query: str, max_results: int = 10, sorting=arxiv.SortCriterion.Relevance):
    """ main loop function for downloading query output from arxiv

    this function will download the paper named change 2110.03353v1 into it's title

    Usage:
        query: 'iclr2020', 'ELECTRA', 'NLP', 'Transformer' ...
        max_results: 10, 20, 30, 40, 50 ...
        sorting: 'relevance', 'submittedDate', 'lastUpdatedDate'

    NLP conference list:
        ACL, EMNLP, NAACL, COLING, EACL

    Args:
        query: str, query string for searching the arxiv database
        max_results: int, maximum number of results to return
        sorting: object, sorting criterion for the search results

    Returns:
        arxiv_df: pd.DataFrame, dataframe containing the search results
    """

    client = arxiv.Client(page_size=1000, delay_seconds=2, num_retries=3)
    result = client.results(
        arxiv.Search(query=query, max_results=max_results, sort_by=sorting)
    )

    paper_list = []
    for paper in tqdm(result):
        paper_list.append(paper)
        title = paper.title
        paper.download_pdf(
            dirpath='./download/',
            filename=f'{title}.pdf'
        )

    arxiv_df = pd.DataFrame([vars(paper) for paper in paper_list])
    return arxiv_df


if __name__ == '__main__':
    q = sys.stdin.readline().rstrip()
    return_results = int(sys.stdin.readline())
    standard = sys.stdin.readline().rstrip()
    values = set_sorting(sorting=standard)

    df = main_loop(
        query=q,
        max_results=return_results,
        sorting=values
    )
    df.to_csv(f'{q}_arxiv.csv', index=False)