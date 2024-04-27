import arxiv
import subprocess
import numpy as np
import pandas as pd
import os, re, sys, csv, json, requests, time, datetime, random, urllib

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
    client = arxiv.Client(page_size=1000, delay_seconds=2, num_retries=3)
    result = client.results(
        arxiv.Search(query=query, max_results=max_results, sort_by=sorting)
    )

    paper_list = []
    for paper in result:
        paper_list.append(paper)

    arxiv_df = pd.DataFrame([vars(paper) for paper in paper_list])
    return arxiv_df


if __name__ == '__main__':
    q, return_results, standard = sys.stdin.readline().split()
    values = set_sorting(sorting=standard)

    df = main_loop(
        query=q,
        max_results=return_results,
        sorting=values
    )
    df.to_csv(f'{q}_arxiv.csv', index=False)
