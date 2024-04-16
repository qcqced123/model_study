import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from typing import List


def print_length_stats_of_text(lengths: List) -> None:
    """ Helper Function for checking the statistics of the text length

    Args:
        lengths: List, the list of token length, index of this array should be the same as the index of the data instance
    """
    print('------------- Length Statistic Info -------------')
    print('Max Length of Sentence : {}'.format(np.max(lengths)))
    print('Min Length of Sentence : {}'.format(np.min(lengths)))
    print('Mean Length of Sentence : {:.2f}'.format(np.mean(lengths)))
    print('Std Length of Sentence : {:.2f}'.format(np.std(lengths)))
    print('Median Length of Sentence : {}'.format(np.median(lengths)))
    print('Q1 Length of Sentence : {}'.format(np.percentile(lengths, 25)))
    print('Q3 Length of Sentence : {}'.format(np.percentile(lengths, 75)))
    return


def token_length_box_plot(lengths: List) -> None:
    """ Helper Function for plotting the token length box plot

    Args:
        lengths: List, the list of token length, index of this array should be the same as the index of the data instance
    """
    sns.set_style(style='dark')
    plt.figure(figsize=(15, 10))

    plt.boxplot(lengths, labels=['count'], showmeans=True)


def log_scale_token_length_plot(lengths: List) -> None:
    """ Helper Function for plotting the log scale token length plot

    Args:
        lengths: List, the list of token length, index of this array should be the same as the index of the data instance
    """
    sns.set_style(style='dark')

    plt.figure(figsize=(15, 10))
    plt.hist(lengths, bins=30, alpha=0.5, color='blue', label='tokens')

    plt.yscale('log')
    plt.title("Log-Histplot of Text length", fontsize=20)

    plt.xlabel("length of tokens", fontsize=16)
    plt.ylabel("number of texts", fontsize=16)
    return


def word_cloud_plot(df: pd.DataFrame, col_name: str) -> None:
    """ Helper Function for plotting the word cloud of the given column in the dataframe

    Args:
        df: pd.DataFrame, the dataframe you want to plot the word cloud
        col_name: str, the column name in the dataframe you want to plot the word cloud
    """
    cloud = WordCloud(width=800, height=600).generate(" ".join(df[f'{col_name}']))
    plt.figure(figsize=(15, 10))
    plt.imshow(cloud)
    plt.axis('off')
    return


def correlation_heatmap(df: pd.DataFrame) -> None:
    """ Helper Function for plotting the correlation heatmap of the given dataframe

    Args:
        df: pd.DataFrame, you must pass the df, which is only left with columns that you want to calculate the corr score
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        df.corr(),
        xticklabels=df.columns,
        yticklabels=df.columns,
        square=True,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.show()
    return

