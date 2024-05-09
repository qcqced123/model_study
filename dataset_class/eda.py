import warnings
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from wordcloud import WordCloud
from typing import List
warnings.filterwarnings('ignore')


def plot_length_stats_of_text(lengths: List) -> None:
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


def plot_token_length_box(lengths: List) -> None:
    """ Helper Function for plotting the token length box plot

    Args:
        lengths: List, the list of token length, index of this array should be the same as the index of the data instance
    """
    sns.set_style(style='dark')
    plt.figure(figsize=(15, 10))
    plt.boxplot(lengths, labels=['count'], showmeans=True)


def plot_log_scale_token_length(lengths: List) -> None:
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


def plot_word_cloud(df: pd.DataFrame, col_name: str) -> None:
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


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
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


def plot_distribution(df: pd.DataFrame, col_name: str, mode: str = 'kde') -> None:
    """ Helper function for plotting the distribution of specific feature (from argument)

    Args:
        df: pd.DataFrame, you must pass the df, which is only left with columns that you want to calculate the corr score
        col_name: default str, specific column name what you want to visualize
        mode: default 'kde'
    """
    plt.figure(figsize=(12,10))
    sns.displot(
        df[f"{col_name}"],
        kind=mode
    )
    plt.show()


def plot_categorical_count(df: pd.DataFrame, x_col: str, hue: str) -> None:
    """ Helper function for plotting the distribution of specific feature (from argument)

    Args:
        df: pd.DataFrame
        x_col: default str, col name which will be setting on x-axis on count plot graph, must be categorical
        hue: default str, col name which will be divided the class of x_col's data

    """
    sns.set_style(style='dark')
    plt.figure(figsize=(15,10))
    sns.countplot(
        data=df,
        x=df[f"{x_col}"],
        hue=hue
    )
    plt.show()


def plot_bar(df: pd.DataFrame, y_col: str, x_col: str) -> None:
    """ Helper function for plotting the bar plot

    Args:
        df: pd.DataFrame
        y_col: default str, col name which will be setting on y-axis on bar plot graph
        x_col: default str, col name which will be setting on x-axis on bar plot graph
    """
    sns.set_style(style='dark')
    plt.figure(figsize=(15, 10))
    sns.barplot(
        data=df,
        y=df[f"{y_col}"],
        x=df[f"{x_col}"],
    )
    plt.show()


def plot_target_ratio_by_cal_feature(df: pd.DataFrame, features: List[str], label: str, n_row: int, n_col: int) -> None:
    """ Helper Function for calculating each feature's explanatory power, by plotting
    target values ratio by categorical feature

    1) set matplotlib.pyplot.figure
    2) set GridSpec
    3) apply subplots adjust
    4) assign each of feature's visualization result into sub grid area


    Insight 1: statistical significance
        you can feature's statistical significance with each feature's stick length
        if stick is too long, it has no meaningful statistical significance

    Insight 2: empower of each feature about label(target value) distinguishing
        if graph has great gap or large variance, this feature have awsome empowered about label
    """
    sns.set_style(style='dark')
    plt.figure(figsize=(30, 30))
    grid = gridspec.GridSpec(n_row, n_col)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # set distance of each subplot

    for idx, feature in enumerate(features):
        ax = plt.subplot(grid[idx])
        sns.barplot(x=feature, y=label, data=df, palette='Set2', ax=ax)


def plot_target_ratio_by_cont_feature(df: pd.DataFrame, features: List[str], label: str, n_row: int, n_col: int) -> None:
    """ Helper Function for calculating each feature's explanatory power, by plotting
    target values ratio by continuous feature

    1) set matplotlib.pyplot.figure
    2) set GridSpec
    3) apply subplots adjust
    4) assign each of feature's visualization result into sub grid area

    you can get two insight here:

    Insight 1: statistical significance
        you can feature's statistical significance with each feature's stick length
        if stick is too long, it has no meaningful statistical significance

    Insight 2: empower of each feature about label(target value) distinguishing
        if graph has great gap or large variance, this feature have awsome empowered about label

    """
    copied_df = df.copy()
    sns.set_style(style='dark')
    plt.figure(figsize=(30, 30))
    grid = gridspec.GridSpec(n_row, n_col)
    plt.subplots_adjust(wspace=0.2, hspace=0.4)  # set distance of each subplot

    for idx, cont_feature in enumerate(features):
        copied_df[cont_feature] = pd.cut(df[cont_feature], 20)
        ax = plt.subplot(grid[idx])
        sns.barplot(x=cont_feature, y=label, data=copied_df, palette='Set2', ax=ax)
        ax.tick_params(axis='x', labelrotation=10)  # for avoiding x-labels over-lap
