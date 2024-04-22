import numpy as np
import torch.nn as nn
import configuration as configuration
from torch import Tensor


def accuracy(y_true: np.array, y_pred: np.array, cfg: configuration.CFG = None) -> float:
    """ accuracy metric function for classification task such as MLM, SentimentAnalysis ... and so on

    Args:
        y_true: ground truth, 1D Array for MLM Task (batch_size*sequence)
        y_pred: prediction, must be 2D Array for MLM Task (batch_size*sequence, vocab size)
        cfg: configuration file for the experiment, for setting the mode of calculating accuracy
    """
    correct, len_label = 0, len(y_true[y_true != -100])
    pred = np.argmax(y_pred, axis=-1)  # return index of max value
    correct += np.sum(pred == y_true).item()
    return round(correct / len_label, 4)


def top_k_acc(y_true: np.array, y_pred: np.array, k: int = 3) -> float:
    """ top k accuracy """
    correct = 0
    pred = np.topk(y_pred, k, dim=1)[1]
    assert pred.shape[0] == len(y_true)
    for i in range(k):
        correct += np.sum(pred[pred[:, i] == y_true]).item()
    return round(correct / len(y_true), 4)


def pearson_score(y_true: np.array, y_pred: np.array) -> float:
    x, y = y_pred, y_true
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    cov = np.sum(vx * vy)
    corr = cov / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)) + 1e-12)
    return corr


def precision(y_true: np.ndarray, y_pred: np.ndarray, cfg: configuration.CFG) -> float:
    """ calculate recall metrics for binary classification

    Args:
        y_true: ground truth, 1D Array for MLM Task (batch_size*sequence)
        y_pred: prediction, must be 2D Array for MLM Task (batch_size*sequence, vocab size)
        cfg: configuration file for the experiment, for setting the mode of calculating precision

    Math:
        precision = tp / (tp + fp)
    """
    y_pred = np.argmax(y_pred, axis=-1)

    # for binary classification
    if cfg.num_labels == 2:
        tp = np.sum((y_true == 1) & (y_pred == 1))  # same as np.bitwise
        fp = np.sum((y_true == 0) & (y_pred == 1))
        if tp + fp == 0:
            score = 0
        else:
            score = tp / (tp + fp)

    # for multi-class classification
    else:
        score, unique_classes = [], np.unique(y_true)
        for class_ in unique_classes:
            tp = np.sum((y_true == class_) & (y_pred == class_))
            fp = np.sum((y_true != class_) & (y_pred == class_))

            # Handling division by zero
            if tp + fp == 0:
                score.append(0.0)
            else:
                score.append(tp / (tp + fp))

    return round(np.mean(score), 4)


def recall(y_true: np.ndarray, y_pred: np.ndarray, cfg: configuration.CFG) -> float:
    """ calculate recall metrics for binary classification, multi-class classification

    Args:
        y_true: ground truth, 1D Array for MLM Task (batch_size*sequence)
        y_pred: prediction, must be 2D Array for MLM Task (batch_size*sequence, vocab size)
        cfg: configuration file for the experiment, for setting the mode of calculating recall

    Math:
        recall = tp / (tp + fn)
    """
    y_pred = np.argmax(y_pred, axis=-1)

    # for binary classification
    if cfg.num_labels == 2:
        tp = np.sum((y_true == y_pred) & (y_true != 1))  # same as np.bitwise
        fn = np.sum((y_true == y_pred) & (y_true == 0))
        if tp + fn == 0:
            score = 0
        else:
            score = tp / (tp + fn)

    # for multi-class classification
    else:
        score, unique_classes = [], np.unique(y_true)
        for class_ in unique_classes:
            tp = np.sum((y_true == class_) & (y_pred == class_))
            fn = np.sum((y_true == class_) & (y_pred != class_))

            # Handling division by zero
            if tp + fn == 0:
                score.append(0.0)
            else:
                score.append(tp / (tp + fn))

    return round(np.mean(score), 4)


def f_beta(y_true: np.ndarray, y_pred: np.ndarray, cfg: configuration.CFG, beta: float = 1) -> float:
    """ calculate function for F_beta score in binary classification, multi-class classification

    Args:
        y_true: ground truth, 1D Array for MLM Task (batch_size*sequence)
        y_pred: prediction, must be 2D Array for MLM Task (batch_size*sequence, vocab size)
        cfg: configuration file for the experiment, for setting the mode of calculating F_beta
        beta: float, default is 2

    Math:
        TP (true positive): pred == 1 && true == 1
        FP (false positive): pred == 1 && true == 0
        FN (false negative): pred == 0 && true == 1
        f_beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
        if you want to emphasize precision, set beta < 1, options: 0.3, 0.6
        if you want to emphasize recall, set beta > 1, options: 1.5, 2

    Reference:
        https://blog.naver.com/PostView.naver?blogId=wideeyed&logNo=221531998840&parentCategoryNo=&categoryNo=2&
    """
    f_precision, f_recall = precision(y_true, y_pred, cfg), recall(y_true, y_pred, cfg)
    numerator, denominator = (1 + beta**2) * f_precision * f_recall, (beta**2 * f_precision + f_recall)

    if denominator == 0:
        score = 0
    else:
        score = numerator / denominator

    return round(np.mean(score), 4)


def cosine_similarity(a: Tensor, b: Tensor, eps=1e-8) -> np.ndarray:
    """ calculate cosine similarity for two tensors of hidden states
    you must pass detached tensor which is already on CPU not GPU
    Args:
        a: Tensor, shape of [batch*seq, dim]
        b: Tensor, shape of [batch*seq, dim]
        eps: for numerical stability
    """
    metric = nn.CosineSimilarity(dim=-1, eps=eps)
    output = metric(a, b).mean().numpy()
    return output


def ppl(loss: np.ndarray) -> float:
    """ for calculating metric named 'Perplexity',
    which is used by validating language modeling task such as rnn, gpt

    Args:
        loss: mean scalar of batch's cross entropy

    Maths:
        PPL(x) = exp(CE)

    """
    return np.exp(loss)
