import numpy as np
import torch.nn as nn
import configuration as configuration

from torch import Tensor
from collections import Counter
from typing import Tuple, List, Dict


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


def precision(y_true: np.ndarray, y_pred: np.ndarray, cfg: configuration.CFG, threshold: float = 0.5) -> float:
    """ calculate recall metrics for binary classification

    Args:
        y_true: ground truth, 1D Array for MLM Task (batch_size*sequence)
        y_pred: prediction, must be 2D Array for MLM Task (batch_size*sequence, vocab size)
        cfg: configuration file for the experiment, for setting the mode of calculating precision
        threshold: boundary value for separating negative and positive classes in binary classification situations

    Math:
        precision = tp / (tp + fp)
    """
    # for binary classification
    if cfg.num_labels == 2:
        y_pred = y_pred[..., 1]
        y_pred = (y_pred >= threshold).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))  # true positive
        fp = np.sum((y_true == 0) & (y_pred == 1))  # false positive
        if tp + fp == 0:
            score = 0
        else:
            score = tp / (tp + fp)

    # for multi-class classification
    else:
        y_pred = np.argmax(y_pred, axis=-1)
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


def recall(y_true: np.ndarray, y_pred: np.ndarray, cfg: configuration.CFG, threshold: float = 0.5) -> float:
    """ calculate recall metrics for binary classification, multi-class classification

    Args:
        y_true: ground truth, 1D Array for MLM Task (batch_size*sequence)
        y_pred: prediction, must be 2D Array for MLM Task (batch_size*sequence, vocab size)
        cfg: configuration file for the experiment, for setting the mode of calculating recall
        threshold: boundary value for separating negative and positive classes in binary classification situations

    Math:
        recall = tp / (tp + fn)
    """
    # for binary classification
    if cfg.num_labels == 2:
        y_pred = y_pred[..., 1]
        y_pred = (y_pred >= threshold).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))  # true positive
        fn = np.sum((y_true == 1) & (y_pred == 0))  # false negative
        if tp + fn == 0:
            score = 0
        else:
            score = tp / (tp + fn)

    # for multi-class classification
    else:
        y_pred = np.argmax(y_pred, axis=-1)
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

def specificity(y_true: np.ndarray, y_pred: np.ndarray, cfg: configuration.CFG, threshold: float = 0.5) -> float:
    """ calculate the specificity metrics for binary classification, multi-class classification

    Args:
        y_true: ground truth, 1D Array for MLM Task (batch_size*sequence)
        y_pred: prediction, must be 2D Array for MLM Task (batch_size*sequence, vocab size)
        cfg: configuration file for the experiment, for setting the mode of calculating recall
        threshold: boundary value for separating negative and positive classes in binary classification situations

    Math:
        specificity = tn / (tn + fp)
    """
    # for binary classification
    if cfg.num_labels == 2:
        y_pred = y_pred[..., 1]
        y_pred = (y_pred >= threshold).astype(int)

        tn = np.sum((y_true == 0) & (y_pred == 0))  # true positive
        fp = np.sum((y_true == 0) & (y_pred == 1))  # false negative

        # for exception handling division by zero
        if tn + fp == 0:
            score = 0
        else:
            score = tn / (tn + fp)

    # for multi-class classification
    else:
        y_pred = np.argmax(y_pred, axis=-1)
        score, unique_classes = [], np.unique(y_true)
        for class_ in unique_classes:
            tn = np.sum((y_true != class_) & (y_pred != class_))
            fp = np.sum((y_true != class_) & (y_pred == class_))

            # for exception handling division by zero
            if tn + fp == 0:
                score.append(0.0)
            else:
                score.append(tn / (tn + fp))

    return round(np.mean(score), 4)

def f_beta(y_true: np.ndarray, y_pred: np.ndarray, cfg: configuration.CFG, beta: float = 1, threshold: float = 0.5) -> float:
    """ calculate function for F_beta score in binary classification, multi-class classification

    Args:
        y_true: ground truth, 1D Array for MLM Task (batch_size*sequence)
        y_pred: prediction, must be 2D Array for MLM Task (batch_size*sequence, vocab size)
        cfg: configuration file for the experiment, for setting the mode of calculating F_beta
        beta: float, default is 1
        threshold: boundary value for separating negative and positive classes in binary classification situations

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
    f_precision, f_recall = precision(y_true, y_pred, cfg, threshold), recall(y_true, y_pred, cfg, threshold)
    numerator, denominator = (1 + beta**2) * f_precision * f_recall, (beta**2 * f_precision + f_recall)

    if denominator == 0:
        score = 0
    else:
        score = numerator / denominator

    return round(np.mean(score), 4)


def roc_auc(y_true: np.ndarray, y_logit: np.ndarray, cfg: configuration.CFG) -> float:
    """ calculate function for roc auc score in binary classification.

    roc auc is the alias of "receiver operating characteristic".
    x-axis of roc curve is meaning of "false negative rate", "1-specificity".
    y-axis of roc curve is meaning of "true positive rate", "recall", "sensitive".

    Args:
        y_true: ground truth, 1D Array for MLM Task (batch_size*sequence)
        y_logit: must be logit prediction, 2D Array for MLM Task (batch_size*sequence, vocab size)
                (please do not pass the class-like prediction array for this metric calculation)
        cfg: configuration file for the experiment, for setting the mode of calculating F_beta
    """
    # logic for binary classification
    # get threshold array from model predictions
    tpr_list, fpr_list = [], []
    y_pred = y_logit[..., 1]  # result's shape must be: [0.6, 0.7, 0.7 ...]
    thresholds = np.sort(y_pred)[::-1]

    # calculate the tpr, fpr from each threshold for making the roc curve
    for threshold in thresholds:
        # get class-like prediction by using current threshold
        tpr = recall(y_true=y_true, y_pred=y_logit, cfg=cfg, threshold=threshold)  # recall == sensitive == tpr
        fpr = 1 - specificity(y_true=y_true, y_pred=y_logit, cfg=cfg, threshold=threshold)  # 1-specificity == fpr
        tpr_list.append(tpr), fpr_list.append(fpr)

    tpr_list = [0] + tpr_list + [1]
    fpr_list = [0] + fpr_list + [1]

    # calculate the auc (area of under the curve) of roc curve
    auc = 0.0  # for avoiding the type cast
    tpr = np.array(tpr_list)
    fpr = np.array(fpr_list)
    for i in range(len(tpr)-1):
        width = fpr[i+1] - fpr[i]
        height = (tpr[i+1] + tpr[i]) / 2
        auc += (width * height)
    return auc


def pr_auc(y_true: np.ndarray, y_logit: np.ndarray, cfg: configuration.CFG) -> float:
    """ calculate function for pr auc score in binary classification.

    pr is the alias of "precision-recall".
    x-axis of pr curve is meaning of "recall".
    y-axis of pr curve is meaning of "precision".

    Args:
        y_true: ground truth, 1D Array for MLM Task (batch_size*sequence)
        y_logit: must be logit prediction, 2D Array for MLM Task (batch_size*sequence, vocab size)
                 (please do not pass the class-like prediction array for this metric calculation)
        cfg: configuration file for the experiment, for setting the mode of calculating F_beta
    """
    # logic for binary classification
    # get threshold array from model predictions
    recall_list, precision_list = [], []
    y_pred = y_logit[..., 1]  # result's shape must be: [0.6, 0.7, 0.7 ...]
    thresholds = np.sort(y_pred)[::-1]

    # calculate the tpr, fpr from each threshold for making the roc curve
    for threshold in thresholds:
        # get class-like prediction by using current threshold
        recall_list.append(recall(y_true=y_true, y_pred=y_logit, cfg=cfg, threshold=threshold))  # recall == sensitive == tpr
        precision_list.append(precision(y_true=y_true, y_pred=y_logit, cfg=cfg, threshold=threshold))  # 1-specificity == fpr

    recall_list = [0] + recall_list + [1]
    precision_list = [0] + precision_list + [1]

    # calculate the auc (area of under the curve) of roc curve
    auc = 0.0  # for avoiding the type cast
    recalls = np.array(recall_list)
    precisions = np.array(precision_list)
    for i in range(len(precision_list)-1):
        width = recalls[i+1] - recalls[i]
        height = (precisions[i+1] + precisions[i]) / 2
        auc += (width * height)

    return auc


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


def bleu(y_true: List[str], y_pred: List[str], n_size: int = 4, cfg: configuration.CFG = None) -> float:
    """ calculate BLEU score for machine translation, text generation task

    you must pass list of string for ground truth and prediction
    string must be tokenized by tokenizer such as 'mecab', 'sentencepiece', 'wordpiece' and so on,
    also they must be decoded by tokenizer to string, not tensor

    Input example:
        prediction= "the the the the the the"
        reference= "the cat is on the mat"

        prediction, reference = prediction.split(' '), reference.split(' ')
        bleu(prediction, reference, 1)
        => 0.3333

    Args:
        y_true: ground truth, 1D Array for MLM Task (batch_size*sequence)
        y_pred: prediction, must be 2D Array for MLM Task (batch_size*sequence, vocab size)
        n_size: int, default is 4, which is meaning of the maximum n-gram size
        cfg: configuration file for the experiment, for setting BLEU-N, tokenizer

    Math:
        BLEU-N = BR*(p1*p2*p3*...pn)^(1/n)
        BR = min(1, exp(1 - ref_len/gen_len))
        => exponential penalty, if gen_Len is shorter than ref_len, than penalty will be much big

    Reference:
        https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py
        https://github.com/huggingface/datasets/blob/main/metrics/bleu/bleu.py

    """

    def calculate_ngram() -> float:
        """ calculate n-gram score for BLEU-N

        mathematical expression of this function is:
             p1*p2*p3*...pn1

        Implementations:
            1) count the number of each n-gram in y_true and y_pred
            2) calculate the number of n-gram overlap between y_true and y_pred
            3) calculate the precision of n-gram
              - apply the smoothing method for avoiding return zero value to bleu metric (x)

        """
        score = 1
        for n in range(1, n_size+1):
            gen_ngram = [tuple(y_pred[i:i+n]) for i in range(len(y_pred) - n + 1)]
            ref_ngram = [tuple(y_true[j:j+n]) for j in range(len(y_true) - n + 1)]

            ref_counter = Counter(ref_ngram)
            gen_count, ref_count = len(gen_ngram), 0
            for gram in gen_ngram:
                if ref_counter[gram] and ref_counter[gram] > 0:
                    ref_counter[gram] -= 1
                    ref_count += 1

            score *= ref_count / gen_count
        return score

    def brevity_penalty() -> float:
        """ calculate brevity penalty for BLEU-N

        mathematical expression of this function is:
            min(1, exp(1 - ref_len/gen_len))
        """
        return min(1, np.exp(1 - len(y_true) / len(y_pred)))

    bleu_score = brevity_penalty() * calculate_ngram()**(1/n_size)
    return round(bleu_score, 4)


def rouge_n(y_true: List[str], y_pred: List[str], n_size: int = 4, beta: float = 1, cfg: configuration.CFG = None) -> float:
    """ calculate ROUGE-N score for text summarization task (ROUGE N-Gram)

    recall is very im portant in text summarization task, because we need to generate important sentences in ref
    so, this metric is very useful for evaluating text summarization model

    you must pass list of string for ground truth and prediction
    string must be tokenized by tokenizer such as 'mecab', 'sentencepiece', 'wordpiece' and so on,
    also they must be decoded by tokenizer to string, not tensor

    Args:
        y_true: ground truth, 1D Array for MLM Task (batch_size*sequence)
        y_pred: prediction, must be 2D Array for MLM Task (batch_size*sequence, vocab size)
        n_size: int, default is 4, which is meaning of the maximum n-gram size
        beta: float, default is 1, which is meaning of the weight of precision and recall
        cfg: configuration file for the experiment, for setting BLEU-N, tokenizer

    Math:
        (original) ROUGE-N = {Common N-Gram in Gen & Ref} / {N-Gram in Ref}
        (modified) ROUGE-N = F1-Score of (precision: no clipping precision of BLEU, recall: original ROUGE)
        => we choose modified version of ROUGE-N, because it is more useful for text summarization task

    Reference:
        https://aclanthology.org/W04-1013.pdf

    """

    gen_ngram = [tuple(y_pred[i:i+n_size]) for i in range(len(y_pred) - n_size + 1)]
    ref_ngram = [tuple(y_true[j:j+n_size]) for j in range(len(y_true) - n_size + 1)]

    common_count = 0
    ref_counter = Counter(ref_ngram)
    for gram in gen_ngram:
        if ref_counter[gram]:
            common_count += 1

    rouge_precision, rouge_recall = common_count / len(gen_ngram), common_count / len(ref_ngram)

    numerator = (1 + beta ** 2) * rouge_precision * rouge_recall
    denominator = (beta ** 2 * rouge_precision + rouge_recall)

    return np.round(numerator / denominator, 4) if denominator else 0


def rouge_l(y_true: str, y_pred: str, cfg: configuration.CFG = None) -> float:
    """ calculate ROUGE-L score for text summarization task (ROUGE Longest Common Sequence)

    recall is very im portant in text summarization task, because we need to generate important sentences in ref
    so, this metric is very useful for evaluating text summarization model

    you must pass list of string for ground truth and prediction
    string must be tokenized by tokenizer such as 'mecab', 'sentencepiece', 'wordpiece' and so on,
    also they must be decoded by tokenizer to string, not tensor

    Args:
        y_true: ground truth, 1D Array for MLM Task (batch_size*sequence)
        y_pred: prediction, must be 2D Array for MLM Task (batch_size*sequence, vocab size)
        cfg: configuration file for the experiment, for setting BLEU-N, tokenizer

    Math:
        (original) ROUGE-N = {Common N-Gram in Gen & Ref} / {N-Gram in Ref}
        (modified) ROUGE-N = F1-Score of (precision: no clipping precision of BLEU, recall: original ROUGE)
        => we choose modified version of ROUGE-N, because it is more useful for text summarization task

    Reference:
        https://aclanthology.org/W04-1013.pdf
    """
    gen_ngram = y_pred.split(' ')
    ref_ngram = y_true.split(' ')

    def cal_longest_common_sequence() -> int:
        """ calculating length of longest common sequence between generated text and reference text """
        result = 0
        rows, cols = len(gen_ngram) + 1, len(ref_ngram)+1

        dp = [[0]*cols for _ in range(rows)]
        for y in range(1, rows):
            for x in range(1, cols):
                if gen_ngram[y-1] == ref_ngram[x-1]:
                    dp[y][x] = dp[y-1][x-1] + 1
                    result = max(result, dp[y][x])
                    continue

                dp[y][x] = max(dp[y-1][x], dp[y][x-1])

        return result

    lcs = cal_longest_common_sequence()
    rouge_precision, rouge_recall = lcs / len(gen_ngram), lcs / len(ref_ngram)

    beta = rouge_precision/rouge_recall
    numerator = (1 + beta ** 2) * rouge_precision * rouge_recall
    denominator = (beta ** 2 * rouge_precision + rouge_recall)

    return np.round(numerator / denominator, 4) if denominator else 0
