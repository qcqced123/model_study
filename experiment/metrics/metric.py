import numpy as np


def accuracy(y_true: np.array, y_pred: np.array) -> float:
    """ accuracy metric function for Masked Langauge Model
    Args:
        y_true: ground truth, 1D Array for MLM Task (batch_size*sequence)
        y_pred: prediction, must be 2D Array for MLM Task (batch_size*sequence, vocab size)
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


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ recall = tp / (tp + fn) """
    tp = np.sum((y_true == 1) & (y_pred == 1))  # same as np.bitwise
    fn = np.sum((y_true == 1) & (y_pred == 0))
    score = tp / (tp + fn)
    return round(score.mean(), 4)


def precision(y_true, y_pred) -> float:
    """ precision = tp / (tp + fp) """
    tp = np.sum((y_true == 1) & (y_pred == 1))  # same as np.bitwise
    fp = np.sum((y_true == 0) & (y_pred == 1))
    score = tp / (tp + fp)
    return round(score.mean(), 4)


def f_beta(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 2) -> float:
    """
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
    tp = np.sum((y_true == 1) & (y_pred == 1))  # same as np.bitwise
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    f_precision = tp / (tp + fp)
    f_recall = tp / (tp + fn)
    score = (1 + beta ** 2) * f_precision * f_recall / (beta ** 2 * f_precision + f_recall)
    return round(score.mean(), 4)

