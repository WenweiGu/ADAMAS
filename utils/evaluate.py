# -*- coding:utf-8 -*-
import numpy as np


def best_f1_score_with_point_adjust(labels, anomaly_scores, delay=10):
    """
    Borrow from https://github.com/NetManAIOps/AutoKAD
    """
    assert len(labels) == len(anomaly_scores)

    def point_adjust(raw_labels):
        adjust_pos = []
        new_labels = raw_labels.copy()

        for j in range(len(raw_labels)):
            if (j == 0) and (raw_labels[j] == 1):
                adjust_pos.append(j)
            elif (j != 0) and (raw_labels[j] == 1) and (raw_labels[j - 1] == 0):
                adjust_pos.append(j)

        for position in adjust_pos:
            new_labels[position: position + delay + 1] = 1
        return new_labels

    # take the points in the interval under the requirement of delay
    def tagging_interval(position, if_used):
        true_p = 0

        j = 0
        while True:
            if j > delay:
                break
            next_pos = position + j
            if (next_pos < len(labels)) and (labels[next_pos] == 1) and (if_used[next_pos] == 0):
                if_used[next_pos] = 1
                true_p += 1
            else:
                break
            j += 1

        j = 1
        while True:
            if j > delay:
                break
            previous_pos = position - j
            if (previous_pos >= 0) and (labels[previous_pos] == 1) and (if_used[previous_pos] == 0):
                if_used[previous_pos] = 1
                true_p += 1
            else:
                break
            j += 1

        return true_p

    # take all the points in the anomaly interval
    def tagging_whole_interval(position, if_used):
        true_p = 0

        j = 0
        while True:
            next_pos = position + j
            if (next_pos < len(labels)) and (labels[next_pos] == 1):
                assert if_used[next_pos] != 1
                if_used[next_pos] = 1
                true_p += 1
            else:
                break
            j += 1

        j = 1
        while True:
            previous_pos = position - j
            if (previous_pos >= 0) and (labels[previous_pos] == 1):
                assert if_used[previous_pos] != 1
                if_used[previous_pos] = 1
                true_p += 1
            else:
                break
            j += 1

        return true_p

    labels = point_adjust(labels)
    used = np.zeros_like(anomaly_scores)
    pos = np.argsort(anomaly_scores)[::-1]
    threshold = -1

    if (labels == 0).all():
        return {'p': 1, 'r': 1, 'f': 1}
    else:
        tp, predicted = 0, 0
        best_f1 = best_p = best_r = 0
        all_true = sum(labels)

        for i in range(len(pos)):
            if used[pos[i]] == 1:
                continue

            if labels[pos[i]] == 1:
                new_predicted = tagging_whole_interval(pos[i], used)
                tp += new_predicted
                predicted += new_predicted
            else:
                predicted += 1

            if tp > 0:
                precision = tp / predicted
                recall = tp / all_true
                f1_score = 2 * precision * recall / (precision + recall)

                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_p = precision
                    best_r = recall
                    threshold = anomaly_scores[pos[i]]

        return {'p': best_p, 'r': best_r, 'f': best_f1, 'ths': threshold}


def best_f1_score_point(labels, anomaly_scores):
    assert len(labels) == len(anomaly_scores)
    pos = np.argsort(anomaly_scores)[::-1]
    threshold = -1

    if (labels == 0).all():
        return {'p': 1, 'r': 1, 'f': 1}
    else:
        tp = 0
        best_f1 = best_p = best_r = 0
        all_true = sum(labels)

        for i in range(len(pos)):
            if labels[pos[i]] == 1:
                tp += 1

            if tp > 0:
                precision = tp / (i + 1)
                recall = tp / all_true
                f1_score = 2 * precision * recall / (precision + recall)

                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_p = precision
                    best_r = recall
                    threshold = anomaly_scores[pos[i]]

        return {'p': best_p, 'r': best_r, 'f': best_f1, 'ths': threshold}
