#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import json
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


# Compute macro-averaged F1-scores, macro-averaged precision
def eval_measures(gt, pred):
    """Compute macro-averaged F1-scores, macro-averaged precision,
    macro-averaged recall, and micro-averaged accuracy according the ad hoc
    rules discussed at the top of this file.
    Parameters
    ----------
    gt : dict
        Ground truth, where keys indicate text file names
        (e.g. `unknown00002.txt`), and values represent
        author labels (e.g. `candidate00003`)
    pred : dict
        Predicted attribution, where keys indicate text file names
        (e.g. `unknown00002.txt`), and values represent
        author labels (e.g. `candidate00003`)
    Returns
    -------
    f1 : float
        Macro-averaged F1-score
    precision : float
        Macro-averaged precision
    recall : float
        Macro-averaged recall
    accuracy : float
        Micro-averaged F1-score
    """
    actual_authors = list(gt.values())
    encoder = LabelEncoder().fit(['<UNK>'] + actual_authors)

    text_ids, gold_authors, silver_authors = [], [], []
    for text_id in sorted(gt):
        text_ids.append(text_id)
        gold_authors.append(gt[text_id])
        try:
            silver_authors.append(pred[text_id])
        except KeyError:
            # missing attributions get <UNK>:
            silver_authors.append('<UNK>')

    assert len(text_ids) == len(gold_authors)
    assert len(text_ids) == len(silver_authors)

    # replace non-existent silver authors with '<UNK>':
    silver_authors = [a if a in encoder.classes_ else '<UNK>'
                      for a in silver_authors]

    gold_author_ints = encoder.transform(gold_authors)
    silver_author_ints = encoder.transform(silver_authors)

    # get F1 for individual classes (and suppress warnings):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f1 = f1_score(gold_author_ints,
                      silver_author_ints,
                      labels=list(set(gold_author_ints)),
                      average='macro')
        precision = precision_score(gold_author_ints,
                                    silver_author_ints,
                                    labels=list(set(gold_author_ints)),
                                    average='macro')
        recall = recall_score(gold_author_ints,
                              silver_author_ints,
                              labels=list(set(gold_author_ints)),
                              average='macro')
        accuracy = accuracy_score(gold_author_ints,
                                  silver_author_ints)

    return f1, precision, recall, accuracy
# end eval_measures


# Calculates evaluation measures for a single attribution problem
def evaluate(ground_truth_file, predictions_file):
    """
    Calculates evaluation measures for a single attribution problem
    :param ground_truth_file:
    :param predictions_file:
    :return:
    """
    gt = {}
    with open(ground_truth_file, 'r') as f:
        for attrib in json.load(f)['ground_truth']:
            gt[attrib['unknown-text']] = attrib['true-author']

    pred = {}
    with open(predictions_file, 'r') as f:
        for attrib in json.load(f):
            if attrib['unknown-text'] not in pred:
                pred[attrib['unknown-text']] = attrib['predicted-author']
    f1, precision, recall, accuracy = eval_measures(gt, pred)
    return f1, precision, recall, accuracy
# end evaluate


# Calculates evaluation measures for a PAN-18 collection of attribution problems
def evaluate_all(path_collection, path_answers, path_out):
    """
    Calculates evaluation measures for a PAN-18 collection of attribution problems
    :param path_collection:
    :param path_answers:
    :param path_out:
    :return:
    """
    infocollection = path_collection + os.sep + 'collection-info.json'
    problems = []
    data = []
    with open(infocollection, 'r') as f:
        for attrib in json.load(f):
            problems.append(attrib['problem-name'])
    scores = [];
    for problem in problems:
        f1, precision, recall, accuracy = evaluate(path_collection + os.sep + problem + os.sep + 'ground-truth.json',
                                                   path_answers + os.sep + 'answers-' + problem + '.json')
        scores.append(f1)
        data.append({'problem-name': problem, 'macro-f1': round(f1, 3), 'macro-precision': round(precision, 3),
                     'macro-recall': round(recall, 3), 'micro-accuracy': round(accuracy, 3)})
        print(str(problem), 'Macro-F1:', round(f1, 3))
    overall_score = sum(scores) / len(scores)
    # Saving data to output files (out.json and evaluation.prototext)
    with open(path_out + os.sep + 'out.json', 'w') as f:
        json.dump({'problems': data, 'overall_score': round(overall_score, 3)}, f, indent=4, sort_keys=True)
    print('Overall score:', round(overall_score, 3))
    prototext = 'measure {\n key: "mean macro-f1"\n value: "' + str(round(overall_score, 3)) + '"\n}\n'
    with open(path_out + os.sep + 'evaluation.prototext', 'w') as f:
        f.write(prototext)
# end evaluate_all

def main():
    parser = argparse.ArgumentParser(description='Evaluation script AA@PAN2018')
    parser.add_argument('-i', type=str,
                        help='Path to evaluation collection')
    parser.add_argument('-a', type=str,
                        help='Path to answers folder')
    parser.add_argument('-o', type=str,
                        help='Path to output files')
    args = parser.parse_args()
    if not args.i:
        print('ERROR: The collection path is required')
        parser.exit(1)
    if not args.a:
        print('ERROR: The answers folder is required')
        parser.exit(1)
    if not args.o:
        print('ERROR: The output path is required')
        parser.exit(1)

    evaluate_all(args.i, args.a, args.o)


if __name__ == '__main__':
    main()