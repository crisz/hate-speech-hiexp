"""
Generate tabular layouts by taking the outputs of hierarchical explanations
as inputs. Output sample:
https://openreview.net/pdf?id=BkxRRkSKwr, Appendix C
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path


def plot_score_array(layers, score_array, sent_words, model_pred=None):
    max_abs = abs(score_array).max()
    width = max(10, score_array.shape[1])
    height = max(5, score_array.shape[0])
    fig, ax = plt.subplots(figsize=(width, height))

    vmin, vmax = -5.0, 5.0

    # fix CLS and SEP
    for xi in range(1, score_array.shape[0]):
        xj = 0
        while xj < score_array.shape[1] and score_array[xi,xj] != 0:
            score_array[xi, xj] = 0
            xj += 1
        xj = score_array.shape[1] - 1
        while xj >= 0 and score_array[xi, xj] != 0:
            score_array[xi, xj] = 0
            xj -= 1
    score_array = score_array[:,1:-1]
    sent_words = sent_words[1:-1]

    # add a score array showing model prediction
    if model_pred is not None:
        arr = np.array([model_pred] * score_array.shape[1]).reshape(1,-1)
        score_array = np.concatenate([score_array, arr], 0)

    im = ax.imshow(score_array, cmap='coolwarm', aspect=0.5, vmin=vmin, vmax=vmax)
    #fig.colorbar(im, orientation='horizontal', fraction=0.05, extend='both')
    ax.set_yticks([])
    ax.set_xticks([])
    #ax.set_yticks(np.arange(len(y_ticks)))
    #ax.set_yticklabels(y_ticks)
    print(layers)
    cnt = 0
    if layers is not None:
        for idx, i in enumerate(sorted(layers.keys())):
            for entry in layers[i]:
                print(">> entry: ", entry)
                start, stop = entry[2] - len(entry[1]) + 1, entry[2]
                for j in range(start, stop + 1):
                    color = (0.0, 0.0, 0.0)
                    ax.text(j, cnt, sent_words[j], ha='center', va='center', fontsize=11 if len(sent_words[j]) < 10 else 8,
                            color=color)
            cnt += 1
    else:
        print(">> score_array.shape: ", score_array.shape)
        for i in range(score_array.shape[0]):
            for j in range(score_array.shape[1]):
                print(">> score_array({}, {}): {}".format(i, j, score_array[i, j]))
                if score_array[i,j] != 0:
                    fontsize = 12
                    if len(sent_words[j]) >= 8:
                        fontsize = 8
                    if len(sent_words[j]) >= 12:
                        fontsize = 6
                    ax.text(j, i, sent_words[j], ha='center', va='center',
                           fontsize=fontsize)
    return im

def visualize_tabs(tab_file, model_name, method_name):
    """
    visualizing hierarchical explanations, take as input the output pkl of hierarchical explanation algorithms
    :param tab_file:
    :param model_name:
    :param method_name:
    :return:
    """
    f = open(tab_file, 'rb')
    data = pickle.load(f)
    for i,entry in enumerate(data):
        sent_words = entry['text'].split()
        score_array = entry['tab']
        label_name = entry['label']
        model_pred = entry.get('pred', None)
        if score_array.ndim == 1:
            score_array = score_array.reshape(1,-1)

        new_score_array = []
        new_sent_words = []
        prev_word = ''
        for xj in range(score_array.shape[1]):
            word = sent_words[xj]
            if word != prev_word:
                prev_word = word
                new_sent_words.append(word)
                new_score_array.append(score_array[:,xj])
        new_score_array = np.stack(new_score_array,0).transpose((1,0))

        score_array, sent_words = new_score_array, new_sent_words

        if score_array.shape[1] <= 400:
            im = plot_score_array(None, score_array, sent_words, model_pred)
            plt.title(label_name, fontsize=14)
            dir = 'figs/{}_{}'.format(model_name, method_name)
            if not os.path.isdir(dir): os.mkdir(dir)
            plt.savefig('figs/{}_{}/fig_{}.png'.format(model_name, method_name, i), bbox_inches='tight')
            plt.close()

def visualize_sequences(txt_file, model_name, method_name):
    f = open(txt_file)
    lines = f.readlines()
    print("Number of lines: ", len(lines))
    for i, line in enumerate(lines):
        print("Curernt line: ", line)

        score_array, sent_words = [], []
        entries = line.strip().split('\t')
        print("entries: ", entries)
        for entry in entries:
            items = entry.split()
            word, score = ' '.join(items[:-1]), float(items[-1].replace("]",))
            score_array.append(score)
            sent_words.append(word)

        score_array = np.array(score_array).reshape(1,-1)

        if score_array.shape[1] <= 400:
            im = plot_score_array(None, score_array, sent_words)
            dir = 'figs/{}_{}'.format(model_name, method_name)
            if not os.path.isdir(dir): os.mkdir(dir)
            plt.savefig('figs/{}_{}/fig_{}.png'.format(model_name, method_name, i), bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    if not os.path.isdir('figs/'): os.mkdir('figs/')
    txt_file = sys.argv[1]
    path = Path(txt_file)
    if not path.exists():
        print("Path does not exit")
        exit(-1)

    visualize_tabs(path, 'bert', 'soc')
