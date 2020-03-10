import os
import sys
import pandas as pd
import numpy as np
import numpy.random as npr
from glob import glob
import argparse
from nltk.tokenize import word_tokenize
import re
import matplotlib.pyplot as plt
import math
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# ADD ANY OTHER IMPORTS YOU LIKE

# DO NOT CHANGE THE SIGNATURES OF ANY DEFINED FUNCTIONS.
# YOU CAN ADD "HELPER" FUNCTIONS IF YOU LIKE.


def part1_load(folder1, folder2, n=100):
    allfiles_class1 = glob("{}/*.txt".format(folder1))
    allfiles_class2 = glob("{}/*.txt".format(folder2))

    counts = {}
    for m1 in range(0, len(allfiles_class1)):
        text = ''

        if 'file name' not in counts:
            counts['file name'] = {m1: allfiles_class1[m1][6:]}
            counts['folder name'] = {m1: folder1}
        else:
            counts['file name'][m1] = allfiles_class1[m1][6:]
            counts['folder name'][m1] = folder1

        with open(allfiles_class1[m1], "r") as thefile:
            for line in thefile:
                text += line
                #text = re.sub(r'[^\w\s]', '', text)
                #text = text.lower()

            words = word_tokenize(text)
            for word in words:
                if word not in counts:
                    counts[word] = {m1: 1}
                else:
                    if m1 not in counts[word]:
                        counts[word][m1] = 1
                    else:
                        counts[word][m1] += 1

    for m2 in range(len(allfiles_class1), len(allfiles_class1)+len(allfiles_class2)):
        length = len(allfiles_class1)
        text = ''

        if 'file name' not in counts:
            counts['file name'] = {m2: allfiles_class2[m2-length][6:]}
            counts['folder name'] = {m2: folder2}
        else:
            counts['file name'][m2] = allfiles_class2[m2-length][6:]
            counts['folder name'][m2] = folder2

        with open(allfiles_class2[m2-length], "r") as thefile:
            for line in thefile:
                text += line
                #text = re.sub(r'[^\w\s]', '', text)
                #text = text.lower()

            words = word_tokenize(text)
            for word in words:
                if word not in counts:
                    counts[word] = {m2: 1}
                else:
                    if m2 not in counts[word]:
                        counts[word][m2] = 1
                    else:
                        counts[word][m2] += 1

    frequency = pd.DataFrame(counts).fillna(0)

    y = [0, 1]
    summary = frequency.drop(frequency.columns[y], axis=1)
    ser = summary.apply(sum).sort_values(ascending=False)
    ser = ser[ser <= n]
    frequency.drop(ser.index, axis=1, inplace=True)
    return frequency


def part2_vis(df, m):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)

    folder = []
    for folder_name in df['folder name']:
        if folder_name not in folder:
            folder.append(folder_name)

    y = [0, 1]
    df_class1 = df.loc[df['folder name'] == folder[0]]
    df_class1 = df_class1.drop(df.columns[y], axis=1)
    ser_1 = df_class1.apply(sum).sort_values(ascending=False)

    df_class2 = df.loc[df['folder name'] == folder[1]]
    df_class2 = df_class2.drop(df.columns[y], axis=1)
    ser_2 = df_class2.apply(sum).sort_values(ascending=False)

    df = df.drop(df.columns[y], axis=1)
    ser = df.apply(sum).sort_values(ascending=False)[0:m]

    data = {folder[0]: ser_1[ser.index], folder[1]: ser_2[ser.index]}
    df = pd.DataFrame(data)
    df.plot(kind='bar')
    plt.legend()
    plt.show()
    return df.plot(kind="bar")


def part3_tfidf(df):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)

    columns = df.columns
    np_raw = df.iloc[:, 2:].to_numpy()
    np_extra = df.iloc[:, :2].to_numpy()

    tf = np.zeros(np_raw.shape)

    for i in range(np_raw.shape[0]):
        row_sum = np.sum(np_raw[i])
        for j in range(np_raw.shape[1]):
            tf[i][j] = (np_raw[i][j]) / row_sum

    idf = np.zeros(np_raw.shape)
    transposed_np = np_raw.T

    for i in range(transposed_np.shape[0]):
        count = list(transposed_np[i] > 0).count(True)
        idf_value = np.log((transposed_np.shape[0] * transposed_np.shape[1]) / count)
        for j in range(transposed_np.shape[1]):
            idf[j, i] = idf_value

    tf_idf = tf * idf

    df_td_idf = np.hstack((np_extra, tf_idf))
    df_td_idf = pd.DataFrame(df_td_idf).fillna(0)
    df_td_idf.columns = columns

    return df_td_idf


# ADD WHATEVER YOU NEED HERE, INCLUDING BONUS CODE.
def classifier_bonus(df):
    y = df['folder name']
    x = df.drop(['folder name', 'file name'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    classifier = SVC(kernel='linear')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FrameNet and document vectors.")
    parser.add_argument("directory1", type=str, help="The first class name.")
    parser.add_argument("directory2", type=str, help="The second class name.")
    parser.add_argument("top_m", type=int, help="The top m term frequences.")

    args = parser.parse_args()
    folder_1 = args.directory1
    folder_2 = args.directory2

    frequent = part1_load(folder_1, folder_2)
    #print(frequent)
    
    part2_vis(frequent, args.top_m)
    
    tfidf = part3_tfidf(frequent)
    
    part2_vis(tdidf, args.top_m)
    
    accurate_without = classifier_bonus(frequent)
    accurate_with = classifier_bonus(tfidf)
    print(accurate_without, accurate_with)
