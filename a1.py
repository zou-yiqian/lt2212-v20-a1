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
# ADD ANY OTHER IMPORTS YOU LIKE

# DO NOT CHANGE THE SIGNATURES OF ANY DEFINED FUNCTIONS.
# YOU CAN ADD "HELPER" FUNCTIONS IF YOU LIKE.

def part1_load(folder1, folder2):
    # CHANGE WHATEVER YOU WANT *INSIDE* THIS FUNCTION.
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

def part2_vis(df):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)

    # CHANGE WHAT YOU WANT HERE
    y = [0, 1]
    df_class1 = df.loc[df['folder name'] == folder_1]
    df_class1 = df_class1.drop(df.columns[y], axis=1)
    ser_1 = df_class1.apply(sum).sort_values(ascending=False)

    df_class2 = df.loc[df['folder name'] == folder_2]
    df_class2 = df_class2.drop(df.columns[y], axis=1)
    ser_2 = df_class2.apply(sum).sort_values(ascending=False)

    df = df.drop(df.columns[y], axis=1)
    ser = df.apply(sum).sort_values(ascending=False)[0:m]

    data = {folder_1: ser_1[ser.index], folder_2: ser_2[ser.index]}
    df = pd.DataFrame(data)
    df.plot(kind='bar')
    plt.legend()
    plt.show()
    return df.plot(kind="bar")

def part3_tfidf(df):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)

    # CHANGE WHAT YOU WANT HERE
    df_class1 = df.loc[df['folder name'] == folder_1]
    df_class2 = df.loc[df['folder name'] == folder_2]

    y = [0, 1]
    df_class1 = df_class1.drop(df.columns[y], axis=1)
    ser_1 = df_class1.apply(sum).sort_values(ascending=False)
    ser_1 = ser_1[ser_1 != 0]
    N_class1 = ser_1.sum()

    df_class2 = df_class2.drop(df.columns[y], axis=1)
    ser_2 = df_class2.apply(sum).sort_values(ascending=False)
    ser_2 = ser_2[ser_2 != 0]
    N_class2 = ser_2.sum()

    df = df.drop(df.columns[y], axis=1)
    ser = df.apply(sum).sort_values(ascending=False)
    #ser = ser[ser != 0]
    N = ser.sum()

    tfidf_class1 = {'folder name': {0: folder_1}}
    for n in ser_1.index:
        tf = ser_1[n]/N_class1
        if n in ser_2.index:
            idf = math.log(2/2)
        else:
            idf = math.log(2/1)
        tfidf = tf * idf
        tfidf_class1[n] = {0: tfidf}
    df_tfidf1 = pd.DataFrame(data=tfidf_class1)

    tfidf_class2 = {'folder name': {0: folder_2}}
    for n in ser_2.index:
        tf = ser_2[n]/N_class2
        if n in ser_1.index:
            idf = math.log(2/2)
        else:
            idf = math.log(2 / 1)
        tfidf = tf * idf
        tfidf_class2[n] = {0: tfidf}
    df_tfidf2 = pd.DataFrame(data=tfidf_class2)

    df_tfidf = df_tfidf1.merge(df_tfidf2, how='outer').fillna(0)
    return df_tfidf

# ADD WHATEVER YOU NEED HERE, INCLUDING BONUS CODE.
def classifier_bonus(df):
    classifier = SVC()

    if 'file name' in df.columns:
        y = df['folder name']
        x = df.drop(['folder name', 'file name'], axis=1)
    else:
        y = df['folder name']
        x = df.drop(['folder name'], axis=1)

    classifier.fit(x, y)
    predictions = classifier.predict(x)
    result = (predictions == y)
    accuracy = sum(result.replace({True: 1, False: 0}))/len(result)

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
