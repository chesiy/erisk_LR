from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
import os
from utilities import process_across_usr
from ERDE_test import test_in_chunk, test_in_sample_v1, test_in_sample_v2
import numpy as np
import argparse

# nega_path="D:/speechlab/bilstm-crf/data/negative_examples_anonymous"
# posi_path="D:/speechlab/bilstm-crf/data/positive_examples_anonymous"
#
# nega_test_path = "D:/speechlab/bilstm-crf/data/negative_examples_test"
# posi_test_path = "D:/speechlab/bilstm-crf/data/positive_examples_test"


def train_model(nega_path, posi_path):
    tv = TfidfVectorizer(binary=False, decode_error='ignore', stop_words='english', min_df=2)
    lda = LatentDirichletAllocation(n_components=25, learning_offset=10., random_state=0, max_iter=50)
    clf = LogisticRegression(random_state=0, C=16, penalty='l1', class_weight={0: 0.2, 1: 0.8}, dual=False,
                             solver='liblinear')

    nega_files = os.listdir(nega_path)
    posi_files = os.listdir(posi_path)

    posires, posinum, feature1 = process_across_usr(posi_files, posi_path, -1)
    negares, neganum, feature2 = process_across_usr(nega_files, nega_path, -1)

    train_feats = np.array(feature1 + feature2)
    train_posts = posires + negares
    # TF-IDF的结果
    tf_res = tv.fit_transform(train_posts)
    # LDA的结果
    lda_res = lda.fit_transform(tf_res)
    tf_res = tf_res.toarray()

    trainX = np.concatenate((tf_res, lda_res, train_feats), axis=1)
    trainY = [1] * posinum + [0] * neganum

    clf.fit(trainX, trainY)

    return tv, lda, clf


if __name__ == '__main__':
    nega_path = "./data/negative_examples_anonymous"
    posi_path = "./data/positive_examples_anonymous"

    nega_test_path = "./data/negative_examples_test"
    posi_test_path = "./data/positive_examples_test"

    tv, lda, clf = train_model(nega_path,posi_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--chunknum", type=int, default=10)
    parser.add_argument("--sample_version", type=int, default=1)
    args = parser.parse_args()

    erde5 = 0
    erde50 = 0
    if args.chunk == 1:
        erde5, erde50 = test_in_chunk(tv, lda, clf, nega_test_path, posi_test_path,args.chunknum)
    elif args.sample_version == 1:
        erde5,erde50 = test_in_sample_v1(tv, lda, clf, nega_test_path, posi_test_path)
    elif args.sample_version == 2:
        erde5, erde50 = test_in_sample_v2(tv, lda, clf, nega_test_path, posi_test_path)

    print('erde5:', erde5, 'erde50:', erde50)

