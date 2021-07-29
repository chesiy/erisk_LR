from utilities import process_across_usr, process_in_usr, cal_F1, process_across_usr_for_chunk
from ERDE import ERDE_sample, ERDE_chunk
import os
import time
import xml
import numpy as np


def test_in_sample_v2(tv, lda, clf, nega_path, posi_path):
    """
    以post的条数为单位进行测试。每读入所有用户的一条新的post就重新测试一次。
    无法选择部分用户进行测试，必须用上全部用户；但可以对用户的部分post进行测试
    """
    negafiles = os.listdir(nega_path)
    posifiles = os.listdir(posi_path)

    nega_recorder = []
    posi_recorder = []
    for file in negafiles:
        dom = xml.dom.minidom.parse(nega_path + "/" + file)
        collection = dom.documentElement
        title = collection.getElementsByTagName('TITLE')
        nega_recorder.append(len(title))

    for file in posifiles:
        dom = xml.dom.minidom.parse(posi_path + "/" + file)
        collection = dom.documentElement
        title = collection.getElementsByTagName('TITLE')
        posi_recorder.append(len(title))

    postnum_recorder = posi_recorder+nega_recorder
    test_user_num = len(posifiles)+len(negafiles)

    sample_pred_probas = []
    labels = [1] * len(posifiles) + [0] * len(negafiles)

    for i in range(test_user_num):
        sample_pred_probas.append([])

    tick1 = time.time()

    for i in range(1,151):
        positest,positestnum,feature3=process_across_usr(posifiles, posi_path,i)
        negatest, negatestnum, feature4 = process_across_usr(negafiles, nega_path,i)

        test_posts = positest + negatest
        test_feats = np.array(feature3 + feature4)

        testY = [1] * positestnum + [0] * negatestnum

        test_tf = tv.transform(test_posts)
        test_lda = lda.transform(test_tf)
        test_tf = test_tf.toarray()

        testX = np.concatenate((test_tf, test_lda, test_feats), axis=1)

        test_y_hat = clf.predict(testX)
        test_y_prob = clf.predict_proba(testX)

        for idx in range(len(test_y_prob)):
            if len(sample_pred_probas[idx]) < postnum_recorder[idx]:
                sample_pred_probas[idx].append(test_y_prob[idx][1])

        print('sample', i)
        cal_F1(testY, test_y_hat)

    # print(sample_pred_probas)

    erde5 = ERDE_sample(sample_pred_probas, labels, o=5)
    erde50 = ERDE_sample(sample_pred_probas, labels, o=50)

    tick2 = time.time()
    print('time:', tick2 - tick1)

    return erde5, erde50



def test_in_sample_v1(tv, lda, clf, nega_path, posi_path):
    """
    以用户为单位，每读入用户一条新的post就重新测试一次；
    可以选择部分用户进行测试，但用上了这些用户的所有post；
    速度比v2快很多
    """
    negafiles = os.listdir(nega_path)
    posifiles = os.listdir(posi_path)

    sample_pred_probas = []
    labels = [1] * len(posifiles) + [0] * len(negafiles)

    tick1 = time.time()

    n = 0
    for usr_file in posifiles:
        usr_prob = process_in_usr(usr_file, posi_path, tv, lda, clf)
        sample_pred_probas.append(usr_prob)
        print('usr:', n)
        n += 1

    for usr_file in negafiles:
        usr_prob = process_in_usr(usr_file, nega_path, tv, lda, clf)
        sample_pred_probas.append(usr_prob)
        print('usr:', n)
        n += 1

    erde5 = ERDE_sample(sample_pred_probas, labels, o=5)
    erde50 = ERDE_sample(sample_pred_probas, labels, o=50)

    tick2 = time.time()
    print('time:', tick2 - tick1)

    return erde5, erde50


def test_in_chunk(tv, lda, clf, nega_path, posi_path, chunk_num):
    """
    把测试集分为chunk_num个chunk，每来一个chunk重新测试一次；
    """
    negafiles = os.listdir(nega_path)
    posifiles = os.listdir(posi_path)

    nega_recorder = []
    posi_recorder = []

    for file in negafiles:
        dom = xml.dom.minidom.parse(nega_path + "/" + file)
        collection = dom.documentElement
        title = collection.getElementsByTagName('TITLE')
        nega_recorder.append(len(title))

    for file in posifiles:
        dom = xml.dom.minidom.parse(posi_path + "/" + file)
        collection = dom.documentElement
        title = collection.getElementsByTagName('TITLE')
        posi_recorder.append(len(title))

    postnum_recorder = posi_recorder+nega_recorder

    chunk_pred_probas = []
    chunk_cum_posts = []
    labels = [1] * len(posi_recorder) + [0] * len(nega_recorder)

    test_user_num = len(nega_recorder)+len(posi_recorder)
    for i in range(test_user_num):
        chunk_pred_probas.append([])
        chunk_cum_posts.append([])

    tick1 = time.time()

    for chunk in range(chunk_num):
        positest,positestnum,feature3=process_across_usr_for_chunk(posifiles, posi_path,posi_recorder,chunk_num,chunk)
        negatest, negatestnum, feature4 = process_across_usr_for_chunk(negafiles, nega_path, nega_recorder, chunk_num, chunk)

        test_posts = positest + negatest
        test_feats = np.array(feature3 + feature4)

        testY = [1] * positestnum + [0] * negatestnum

        test_tf = tv.transform(test_posts)
        test_lda = lda.transform(test_tf)
        test_tf = test_tf.toarray()
        # print(test_tf.shape,test_lda.shape,type(docres),features.shape)
        testX = np.concatenate((test_tf, test_lda, test_feats), axis=1)

        test_y_hat = clf.predict(testX)
        test_y_prob = clf.predict_proba(testX)

        for idx in range(len(test_y_prob)):
            chunk_pred_probas[idx].append(test_y_prob[idx][1])
            chunk_cum_posts[idx].append(postnum_recorder[idx]*(chunk+1)/chunk_num)

        print('chunk',chunk)
        cal_F1(testY, test_y_hat)

    chunk_pred_probas = np.array(chunk_pred_probas)
    chunk_cum_posts = np.array(chunk_cum_posts)
    print(chunk_pred_probas.shape,chunk_cum_posts.shape)

    erde5 = ERDE_chunk(chunk_pred_probas,chunk_cum_posts, labels, 0.5, o=5)
    erde50 = ERDE_chunk(chunk_pred_probas,chunk_cum_posts, labels, 0.5, o=50)

    tick2 = time.time()
    print('time:', tick2 - tick1)

    return erde5,erde50
