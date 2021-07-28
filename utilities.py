import xml.dom.minidom
import re
import nltk
import os
import numpy as np
import liwc
from collections import Counter

antidepressants = ['abilify','aripiprazole', 'adapin', 'doxepin','anafranil', 'clomipramine','Aplenzin', 'bupropion','asendin', 'amoxapine',
'aventyl','nortriptyline','brexipipzole','rexulti','celexa','citalopram','cymbalta','duloxetine','desyrel','trazodone','effexor','venlafaxine','emsam' ,'selegiline',
'esketamine','spravato','etrafon','elavil','amitriptyline','endep','fetzima','levomilnacipran','khedezla','desvenlafaxine','latuda','lurasidone',
'lamictal', 'lamotrigine','lexapro', 'escitalopram','limbitrol','chlordiazepoxide','marplan','isocarboxazid',
'nardil','phenelzine','norpramin','desipramine','oleptro','trazodone','pamelor','nortriptyline','parnate','tranylcypromine','paxil','paroxetine',
'pexeva','paroxetine','prozac','fluoxetine','pristiq','desvenlafaxine','remeron','mirtazapine','sarafem','fluoxetine','seroquel','quetiapine','serzone','nefazodone','sinequan','doxepin',
'surmontil','trimipramine','symbyax','tofranil', 'imipramine','triavil','trintelllix','vortioxetine','viibryd','vilazodone','vivactil','protriptyline','wellbutrin','bupropion',
'zoloft','sertraline','zyprexa','olanzapine']


LIWC_parse, category_names = liwc.load_token_parser('LIWC2015_English.dic')

def get_LIWC(tokened_text):
    liwc_counts = Counter(category for token in tokened_text for category in LIWC_parse(token))
    total_words = sum(liwc_counts.values())         # the total number of lexicon words found
    return liwc_counts,total_words


def tokenize_str(res):
    res.lower()
    res = re.sub('\n', ' ', res)
    res = res.strip()
    res = nltk.word_tokenize(res)
    words = []
    for word in res:
        if word.isalpha():
            words.append(word)
    # print('f',words)
    return words


def get_input_data(file, path):
    dom = xml.dom.minidom.parse(path + "/" + file)
    collection = dom.documentElement
    title = collection.getElementsByTagName('TITLE')
    text = collection.getElementsByTagName('TEXT')

    return title,text


FEATURE_NUM = 9

def cal_LIWC_features(posts,postnum):
    """
    calculate LIWC features of a user
    """
    x = np.zeros(FEATURE_NUM, dtype=float)

    for post in posts:
        liwc_counts, total_num = get_LIWC(post)
        x[0] += liwc_counts['function (Function Words)']
        x[1] += liwc_counts['pronoun (Pronouns)']
        x[2] += liwc_counts['i (I)']
        x[3] += liwc_counts['ppron (Personal Pronouns)']
        x[4] += liwc_counts['verb (Verbs)']
        x[5] += liwc_counts['cogproc (Cognitive Processes)']
        x[6] += liwc_counts['focuspresent (Present Focus)']

    for i in range(FEATURE_NUM-2):
        x[i] /= postnum

    return x


def process_posts(title, text, num_k):
    """
    process num_k posts of a user
    """
    res = ""
    User_text = []
    post_num = 0
    for i in range(len(title)):
        res = res + title[i].firstChild.data + text[i].firstChild.data+'\n'
        tmp = tokenize_str(title[i].firstChild.data + text[i].firstChild.data)
        if len(tmp)>0:
            User_text.append(tmp)
        post_num += 1
        ''''''
        if post_num == num_k:
            break
        ''''''
    res.lower()
    res = re.sub('\n', ' ', res)
    res = res.strip()
    res = res.split()

    # LIWC features
    feats = cal_LIWC_features(User_text,post_num)
    # emoji & antidepressants
    emoji_cnt = 0
    antidep_cnt = 0
    for word in res:
        if word==':)' or word==':(' or word=='):' or word=='(:':
            emoji_cnt += 1
        if word in antidepressants:
            antidep_cnt += 1
    feats[FEATURE_NUM-2] = emoji_cnt/post_num
    feats[FEATURE_NUM-1] = antidep_cnt

    res = ' '.join(res)
    return_str = ""
    words = nltk.word_tokenize(res)
    for word in words:
        if word.isalpha():
            return_str= return_str + word + ' '

    return return_str, post_num, feats


def process_across_usr(files, path, num_k):
    """
    process all users' num_k posts in the directory
    return: all users' text, user num, all users' features (LIWC+emoji+antidepressants)
    """
    n = 0
    res = []
    features = []
    for file in files:
        title, text = get_input_data(file, path)
        text, postnum, feats = process_posts(title, text, num_k)
        res.append(text)
        features.append(feats)
        n = n+1

    return res, n, features


def process_across_usr_for_chunk(files, path, postnum_recorder, chunknum, chunkid):
    """
    process all users' num_k posts in the directory.
    Since each user has different postnum, num_k vary according to user and chunkid.
    return: all users' text, user num, all users' features (LIWC+emoji+antidepressants)
    """
    n = 0
    res = []
    features = []
    num_k = -1
    for file in files:
        if len(postnum_recorder) != 0:
            num_k = postnum_recorder[n] * (chunkid + 1) / chunknum
        title, text = get_input_data(file, path)
        text, postnum, feats = process_posts(title, text, num_k)
        res.append(text)
        features.append(feats)
        n = n+1

    return res, n, features


def process_in_usr(file, path, tv, lda, clf):
    """
    process one user's posts one by one in order.
    return: the user's predicted depression prob at each sample
    """
    title, text = get_input_data(file, path)
    postnum = len(title)

    usrX = []

    cur_text = ""
    prevfeats = np.zeros((1, FEATURE_NUM))
    for i in range(postnum):
        post = title[i].firstChild.data + text[i].firstChild.data + '\n'
        post.lower()
        post = re.sub('\n', ' ', post)
        post = post.strip()
        cur_text += post

        tv_res = tv.transform([cur_text])
        lda_res = lda.transform(tv_res)
        tv_res = tv_res.toarray()

        liwc_counts, total_num = get_LIWC(tokenize_str(post))
        prevfeats[0][0] += liwc_counts['function (Function Words)']
        prevfeats[0][1] += liwc_counts['pronoun (Pronouns)']
        prevfeats[0][2] += liwc_counts['i (I)']
        prevfeats[0][3] += liwc_counts['ppron (Personal Pronouns)']
        prevfeats[0][4] += liwc_counts['verb (Verbs)']
        prevfeats[0][5] += liwc_counts['cogproc (Cognitive Processes)']
        prevfeats[0][6] += liwc_counts['focuspresent (Present Focus)']

        emoji_cnt = 0
        antidep_cnt = 0
        for word in post.split():
            if word == ':)' or word == ':(' or word == '):' or word == '(:':
                emoji_cnt += 1
            if word in antidepressants:
                antidep_cnt += 1

        prevfeats[0][7] += emoji_cnt
        prevfeats[0][8] += antidep_cnt * (i+1)

        curfeats = prevfeats / (i+1)
        print('cur',curfeats)
        # print(tv_res.shape,lda_res.shape,curfeats.shape)
        curX = np.squeeze(np.concatenate((tv_res, lda_res, curfeats), axis=1))
        # print('curX',curX)

        usrX.append(curX)

    usrX = np.array(usrX)
    print('usrX',usrX.shape)
    test_y_prob = clf.predict_proba(usrX)
    usr_prob = []
    for prob in test_y_prob:
        usr_prob.append(prob[1])

    print(len(usr_prob),np.sum(usr_prob))

    return usr_prob


def cal_F1(y, y_hat):
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(len(y)):
        if y[i] == 1:
            if y_hat[i] == 1:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if y_hat[i] == 1:
                fp = fp + 1
            else:
                tn = tn + 1

    p = tp / (tp + fp)
    r = tp / (tp + fn)
    F1 = 2 * p * r / (p + r)
    acc = (tn + tp) / (tn + tp + fp + fn)
    print('F1:' + str(F1) + " p:" + str(p) + " r:" + str(r) + ' acc:' + str(acc))