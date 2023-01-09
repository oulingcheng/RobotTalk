# -*- coding: UTF-8 -*-
import logging
from fastapi import FastAPI
import configs as cfg
import uvicorn
import similarities
import math
import pandas as pd
import json
import time

import jieba  # 使用结巴分词器
import numpy
import text2vec
import os

logging.getLogger().setLevel(logging.INFO)
app = FastAPI()

question_file_list = [
    'data/standardQA/common_QA.csv',
    'data/standardQA/product_QA_1.csv',
    'data/standardQA/standard_QA.csv',
]
abs_path = cfg.base_dir

question_list = []
q2a = {}
for file in question_file_list:
    df = pd.read_csv(abs_path + "/" + file, sep=',')
    df = df.dropna()
    for index, data in df.iterrows():

        q = data.get('question', None)
        a = data.get('answer', None)
        if q2a.get(q, None) is not None:
            continue
        if q == "" or q is None or a == "" or a is None:
            continue
        q2a[q] = a
        question_list.append(q)
    logging.info(f"file:{file},question.size:{len(question_list)}")
logging.info(f"question.size:{len(question_list)}")

# 全局变量 需要提前加载
start_time = time.time()
sm = similarities.WordEmbeddingSimilarity(corpus=question_list)
end_time = time.time()
logging.info(f"corpus embedding take time:{end_time - start_time}s")


def get_most_similar_answer(question):
    try:
        logging.info(f"req:{question}")
        similar_result = sm.most_similar(queries=question, topn=5)
        result = []
        for k in similar_result.get(0):
            score = similar_result.get(0)[k]
            most_similar_question = sm.corpus.get(k)
            result.append({"most_similar_question": most_similar_question, "answer": q2a.get(most_similar_question),
                           "score": score})
        logging.info(f"rsp:{result}")
        return result
    except Exception as e:
        logging.error(f"error:{e}")
    return ""


corpus = list(q2a.keys())
stopwords = text2vec.load_stopwords('stopwords.txt')

vocabulary = set()
corpus_tokens = []
for sentence in corpus:
    # 分词
    tokens = jieba.cut(sentence)
    # 去除停用词
    tokens_without_stops = [token for token in tokens if token not in stopwords]
    corpus_tokens.append(tokens_without_stops)
    for word in tokens_without_stops:
        vocabulary.add(word)

vocabulary = sorted(list(vocabulary))

search_dict = {}
for word in vocabulary:
    search_dict[word] = []
    for i, tokens in enumerate(corpus_tokens):
        count = tokens.count(word)
        if count != 0:
            search_dict[word].append([i, count])


def search_related_questions(question):
    # 进行检索
    question_tokens = jieba.cut(question)
    question_tokens_without_stops = [token for token in question_tokens if token not in stopwords]

    # 相关的问题有
    related_question = []
    for token in question_tokens_without_stops:
        if token in search_dict.keys():
            related_question += search_dict[token]

    # 相关问题匹配到的次数
    related_question_dict = {}
    for id_count_pair in related_question:
        if id_count_pair[0] in related_question_dict:
            related_question_dict[id_count_pair[0]] += id_count_pair[1]
        else:
            related_question_dict[id_count_pair[0]] = id_count_pair[1]

    # 进行排序
    sorted_question = sorted(related_question_dict.items(), key=lambda item: item[1], reverse=True)

    # 输出相似问题排名
    related_question_str = []
    for question in sorted_question:
        related_question_str.append(corpus[question[0]])
    return related_question_str
