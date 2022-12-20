# -*- coding: UTF-8 -*-
import logging
from fastapi import FastAPI
import uvicorn
import similarities
import math
import pandas as pd
import json

logging.getLogger().setLevel(logging.INFO)
app = FastAPI()


question_file_list = [
    # 'data/common_QA.csv',
    './data/standardQA/product_QA.csv',
    './data/standardQA/standard_QA.csv']

question_list = []
q2a = {}
for file in question_file_list:
    df = pd.read_csv(file, sep=',')
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

#全局变量 需要提前加载
sm = similarities.WordEmbeddingSimilarity(corpus=question_list)


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
