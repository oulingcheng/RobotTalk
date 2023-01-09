# -*- coding: utf-8 -*-

import os
import traceback
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from logger import logger
from pydantic import BaseModel
from model.qa_qq import get_most_similar_answer
from model.qa_qq import search_related_questions
import requests
import json
import threading


# from nlp_dec.simcse_qa import simcse_FAQ


def simcse_FAQ():
    return ""


# from model.chitchat import chat_response


# 指定用几张gpu和gpu设备号
gpu_id = os.getenv('GPU_ID')
if gpu_id is None or gpu_id == -1:
    gpu_id = 'cpu'
else:
    gpu_id = '0'

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
logger.info(f'set CUDA_VISIBLE_DEVICES={gpu_id}')
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    # 允许跨域的源列表，例如 ["http://www.example.org"] 等等，["*"] 表示允许任何源
    allow_origins=["*"],
    # 跨域请求是否支持 cookie，默认是 False，如果为 True，allow_origins 必须为具体的源，不可以是 ["*"]
    allow_credentials=False,
    # 允许跨域请求的 HTTP 方法列表，默认是 ["GET"]
    allow_methods=["*"],
    # 允许跨域请求的 HTTP 请求头列表，默认是 []，可以使用 ["*"] 表示允许所有的请求头
    # 当然 Accept、Accept-Language、Content-Language 以及 Content-Type 总之被允许的
    allow_headers=["*"],
    # 可以被浏览器访问的响应头, 默认是 []，一般很少指定
    # expose_headers=["*"]
    # 设定浏览器缓存 CORS 响应的最长时间，单位是秒。默认为 600，一般也很少指定
    # max_age=1000
)


# 心跳测试接口
@app.get("/health")
def read_root():
    return {"Hello": "World"}


# 对话入参  格式(json)： {"content":"helloWorld", "req_id":"uuid", "pic_url":"http://www.baodu.com/机器人.jpg"}
class Dialog(BaseModel):
    # 对话内容
    content: str
    # 可追踪id
    req_id: str
    # 图片链接
    # pic_url: str


class MyThread(threading.Thread):  # 重写threading.Thread类，加入获取返回值的函数

    def __init__(self, req: Dialog):
        threading.Thread.__init__(self)
        self.dialog = req
        self.result = ''

    def run(self):  # 新加入的函数，该函数目的：
        self.result = chitchat(self.dialog)  # ①。调craw(arg)函数，并将初试化的url以参数传递——实现爬虫功能
        # ②。并获取craw(arg)函数的返回值存入本类的定义的值result中

    def get_result(self):  # 新加入函数，该函数目的：返回run()函数得到的result
        return self.result


# 对话接口
@app.post("/dialog")
def dialog(req: Dialog):  # todo 参数检查
    logger.info(f'dialog req content： {req}')
    content = req.content
    req_id = req.req_id
    try:
        thread = MyThread(req)
        thread.start()

        # 模型或者逻辑处理
        res = get_most_similar_answer(content)
        if res[0].get('score', 0) > 0.9:
            return {"code": 200, "message": "success", "result": res, 'req_id': req_id}
        else:
            thread.join()
            result = thread.get_result()
            result = result['result'].replace(' ', '')
            return {"code": 200, "message": "success", "result": [{"answer": result,
                                                                   }], 'req_id': req_id}

    except:
        logger.error(traceback.format_exc())
        return {"code": 500, "message": traceback.format_exc(), "result": "Error", 'req_id': req_id}


# 联想接口
@app.post("/association")
def association(req: Dialog):  # todo 参数检查
    logger.info(f'association req content： {req}')
    content = req.content
    req_id = req.req_id
    try:
        # 模型或者逻辑处理
        res = search_related_questions(content)
        return {"code": 200, "message": "success", "result": res[:10], 'req_id': req_id}
    except:
        logger.error(traceback.format_exc())
        return {"code": 500, "message": traceback.format_exc(), "result": "Error", 'req_id': req_id}


# 纯知识库
@app.post("/knowledge")
def knowledge(req: Dialog):  # todo 参数检查
    logger.info(f'knowledge req content： {req}')
    content = req.content
    req_id = req.req_id
    try:
        # 模型或者逻辑处理
        res = get_most_similar_answer(content)
        return {"code": 200, "message": "success", "result": res, 'req_id': req_id}
    except:
        logger.error(traceback.format_exc())
        return {"code": 500, "message": traceback.format_exc(), "result": "Error", 'req_id': req_id}


# 纯闲谈
@app.post("/chitchat")
def chitchat(req: Dialog):  # todo 参数检查
    logger.info(f'chitchat req content： {req}')
    content = req.content
    req_id = req.req_id
    try:
        headers = {'Content-Type': 'application/json'}
        req_json = {'req_id': req_id, 'content': content}
        # 模型或者逻辑处理
        res = requests.post("http://127.0.0.1:8888/chitchat", headers=headers,
                            verify=False,

                            json=req_json)
        logger.info(f'chitchat res： {res.text}')
        return json.loads(res.text)
    except Exception as e:
        logger.error(traceback.format_exc())
        return {"code": 500, "message": traceback.format_exc(), "result": "Error", 'req_id': req_id}


# 对话接口
@app.post("/simces")
def read_item(dialog: Dialog):  # todo 参数检查
    logger.info(f'req content： {dialog}')
    content = dialog.content
    req_id = dialog.req_id
    try:
        # 模型或者逻辑处理
        res = simcse_FAQ(content)
        # res = content_dialog.predict(content)
        return {"code": 200, "message": "success", "result": res, 'req_id': req_id}
    except:
        logger.error(traceback.format_exc())
        return {"code": 500, "message": traceback.format_exc(), "result": "Error", 'req_id': req_id}


# 服务启动方法
if __name__ == '__main__':
    uvicorn.run('robot_server:app', host='0.0.0.0', port=80)
