# -*- coding: utf-8 -*-

import os
import traceback
import uvicorn

from fastapi import FastAPI
from logger import logger
from pydantic import BaseModel
from JuggingFace.content_dialog import ContentDialog

# 指定用几张gpu和gpu设备号
gpu_id = os.getenv('GPU_ID')
if gpu_id is None or gpu_id == -1:
    gpu_id = 'cpu'
else:
    gpu_id = '0'

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
logger.info(f'set CUDA_VISIBLE_DEVICES={gpu_id}')
app = FastAPI()
# TODO 初始化模型处理类
content_dialog = ContentDialog()


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


# 对话接口
@app.post("/dialog")
def read_item(dialog: Dialog):  # todo 参数检查
    logger.info(f'req content： {dialog}')
    content = dialog.content
    req_id = dialog.req_id
    try:
        # 模型或者逻辑处理
        res = content_dialog.predict(content)
        return {"code": 200, "message": "success", "result": res, 'req_id': req_id}
    except:
        return {"code": 500, "message": traceback.format_exc(), "result": "Error", 'req_id': req_id}


# 服务启动方法
if __name__ == '__main__':
    uvicorn.run('robot_server:app', host='0.0.0.0', port=8080)