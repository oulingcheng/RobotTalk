import logging
import os

from easydict import EasyDict as edict

cfg = edict()
LOGGER = logging.getLogger('content_dialog')
cfg['data'] = './data/xxxx.yaml'
cfg['half'] = False
cfg['dnn'] = False
cfg['weights'] = './model/xxx.pt'
cfg['gpu_id'] = 0


# 转换为绝对路径
base_dir = os.path.dirname(os.path.realpath(__file__))
cfg.data = os.path.join(base_dir, cfg.data)
cfg.weights = os.path.join(base_dir, cfg.weights)
