import torch

from configs import cfg


class ContentDialog:

    def __init__(self):
        if cfg.gpu_id == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{cfg.gpu_id}')

        # TODO 加载模型
        # self.model = 模型预加载

    def predict(self, content):
        if content is None:
            return

        cnt_sta = "模型结果"
        return cnt_sta
