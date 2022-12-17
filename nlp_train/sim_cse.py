import paddle.nn as nn
import paddle
import paddlenlp as ppnlp
import paddle.nn.functional as F
import configs as cfg


class SimCSE(nn.Layer):
    def __init__(self,
                 pretrained_model,
                 dropout=None,
                 margin=0.0,
                 scale=20,
                 output_emb_size=None):

        super().__init__()

        self.ptm = pretrained_model
        # 显式的加一个dropout来控制
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # 考虑到性能和效率，我们推荐把output_emb_size设置成256
        # 向量越大，语义信息越丰富，但消耗资源越多
        self.output_emb_size = output_emb_size
        if output_emb_size > 0:
            weight_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(std=0.02))
            self.emb_reduce_linear = paddle.nn.Linear(
                768, output_emb_size, weight_attr=weight_attr)

        self.margin = margin

        # 为了使余弦相似度更容易收敛，我们选择把计算出来的余弦相似度扩大scale倍，一般设置成20左右
        self.sacle = scale
        # 二分类计算
        self.classifier = nn.Linear(output_emb_size, 2)
        # R-Drop的损失
        self.rdrop_loss = ppnlp.losses.RDropLoss()

    # 加入jit注释能够把该提取向量的函数导出成静态图
    # 对应input_id,token_type_id两个
    @paddle.jit.to_static(input_spec=[
        paddle.static.InputSpec(
            shape=[None, None], dtype='int64'), paddle.static.InputSpec(
            shape=[None, None], dtype='int64')
    ])
    def get_pooled_embedding(self,
                             input_ids,
                             token_type_ids=None,
                             position_ids=None,
                             attention_mask=None,
                             with_pooler=True):

        # Note: cls_embedding is poolerd embedding with act tanh
        sequence_output, cls_embedding = self.ptm(input_ids, token_type_ids,
                                                  position_ids, attention_mask)

        if with_pooler == False:
            cls_embedding = sequence_output[:, 0, :]

        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(cls_embedding)

        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)

        return cls_embedding

    def get_semantic_embedding(self, data_loader):
        self.eval()
        with paddle.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data
                input_ids = paddle.to_tensor(input_ids)
                token_type_ids = paddle.to_tensor(token_type_ids)

                text_embeddings = self.get_pooled_embedding(
                    input_ids, token_type_ids=token_type_ids)

                yield text_embeddings

    def cosine_sim(self,
                   query_input_ids,
                   title_input_ids,
                   query_token_type_ids=None,
                   query_position_ids=None,
                   query_attention_mask=None,
                   title_token_type_ids=None,
                   title_position_ids=None,
                   title_attention_mask=None,
                   with_pooler=True):

        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids,
            query_token_type_ids,
            query_position_ids,
            query_attention_mask,
            with_pooler=with_pooler)

        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids,
            title_token_type_ids,
            title_position_ids,
            title_attention_mask,
            with_pooler=with_pooler)

        cosine_sim = paddle.sum(query_cls_embedding * title_cls_embedding,
                                axis=-1)
        return cosine_sim

    def forward(self,
                query_input_ids,
                title_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                title_token_type_ids=None,
                title_position_ids=None,
                title_attention_mask=None):
        # 第 1 次编码: 文本经过无监督语义索引模型编码后的语义向量
        # [N, output_emb_size]
        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_position_ids,
            query_attention_mask)

        # 第 2 次编码: 文本经过无监督语义索引模型编码后的语义向量
        # [N, output_emb_size]
        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids, title_token_type_ids, title_position_ids,
            title_attention_mask)

        # 使用R-Drop
        logits1 = self.classifier(query_cls_embedding)
        logits2 = self.classifier(title_cls_embedding)
        kl_loss = self.rdrop_loss(logits1, logits2)

        # 相似度矩阵: [N, N]
        cosine_sim = paddle.matmul(
            query_cls_embedding, title_cls_embedding, transpose_y=True)

        # substract margin from all positive samples cosine_sim()
        # 填充self.margin值，比如margin为0.2，query_cls_embedding.shape[0]=2
        # margin_diag: [0.2,0.2]
        margin_diag = paddle.full(
            shape=[query_cls_embedding.shape[0]],
            fill_value=self.margin,
            dtype=paddle.get_default_dtype())
        # input paddle.diag(margin_diag): [[0.2,0],[0,0.2]]
        # input cosine_sim : [[1.0,0.6],[0.6,1.0]]
        # output cosine_sim: [[0.8,0.6],[0.6,0.8]]
        cosine_sim = cosine_sim - paddle.diag(margin_diag)

        # scale cosine to ease training converge
        cosine_sim *= self.sacle
        # 转化成分类任务: 对角线元素是正例，其余元素为负例
        # labels : [0,1,2,3]
        labels = paddle.arange(0, query_cls_embedding.shape[0], dtype='int64')
        # labels : [[0],[1],[2],[3]]
        labels = paddle.reshape(labels, shape=[-1, 1])
        # 交叉熵损失函数
        loss = F.cross_entropy(input=cosine_sim, label=labels)

        return loss, kl_loss
