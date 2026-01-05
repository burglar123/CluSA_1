import torch
import torch.nn as nn
import math
import json
from torch.utils.checkpoint import checkpoint
from sklearn import random_projection
import numpy as np

class SoftmaxAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.head_dim = config["head_dim"]

    def forward(self, Q, K, V, mask):
        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
        dot = dot - 1e6 * (1 - mask[:, None, None, :])

        attn = nn.functional.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)
        return X

class NoneAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, Q, K, V, mask):
        return V

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.grad_checkpointing = config["attention_grad_checkpointing"]

        self.dim = config["transformer_dim"]
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.attn_type = config["attn_type"]

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)
        # self.kk = config["kk"]
        self.kk = config["dota_k"]
        self.kk_dim = config["kk_dim"]

        self.P_lr = random_projection.SparseRandomProjection(n_components=self.kk, density=1/3.0)
        self.W_qhat = nn.Linear(self.kk, self.num_head * self.kk_dim)
        self.W_khat = nn.Linear(self.kk, self.num_head * self.kk_dim)


        if self.attn_type == "softmax":
            self.attn = SoftmaxAttention(config)
        elif self.attn_type == "none":
            self.attn = NoneAttention(config)
        elif self.attn_type.startswith("sbm"):
            from attention_sbm import SBMAttention
            self.attn = SBMAttention(config)
        elif self.attn_type.startswith("linformer"):
            from attention_linformer import LinformerAttention
            self.attn = LinformerAttention(config)
        elif self.attn_type.startswith("reformer"):
            from attention_reformer import LSHAttention
            self.attn = LSHAttention(config, self.W_q, self.W_k, self.W_v)
        elif self.attn_type.startswith("nystrom"):
            from attention_nystrom import NystromAttention
            self.attn = NystromAttention(config)
        elif self.attn_type.startswith("performer"):
            from attention_performer import PerformerAttention
            self.attn = PerformerAttention(config)
        elif self.attn_type.startswith("linear"):
            from attention_linear import LinearAttention
            self.attn = LinearAttention(config)

        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, inputs):
        
        X, mask = inputs

        if self.attn_type.startswith("longformer") or self.attn_type.startswith("reformer"):
            with torch.cuda.amp.autocast(enabled = False):
                attn_out = self.attn(X.float(), mask.float())
        elif self.attn_type.startswith("sbm"):

            # batch_size = X.shape[0]
            # print(X.shape)#32 2048 64
            # print(batch_size)#32
            # Xqlist = []
            # for i in range(batch_size):
            #     # 获取当前批次的数据
            #     X_batch = X[i]
            #     # 对当前批次的数据进行投影
            #     X_batch_projected = self.P_lr.fit_transform(X_batch.cpu().detach().numpy())
            #     # 将投影后的结果添加到列表中
            #     Xqlist.append(X_batch_projected)

            # Xq = np.concatenate(Xqlist, axis=0)
            # Xq = torch.from_numpy(Xq)
            # print(Xq.shape)#65536 32 
            # print(type(Xq))
            # exit(0)
            # Xq = self.P_lr.fit_transform(X.cpu().detach().numpy())
            # QH = self.split_heads(self.W_qhat(Xq.detach().numpy()))
            # KH = self.split_heads(self.W_khat(Xq.detach().numpy()))
            # t = self.W_k(X)
            # print(t.shape)#([32, 2048, 128])


            b,n,_ = X.shape
            Xq = np.zeros((b, n, self.kk))
            for i in range(b):
                Xq[i] = self.P_lr.fit_transform(X[i].cpu().detach().numpy())
            Xq = torch.tensor(Xq, dtype=torch.half).cuda()
            QH = self.split_kkheads(self.W_qhat(Xq))
            KH = self.split_kkheads(self.W_khat(Xq))


            # print(Xq.shape)
            # exit(0)
            Q = self.split_heads(self.W_q(X))
            K = self.split_heads(self.W_k(X))
            V = self.split_heads(self.W_v(X))
            # print(Q.shape)#([32, 4, 2048, 32])
            # exit(0)

            # print(type(Xq))
            # exit(0)
            with torch.cuda.amp.autocast(enabled = False):
                if self.grad_checkpointing:
                    attn_out, sparsity = checkpoint(self.attn, Q.float(), K.float(), V.float(), mask.float())
                else:
                    attn_out, sparsity = self.attn(Q.float(), K.float(), V.float(), mask.float(), QH.float(), KH.float())
                    # attn_out, sparsity = self.attn(Q.float(), K.float(), V.float(), mask.float())
            attn_out = self.combine_heads(attn_out)

            out = self.ff(attn_out)
            return out, sparsity
        else:
            Q = self.split_heads(self.W_q(X))
            K = self.split_heads(self.W_k(X))
            V = self.split_heads(self.W_v(X))
            
            with torch.cuda.amp.autocast(enabled = False):
                if self.grad_checkpointing:
                    attn_out = checkpoint(self.attn, Q.float(), K.float(), V.float(), mask.float())
                else:
                    attn_out = self.attn(Q.float(), K.float(), V.float(), mask.float())
            attn_out = self.combine_heads(attn_out)

        out = self.ff(attn_out)

        return out


    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X

    def split_kkheads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.kk_dim)
        X = X.transpose(1, 2)
        return X