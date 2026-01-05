import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

# kernel_initializer = torch.nn.init.xavier_uniform_()
#   nn.init.kaiming_normal_(self.clusters)
# bias_initializer = torch.nn.init.normal_()


# result = -torch.ones(x.shape[0], d_ff//n_elements_in_block)

# 增加一个remember初始化
# class remember(nn.Module):
#     def __init__(self):
#         super(remember, self).__init__()
#         # self._elements = elements
#         # self._mode = mode
#     def forward(self,x):
#         # if self._mode == "train":
#         #     return x,-torch.ones(x.shape[0], self._elements)
#         # return x,
#         if 'running_second_time_yes' in self.state[1]:
#             result = self.state[0]
#         else:
#             result = x
#         self.state = (x, {'running_second_time': ()})
#         return result
#
# class recall(nn.Module):
#     def __init__(self,remember_layer,elements):
#         self._remember_layer = remember_layer
#         self._elements = elements
#         super(recall, self).__init__()
#
#     def forward(self, x):
#         if (self._remember_layer.state and
#                 'running_second_time_yes' in self._remember_layer.state[1]):
#             # It's reverse_and_grad, so we pull the quant_mask from remembering layer.
#             result = self._remember_layer.state[0]
#         else:
#             result = -torch.ones(x.shape[0], self._elements)
#         return (x, result)

# class ff(nn.Module):
#     def __init__(self,d_ff , n_elements_in_block):
#         super(ff, self).__init__()
#         # self._remember = remember()
#         # self._recall = recall(self._remember,elements=d_ff // n_elements_in_block)
#         # self.state=(torch.zeros(x.shape[0], self._elements),
#         #           {'running_second_time': ()})
#         self.state = None
#     def forward(self,x):
#         if self.state == None:
#             result = -torch.ones(x.shape[0], self._elements)
#         else:
#             result = self.state[0]
#
#         if self.mode == 'train' or self.main.multiply_by_controller_output:
#             # emb, quant_mask
#             # n_in = 2, n_out = 2
#             quant_mask, mask = self.controller(x, result)
#
#             # x应该是一个元组，在model.py和model_wraper.py中定义一下
#             # quant_mask, mask, emb
#             # n_in = 3, n_out = 2
#             quant_mask, res = self.main((quant_mask, mask, x))
#             # quant_mask, emb/output
#
#             # emb/output
#             # return res
#
#         else:
#         # n_in = 2, n_out = 1
#             quant_mask = self.controller(x,result)
#         # n_in = 2, n_out = 2
#
#             quant_mask, res = self.main((quant_mask, x))
#
#             # return res
#
#         if 'running_second_time_yes' in self.state[1]:
#             result = self.state[0]
#         else:
#             result = quant_mask
#         self.state = (result, {'running_second_time': ()})
#
#         return res

class SparseFFController(nn.Module):
    def __init__(self, d_model, d_ff, n_elements_in_block, d_lowrank, temperature, mode,
                 also_return_nondiscrete_output):
    # def __init__(self,config):
        super(SparseFFController, self).__init__()
        # d_ff = config["transformer_hidden_dim"]
        # d_model = config["transformer_dim"]
        # d_lowrank = config["d_lowrank"]
        # temperature = config["temperature"]
        # mode = config["mode"]
        # n_elements_in_block = config["n_elements_in_block"]
        self.d_ff = d_ff
        self.d_lowrank = d_lowrank
        self.temperature = temperature if mode == 'train' else 0.0
        self.mode = mode
        self.n_elements_in_block = n_elements_in_block
        self.also_return_nondiscrete_output = also_return_nondiscrete_output

        assert self.d_ff % self.n_elements_in_block == 0
        self.d1 = self.d_ff // self.n_elements_in_block
        self.d2 = self.n_elements_in_block

        self.m1 = torch.nn.init.xavier_uniform_(torch.empty(d_model, d_lowrank))
        self.m2 = torch.nn.init.xavier_uniform_(torch.empty(d_lowrank, d_ff)).reshape(d_lowrank, self.d1, self.d2)
        self.mb = torch.nn.init.normal_(torch.empty(d_ff,),mean=0, std=1).reshape(self.d1, self.d2)

        # if use_bfloat16:
        self.m1 = self.m1.to(torch.bfloat16).cuda()
        self.m2 = self.m2.to(torch.bfloat16).cuda()
        self.mb = self.mb.to(torch.bfloat16).cuda()

    def forward(self, x):
        x, recalled_quant_mask = x
        x_shape = x.shape
        # 展平前的x维度：[32,2048,64]
        x = x.view(-1, x_shape[-1])
        # 展平后的x：[65536,64]
        # recalled_quant_mask的维度[32,4]
        # recalled_quant_mask.to('cuda:0')

        mask_logits = torch.einsum('bd,dl,lxy->bxy', x, self.m1, self.m2) + self.mb
        # mask_logits维度：[65536,4,32]
        if self.also_return_nondiscrete_output:
            mask_logsumexp = torch.logsumexp(mask_logits, dim=-1, keepdim=True)
            log_mask = mask_logits - mask_logsumexp
            # log_mask维度：[65536,4,32]
            mask = torch.exp(log_mask)

            if self.temperature == 0.0:
                quant_mask = torch.argmax(log_mask, dim=-1)
            else:
                u = torch.empty(mask.size(), dtype=torch.float32).uniform_(1e-6, 1.0 - 1e-6)
                g = -torch.log(-torch.log(u)).cuda()
                quant_mask = torch.argmax(log_mask + g * self.temperature, dim=-1)
                # quant_mask维度：[65536,4]
        else:
            quant_mask = torch.argmax(mask_logits, dim=-1)
        # quant_mask维度：[65536,4]
        if self.mode == 'train':
            # quant_mask.cuda()
            # recalled_quant_mask.cuda()
            # print(quant_mask.device, recalled_quant_mask.device)
            recalled_quant_mask.to(quant_mask.device)
            quant_mask = torch.where(recalled_quant_mask == -1, quant_mask, recalled_quant_mask)

        if self.also_return_nondiscrete_output:
            return quant_mask, mask
        else:
            return quant_mask


class SparseFFMain(nn.Module):
    def __init__(self, d_model, d_ff, n_elements_in_block, d_lowrank, quant_prob,
                  mode, multiply_by_controller_output, kernel_scaling):
        super(SparseFFMain, self).__init__()
        self.d_ff = d_ff
        self.d_lowrank = d_lowrank
        self.quant_prob = quant_prob
        self.n_elements_in_block = n_elements_in_block
        self.multiply_by_controller_output = multiply_by_controller_output
        self.mode = mode

        assert self.d_ff % self.n_elements_in_block == 0
        self.d1 = self.d_ff // self.n_elements_in_block
        self.d2 = self.n_elements_in_block

        # self.w1 = kernel_initializer((d_model, d_ff)).view(-1, self.d1, self.d2)
        # self.w2 = kernel_initializer((d_ff, d_model)).view(self.d2, self.d1, -1)
        # self.b2 = bias_initializer((d_model,))

        self.w1 = torch.nn.init.xavier_uniform_(torch.empty(d_model, d_ff)).view(-1, self.d1, self.d2)
        self.w2 = torch.nn.init.xavier_uniform_(torch.empty(d_ff, d_model)).view(self.d2, self.d1, -1)
        self.b2 = torch.nn.init.normal_(torch.empty(d_model,), mean=0, std=1).cuda()


        # if use_bfloat16 or big_weights_in_bfloat16:
        self.w1 = self.w1.to(torch.bfloat16).cuda()
        self.w2 = self.w2.to(torch.bfloat16).cuda()

        if kernel_scaling:
            self.w2 *= (self.n_elements_in_block ** 0.5)

    def forward(self, x):
        if self.mode == 'train' or self.multiply_by_controller_output:
            quant_mask, mask, x = x
        else:
            quant_mask, x = x
        if self.mode == 'predict':
            w1 = self.w1.permute(1, 2, 0)  # dm, d1, d2 -> d1, d2, dm
            w2 = self.w2.permute(1, 0, 2)  # d2, d1, dm -> d1, d2, dm
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])

        if self.mode == 'train':
            quant_mask = F.one_hot(quant_mask, num_classes=self.n_elements_in_block).float()
            # quant_mask.requires_grad_(False)
            quant_mask = quant_mask.detach() + mask - mask.detach()
            # select = torch.rand(1).item()
            select = torch.empty((), dtype=torch.float32).uniform_(0.0, 1.0)
            quant_mask = torch.where(select < self.quant_prob, quant_mask, mask)

            mid = torch.einsum('bd,dxy->bxy', x, self.w1) * quant_mask
            relu = F.relu(mid)
            if self.multiply_by_controller_output:
                mask_mult = torch.where(select < self.quant_prob, mask, torch.ones_like(mask))
                mask_mult = mask_mult.detach()
                relu *= mask_mult
            res = torch.einsum('bxy,yxd->bd', relu, self.w2) + self.b2
        elif self.mode == 'predict':
            batch_size = quant_mask.size(0)
            idx1 = torch.tensor([torch.arange(self._d1)] * batch_size)
            # idx1 = torch.arange(self.d1).repeat(batch_size)
            idx1 = idx1.view(-1)
            idx2 = quant_mask.view(-1)
            w = self.w1[idx1, idx2, :].view(batch_size, self.d1, -1)
            mid = torch.einsum('ai,aji->aj', x, w)
            relu = F.relu(mid)
            if self.multiply_by_controller_output:
                mask_mult = torch.gather(mask, dim=-1, index=quant_mask.unsqueeze(-1)).squeeze(-1)
                relu *= mask_mult
            v = self.w2[idx1, idx2, :].view(batch_size, self.d1, -1)
            res = torch.einsum('ai,aij->aj', relu, v) + self.b2
        else:
            quant_mask = F.one_hot(quant_mask, self.n_elements_in_block).float()
            mid = torch.einsum('bd,dxy->bxy', x, self.w1) * quant_mask
            relu = F.relu(mid)
            if self.multiply_by_controller_output:
                relu *= mask
            res = torch.einsum('bxy,yxd->bd', relu, self.w2) + self.b2

        return quant_mask, res.view(x_shape)


class SparseFF(nn.Module):
    # def __init__(self, d_ff, n_elements_in_block=32, d_lowrank=64, temperature=0.1,
    #              quant_prob=0.3,mode='train',
    #              dropout_rate=0.0, dropout_shared_axes=None, multiply_by_controller_output=False, kernel_scaling=False):
    def __init__(self,config):
        super(SparseFF, self).__init__()
        mode = config["mode"]
        multiply_by_controller_output = config["multiply_by_controller_output"]
        n_elements_in_block= config["n_elements_in_block"]
        d_lowrank = config["d_lowrank"]
        d_ff = config["transformer_hidden_dim"]
        d_model = config["transformer_dim"]
        temperature = config["temperature"]
        quant_prob = config["quant_prob"]
        kernel_scaling = config["kernel_scaling"]

        if mode == 'train' or multiply_by_controller_output:
            also_return_nondiscrete_output = True
        else:
            also_return_nondiscrete_output = False

        self.controller = SparseFFController(
            d_model= d_model ,d_ff=d_ff, n_elements_in_block=n_elements_in_block, d_lowrank=d_lowrank,
            temperature=temperature, mode=mode,
            also_return_nondiscrete_output=also_return_nondiscrete_output)

        self.main = SparseFFMain(
            d_model= d_model , d_ff=d_ff, n_elements_in_block=n_elements_in_block, d_lowrank=d_lowrank,
            quant_prob=quant_prob, mode=mode,
            multiply_by_controller_output=multiply_by_controller_output, kernel_scaling=kernel_scaling)

        self.state = None
        self._elements = d_ff//n_elements_in_block
        self.mode = mode
        # self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else None

    def forward(self, x):
        # if self.dropout is not None:
        #     x = self.dropout(x)

        if self.state == None:
            result = -torch.ones(x.shape[0] * x.shape[1], self._elements)
        else:
            result = self.state[0]

        if self.mode == 'train' or self.main.multiply_by_controller_output:
            # emb, quant_mask
            # n_in = 2, n_out = 2
            quant_mask, mask = self.controller((x, result))
            x,_ = x
            # x应该是一个元组，在model.py和model_wraper.py中定义一下
            # quant_mask, mask, emb
            # n_in = 3, n_out = 2
            quant_mask, res = self.main((quant_mask, mask, x))
            # quant_mask, emb/output

            # emb/output
            # return res

        else:
            # n_in = 2, n_out = 1
            quant_mask = self.controller((x, result))
            # n_in = 2, n_out = 2

            quant_mask, res = self.main((quant_mask, x))

            # return res

        if 'running_second_time_yes' in self.state[1]:
            result = self.state[0]
        else:
            result = quant_mask
        self.state = (result, {'running_second_time': ()})

        return res