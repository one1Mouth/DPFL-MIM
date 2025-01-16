import copy
import math

import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *


class clientCLDP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.sample_clients_num = args.join_ratio * args.num_clients
        self.batch_sample_ratio = args.batch_sample_ratio
        self.minibatch_size = int(self.train_samples * self.batch_sample_ratio)
        if args.privacy:
            self.dp_C = args.dp_C

        # 将BatchNorm层参数可微分,否则ResNet等带有bn层的模型，在这套代码里没法计算delta_model
        # 目前只能用CNN，ResNet还是用不liao，问题可能还是出在bn层不求导、model.param拿不到，可能要改用state_dict()
        # for layer in self.model.modules():
        #     if isinstance(layer, torch.nn.BatchNorm2d):
        #         for param in layer.parameters():
        #             param.requires_grad = True

        self.delta_model = copy.deepcopy(args.model)

    def train(self):
        # trainloader = self.load_train_data() # Epoch的形式

        # Minibatch的形式,采iterations个batch,默认是泊松采样
        trainloader = self.load_train_data_minibatch(minibatch_size=self.minibatch_size, iterations=1)

        self.model.train()

        global_model = copy.deepcopy(self.model)  # 存储全局模型

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        # 本地更新结束，现在去算eta_l*grad的叠加
        for delta_model_param, updated_model_param, global_model_param in zip(self.delta_model.parameters(),
                                                                              self.model.parameters(),
                                                                              global_model.parameters()):
            delta_model_param.data = updated_model_param.data.clone() - global_model_param.data.clone()

        if self.privacy:
            # 裁剪加噪
            l2_norm = 0
            for param in self.delta_model.parameters():
                l2_norm += torch.norm(param) ** 2
            l2_norm = torch.sqrt(l2_norm)
            # print("l2_norm:", l2_norm)
            alpha = min(1, self.dp_C / l2_norm)
            for param in self.delta_model.parameters():
                param.data = param.data.clone() * alpha  # 裁剪
                noise = self.dp_C * self.dp_sigma / math.sqrt(self.sample_clients_num) * torch.randn_like(
                    param.data)  # 理解为逐层的噪声？
                param.data.add_(noise)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
