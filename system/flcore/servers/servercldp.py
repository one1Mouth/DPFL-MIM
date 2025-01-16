import copy
import os
import time
import random

import h5py
import ujson

from flcore.clients.clientcldp import clientCLDP
from flcore.optimizers.utils.RDP.compute_dp_sgd import apply_dp_sgd_analysis
from flcore.optimizers.utils.RDP.get_max_steps import get_max_steps
from flcore.servers.serverbase import Server
from threading import Thread


def compute_epsilon_by_rounds(q, sigma, steps, delta=1e-5):
    """
    q: 客户端采样率
    sigma: 噪声乘子
    steps: 消耗隐私预算的步数
    """
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
    eps, opt_order = apply_dp_sgd_analysis(q=q,
                                           sigma=sigma,
                                           steps=steps,
                                           orders=orders,
                                           delta=delta)
    return eps


def compute_rounds_by_epsilon(q, sigma, epsilon, delta=1e-5):
    """
    q: 客户端采样率
    sigma: 噪声乘子
    epsilon: 隐私预算
    """
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
    rounds = get_max_steps(epsilon, delta, q, sigma, orders)
    return rounds


class CLDP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientCLDP)
        self.server_learning_rate = args.server_learning_rate

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma
        self.dp_epsilon = args.dp_epsilon
        self.dp_delta = args.dp_delta
        self.epsilon_list = []

        self.stop_decide_by_accuracy = args.stop_decide_by_accuracy
        self.stop_decide_by_loss = args.stop_decide_by_loss
        self.dp_decide_global_rounds = args.dp_decide_global_rounds

        if self.stop_decide_by_accuracy + self.dp_decide_global_rounds + self.stop_decide_by_loss >= 2:
            error_message = "Global rounds can not decide by more than 2 flag. (stop_decide_by_accuracy, stop_decide_by_loss, dp_decide_global_rounds)"
            raise ValueError(error_message)
        if self.stop_decide_by_accuracy or self.stop_decide_by_loss:  # 是否由 目标准确率/目标损失值 控制迭代轮数
            self.global_rounds = 2000
            self.args.global_rounds = 2000
        if self.dp_decide_global_rounds:  # 是否由隐私预算控制迭代轮数
            rounds = compute_rounds_by_epsilon(q=self.join_ratio, sigma=self.dp_sigma,
                                               epsilon=self.dp_epsilon, delta=self.dp_delta)
            self.global_rounds = rounds
            self.args.global_rounds = rounds
            print("Stop controled by dp_epsilon={}, Communication_rounds={}".format(self.dp_epsilon, rounds))
        self.target_accuracy = args.target_accuracy
        self.target_loss = args.target_loss
        self.epsilon_when_target = 0.0
        self.dp_sigma = args.dp_sigma

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model (by personalized)")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()  # 重载了这个函数，收上来的东西改成了delta_model
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()  # 重载了这个函数，聚合方式之后的delta_model多一步梯度下降的操作

            if i % self.eval_gap == 0:
                print("\n\nEvaluate global model (by global)")
                self.evaluate_server(q=0.2, test_batch_size=64)

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.privacy:
                # 计算epsilon
                self.epsilon_when_target = compute_epsilon_by_rounds(q=self.join_ratio, sigma=self.dp_sigma,
                                                                     steps=i)
                print("\nEpsilon={:.2f} when ACC at {:.4f}, Train_loss at {:.4f}, communication rounds at {}".format(
                    self.epsilon_when_target, self.rs_server_acc[-1], self.rs_train_loss[-1], i))
                self.epsilon_list.append(self.epsilon_when_target)

            if self.stop_decide_by_accuracy:
                if self.rs_server_acc[-1] >= self.target_accuracy:
                    break

            if self.stop_decide_by_loss:
                if self.rs_train_loss[-1] <= self.target_loss:
                    break

        print("\nBest accuracy (personalized). :{:.4f}".format(max(self.rs_test_acc)))
        print("\nBest accuracy (global). :{:.4f}".format(max(self.rs_server_acc)))
        print("\nAverage time cost per round: {}".format(sum(self.Budget[1:]) / len(self.Budget[1:])))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientCLDP)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))  # 这里是兼顾掉线率

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_delta_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_delta_models.append(client.delta_model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_delta_models) > 0)

        self.global_delta_model = copy.deepcopy(self.uploaded_delta_models[0])
        for param in self.global_delta_model.parameters():
            param.data.zero_()

        for w, client_delta_model in zip(self.uploaded_weights, self.uploaded_delta_models):
            self.add_parameters(w, client_delta_model)

        # 聚合完还要走一步
        for global_model_param, delta_model_param in zip(self.global_model.parameters(),
                                                         self.global_delta_model.parameters()):
            global_model_param.data += self.server_learning_rate * delta_model_param.data.clone()

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"

        # 去拿config.json的信息
        current_path = os.path.abspath(__file__)  # 获取当前脚本的绝对路径
        parent_directory = os.path.dirname(current_path)  # 找到当前脚本的父目录
        parent_directory = os.path.dirname(parent_directory)  # 找到父目录的父目录
        parent_directory = os.path.dirname(parent_directory)  # system
        root_directory = os.path.dirname(parent_directory)  # 项目根目录的绝对路径
        config_json_path = root_directory + "\\dataset\\" + self.dataset + "\\config.json"

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc) or len(self.rs_server_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            extra_msg = f"dataset = {self.dataset}, learning_rate = {self.learning_rate},\n" \
                        f"server_learning_rate={self.args.server_learning_rate},\n" \
                        f"rounds = {self.global_rounds}, local_epochs= {self.args.local_epochs} \n" \
                        f"total_num_clients = {self.num_clients}, algorithm = {self.algorithm} \n" \
                        f"client_sample_ratio = {self.join_ratio}, algorithm = {self.algorithm} \n" \
                        f"have_PD = {self.args.privacy}, dp_sigma = {self.args.dp_sigma}\n" \
                        f"dp_C = {self.args.dp_C}, epsilon = {self.args.dp_epsilon}\n" \
                        f"minibatch_sample_ratio={self.args.batch_sample_ratio}"
            with open(config_json_path) as f:
                data = ujson.load(f)

            extra_msg = extra_msg + "--------------------config.json------------------------\n" \
                                    "num_clients={}, num_classes={}\n" \
                                    "non_iid={}, balance={},\n" \
                                    "partition={}, alpha={}\n".format(
                data["num_clients"], data["num_classes"], data["non_iid"],
                data["balance"], data["partition"], data["alpha"])

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_server_acc', data=self.rs_server_acc)
                hf.create_dataset('rs_server_loss', data=self.rs_server_loss)
                hf.create_dataset('extra_msg', data=extra_msg, dtype=h5py.string_dtype(encoding='utf-8'))
                if self.privacy:
                    hf.create_dataset('epsilon_list', data=self.epsilon_list)
