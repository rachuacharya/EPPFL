import torch
import os
import numpy as np
import h5py
import copy
from datetime import datetime

from utils.compresion import *

from utils.data_utils import read_client_data

# from seal import *


def print_parameters(context):
    context_data = context.key_context_data()
    if context_data.parms().scheme() == scheme_type.bfv:
        scheme_name = 'bfv'
    elif context_data.parms().scheme() == scheme_type.ckks:
        scheme_name = 'ckks'
    else:
        scheme_name = 'none'
    print('/')
    print('| Encryption parameters')
    print('| scheme: ' + scheme_name)
    print(
        f'| poly_modulus_degree: {context_data.parms().poly_modulus_degree()}')
    coeff_modulus = context_data.parms().coeff_modulus()
    coeff_modulus_sum = 0
    for j in coeff_modulus:
        coeff_modulus_sum += j.bit_count()
    print(f'| coeff_modulus size: {coeff_modulus_sum}(', end='')
    for i in range(len(coeff_modulus) - 1):
        print(f'{coeff_modulus[i].bit_count()} + ', end='')
    print(f'{coeff_modulus[-1].bit_count()}) bits')
    if context_data.parms().scheme() == scheme_type.bfv:
        print(
            f'| plain_modulus: {context_data.parms().plain_modulus().value()}')
    print('\\')


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.local_steps = args.local_steps
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = args.algorithm
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100

        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap


        self.r = args.r  
        self.transformation = args.transformation
        self.global_model_c = Packages()
        self.global_model_c.pack_up(copy.deepcopy(self.global_model))
        self.global_model_c.package_compresion(self.r, self.transformation)
        # Strip Enc-Dec
        # self.global_model_c.package_en(self.ckks_tools)
        self.init = False
        
        self.min_wt = float('inf')

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                            #    ckks=self.ckks_tools,
                               )
            self.clients.append(client)


    def select_clients(self):
        selected_clients = list(np.random.choice(
            self.clients, self.join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.selected_clients) > 0)
        # Strip Enc-Dec
        # self.global_model_c.package_de(self.ckks_tools)
        self.global_model_c.is_Compressed = True
        for client in self.selected_clients:
            client.compressed_model = self.global_model_c

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples)
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.compressed_model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        

    def receive_models_c(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples)
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            # self.uploaded_models.append(copy.deepcopy(
            #     client.compressed_model.Packed_item))
            self.uploaded_models.append(client.compressed_model.Packed_item_en)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters_c(self):
        assert (len(self.uploaded_models) > 0)

        temp = None
        for i in range(len(self.uploaded_models)):
            if i is 0:
                plain_w = self.ckks_tools["ckks_encoder"].encode(
                    self.uploaded_weights[i], self.ckks_tools["ckks_scale"])
                temp = self.ckks_tools["evaluator"].multiply_plain(self.uploaded_models[i], plain_w)
                # temp = self.ckks_tools["evaluator"].rescale_to_next_inplace(temp)
            else:
                plain_w = self.ckks_tools["ckks_encoder"].encode(
                    self.uploaded_weights[i], self.ckks_tools["ckks_scale"])
                temp2 = self.ckks_tools["evaluator"].multiply_plain(self.uploaded_models[i], plain_w)
                # temp2 = self.ckks_tools["evaluator"].rescale_to_next_inplace(temp2)
                temp = self.ckks_tools["evaluator"].add(temp, temp2)
        self.global_model_c.Packed_item_en = temp

        # temp = copy.deepcopy(self.uploaded_models[0])
        # for item in self.global_model_c.Packed_item:
        #     item *= 0
        # self.global_model_c *= 0
        # temp = self.ckks_tools["evaluator"].multiply(temp, 0)

        # for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
        #     temp = self.ckks_tools["evaluator"].add(
        #         temp, self.ckks_tools["evaluator"].multiply(client_model, w))
            # for server_param, client_param in zip(self.global_model_c.Packed_item, client_model.Packed_item):
            #     server_param += client_param * w

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
    
        self.global_model_c = self.uploaded_models[0]
        self.global_model_c.Packed_item = self.global_model_c.Packed_item.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
        self.global_model_c.Packed_item = torch.tensor(np.array(self.global_model_c.Packed_item.clone()) / self.num_clients)
        self.min_wt = min(self.global_model_c.Packed_item.min(), self.min_wt)
        
        print(f'Minimum Weight: {self.min_wt}')
        
        

    def add_parameters(self, w, client_model):
        for client_param in client_model.Packed_item:
<<<<<<< Updated upstream
            self.global_model_c.Packed_item += client_param.clone() * w
        
        
=======
            self.global_model_c.Packed_item += client_param  # * w
>>>>>>> Stashed changes

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(
            model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(
            model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(
            model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + str(self.times)
            file_path = result_path + f"{algo} + '_' + {self.join_clients} + '_' + {self.r}+ '_' +{str(datetime.now())}.h5"            
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(
            self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.selected_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.selected_clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.selected_clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.selected_clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(
                    acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(
                    acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(
                    acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(
                    acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True
