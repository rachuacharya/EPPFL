#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
from torch.nn.functional import dropout
import torchvision

from flcore.servers.serverce import FedCE
from flcore.servers.distribution_center import DistServer


from flcore.trainmodel.models import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

warnings.simplefilter("ignore")
torch.manual_seed(0)

def assign_client_keys(distServer, AggServer):
    for c_id in range(distServer.num_clients):
        AggServer.clients[c_id].key = distServer.client_keys[c_id]
        AggServer.clients[c_id].random = distServer.random
    AggServer.sk_agg = distServer.sk_aggregation
    AggServer.enc_p = distServer.p
    AggServer.agg_public_keys = distServer.agg_public_keys
    
def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        #  Define Model and Move it to Specified device 
        if model_str == "mlr":
            # Multi Layer Perceptron
            if args.dataset == "mnist" or args.dataset == "fmnist":
                args.model = FedAvgMLP(1*28*28, num_classes=args.num_classes).to(args.device)

        elif model_str == "cnn":
            # Convolutional Neural Network
            if args.dataset == "mnist" or args.dataset == "fmnist":
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif args.dataset == "cifar10":
                # args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
                # args.model = DeepCNN(in_features=3, num_classes=args.num_classes).to(args.device)
                # args.model = FedAvgMLP().to(args.device)
                args.model = ResNet(BasicBlock, [2,2,2,2], num_classes = 10).to(args.device)
       
        elif model_str == "resnet":
            # Resnet for Tranfer Learning
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)

        else:
            raise NotImplementedError

        # Select algorithm
        if args.algorithm == "FedCE":
            AggServer = FedCE(args, i)
            args.num_param = len(AggServer.global_model_c.Packed_item) * len(AggServer.global_model_c.Packed_item[0])
            print(f"Number of Param:{args.num_param}")
            Distserver = DistServer(args)

            
        else:
            raise NotImplementedError

        # Assign Client keys
        assign_client_keys(Distserver, AggServer)
        
        # Train Server
        AggServer.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    # args.goal is  not defined
    average_data(dataset=args.dataset, 
                algorithm=args.algorithm, 
                # goal=args.goal, 
                goal = "",
                times=args.times, 
                result_ts = AggServer.result_ts,
                length=args.global_rounds/args.eval_gap+1)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-p', "--predictor", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=100) # default 1000
    parser.add_argument('-ls', "--local_steps", type=int, default=20)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedCE")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Total number of clients") # default 20
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='models')
    parser.add_argument('-r', "--r", type=float, default=0.00004)
    parser.add_argument('-tf', "--transformation", type=str, default = 'dct')

    


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))

    print("Global rounds: {}".format(args.global_rounds))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Local model: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Compresion Rate: {}".format(args.r))

    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("=" * 50)

    run(args)
