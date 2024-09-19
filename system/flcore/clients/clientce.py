import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.compresion import *
import pywt


class clientCE(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate)

        self.init = False
        self.r = args.r  
        # self.ckks_tools = ckks
        self.transformation = args.transformation

        self.alpha = 1
        self.mu = 0.001
        self.epochs = args.epochs

    def train(self):
        """
        This is where the main Client Model Training happens
        """
        trainloader = self.load_train_data()

        start_time = time.time()

        # if self.init:
        #     self.compressed_model.package_decompresion(self.r)
        #     self.compressed_model.unpack(self.model,self.device)

        # self.init = True
        # self.compressed_model.package_de(self.ckks_tools)
        if self.compressed_model.is_Compressed is True:
            self.compressed_model.package_decompresion(self.r ,self.transformation)
        self.compressed_model = copy.deepcopy(
            self.compressed_model.unpack(copy.deepcopy(self.model), self.device))

        # self.model.to(self.device)

        # Set Model to Training Mode
        self.model.train()

        #  Create a deep copy of the model to use as a fixed reference for regularization during training.
        fixed_model = copy.deepcopy(self.model)

        for epoch in range(self.epochs):
            max_local_steps = self.local_steps

            for step in range(max_local_steps):
                for i, (x, y) in enumerate(trainloader):
                    # Move data to specified Device
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    # Zeros the gradients of the optimizer.
                    self.optimizer.zero_grad()

                    # Pass the input x through the model to get the output.
                    output = self.model(x)

                    # Compute the cross-entropy loss 
                    ce_loss = self.loss(output, y)

                    # Compute the regularization loss
                    reg_loss = 0
                    fixed_params = {n: p for n, p in fixed_model.named_parameters()}
                    for n, p in self.model.named_parameters():
                        reg_loss += ((p - fixed_params[n].detach()) ** 2).sum()

                    # Combine the cross-entropy loss and regularization loss into a total loss.
                    loss = self.alpha * ce_loss + 0.5 * self.mu * reg_loss

                    # Backpropagate the loss and updates the model parameters using the optimizer.
                    loss.backward()

                    # Update the model parameters using the computed gradients.
                    self.optimizer.step()

        self.compressed_model = Packages()
        self.compressed_model.pack_up(copy.deepcopy(self.model))
        self.compressed_model.package_compresion(self.r, self.transformation)
        # self.compressed_model.package_en(self.ckks_tools)
        

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
