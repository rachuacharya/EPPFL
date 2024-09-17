import time
from flcore.clients.clientce import clientCE
from flcore.servers.serverbase import Server

class FedCE(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(args, clientCE)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.result_ts = ""


    def train(self):
        
        for i in range(self.global_rounds+1):
            print(f"Preparing Clients of Local Training Round {i}")

            s_t = time.time()
            self.selected_clients = self.select_clients()

            self.send_models()
            print(f"Global Model Sent to Clients")

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for id, client in enumerate(self.selected_clients):
                print(f"Training Client: {id}")
                client.train()

            print(f"Local Training Completed for Round {i}")
            self.receive_models_c()
            self.aggregate_parameters_c()

            print(f"Aggregation Complete for Round {i}")
          
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        print(self.rs_test_acc)

        self.save_results()
        # self.save_global_model()
