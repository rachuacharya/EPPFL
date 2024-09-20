# Import Dependencies
import sys
import gmpy2
import torch
import random
import numpy as np
from secrets import randbelow
from mife.data.matrix import Matrix
from mife.data.zmod_r import ZmodR
from Crypto.Util.number import getPrime
from mife.single.selective.lwe import FeLWE, _FeLWE_MK, _FeLWE_C
        
class DistServer(object):
    def __init__(self, args):
        # Set up the main attributes
        self.num_clients = args.num_clients
        self.num_weights = args.num_param
        self.msg_bit, self.func_bit = (4, 4)
        self.l = self.num_weights
        self.n = self.num_weights
        
        print(self.num_weights)
        print(self.l)
        print(self.n)

        # Intialize Encryption Parameters
        self.p = getPrime((self.msg_bit + self.func_bit) * 2 + self.l.bit_length() + 1)
        self.q = getPrime(self.p.bit_length() + self.n.bit_length() * 2 + (self.msg_bit + self.func_bit) + self.l.bit_length() // 2)
        self.G = ZmodR(self.q)
        
        m = 2 * (self.l + self.n + 1) * self.q.bit_length() + 1
        self.A = Matrix([[self.G(random.randrange(self.q)) for _ in range(self.n)] for _ in range(m)])

        # Generate Client Keys
        self.client_keys = dict()
        print(f"Generating Client Keys ....")
        for c_id in range(self.num_clients):
            self.client_keys[c_id] = FeLWE.generate(self.p, self.q, self.l, self.G, self.A,  self.msg_bit, self.func_bit, self.n)

          
        # Should be created by client themselves
        self.random = Matrix([self.client_keys[0].G(randbelow(2)) for _ in range(self.client_keys[0].m)])
      
        # Generate Functional Decryption Key
        print(f'Transposing the Keys')
        transposed_keys = [
            _FeLWE_MK(
                p=self.p, q=self.q, l=self.num_clients, n=self.n, m=m, 
                delta=round(self.q/self.p), G=self.G, A=self.A, 
                mpk=[self.client_keys[c_id].mpk[wt] for c_id in range(self.num_clients)], 
                msk=[self.client_keys[c_id].msk[wt] for c_id in range(self.num_clients)]
            ) 
            for wt in range(self.num_weights)
        ]    

        print(f'Getting al the public keys')
        self.agg_public_keys = [transposed_keys[wt].get_public_key() for wt in range(self.num_weights)]
        
        print('Getiing aggregation keys')
        y = [1] * self.num_clients
        self.sk_aggregation = [FeLWE.keygen(y, transposed_keys[wt]) for wt in range(self.num_weights)]