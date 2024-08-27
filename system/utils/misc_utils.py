from mife.single.selective.lwe import FeLWE, _FeLWE_MK, _FeLWE_C
from mife.data.matrix import Matrix


def encode_floats(float_list, precision = 7, shift = 200):
    scaling_factor = 10 ** precision
    return [int(round((value + shift) * scaling_factor)) for value in float_list]

def decode_integers(encoded_list, num_clients, precision = 7, shift = 200):
    scaling_factor = 10 ** precision
    return [(value  - shift * num_clients)/ scaling_factor  for value in encoded_list]

def transpose_cipher(uploaded_models, num_clients):
    ar_lst = []
    num_weights = len(uploaded_models[0].export()['a_r']['M'][0])
    
    ar_m = [uploaded_models[0].a_r.M[0][wt] for wt in range(num_weights)]
    ar = Matrix([ar_m])
    
    # Transpose FLWE_C attribute c
    transposed_c = [[uploaded_models[c_id].c[wt] for c_id in range(num_clients)] 
                        for wt in range(num_weights)
                    ]
    
    # Transpose Cipher
    transposed_cipher = {
            wt: _FeLWE_C(a_r=ar, c=transposed_c[wt]) 
            for wt in range(num_weights)
    }

    return transposed_cipher