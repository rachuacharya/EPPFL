import numpy as np
import sys
import copy
import pywt
import scipy.fftpack as spfft
import math
import torch
import matplotlib.pyplot as plt
from sympy import fwht, ifwht
from mife.single.selective.lwe import FeLWE, _FeLWE_MK, _FeLWE_C
from utils.misc_utils import encode_floats

global_var = list()

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

def dct_2d(x):
    # Apply transformation to the Transpose and then tranpose it back to original orientation for another transformation
    return torch.tensor(spfft.dct(spfft.dct(x.numpy().T, norm='ortho').T, norm='ortho'))

def idct_2d(x):
    return torch.tensor(spfft.idct(spfft.idct(x.numpy().T, norm='ortho').T, norm='ortho'))

def fft_2d(x):
    return torch.tensor(spfft.rfft(spfft.rfft(x.numpy().T).T))

def ifft_2d(x):
    return torch.tensor(spfft.irfft(spfft.irfft(x.numpy().T).T))

def wav_2d(x, signal):
    coeffs = pywt.dwt2(data = x.numpy(), wavelet = signal)
    return torch.tensor(coeffs[0])

def iwav_2d(x, signal):
    coeffs = x.numpy(), (None, None, None)
    row, col = x.numpy().shape
    res = pywt.idwt2(coeffs, signal) 
    return torch.tensor(res[:row, :col])

def had_2d(x):
    return torch.tensor(fwht( x.numpy().astype(np.float32)), dtype = torch.float32)

def ihad_2d(x):
    return torch.tensor(ifwht(x.numpy().astype(np.float32)), dtype = torch.float32)

def get_u_transpose(shape):
    u_transpose = np.zeros((shape[0]**2, shape[1]**2))
    n = shape[0]
    k = 0
    i = 0
    for row in u_transpose:
        row[k+i] = 1
        k += n
        if k >= n*n:
            k = 0
            i += 1

    return u_transpose


def get_transposed_diagonals(u_transposed):
    transposed_diagonals = np.zeros(u_transposed.shape)
    for i in range(u_transposed.shape[0]):
        a = np.diagonal(u_transposed, offset=i)
        b = np.diagonal(u_transposed, offset=u_transposed.shape[0]-i)
        transposed_diagonals[i] = np.concatenate([a, b])

    return transposed_diagonals

class Packages(object):
    def __init__(self, shape=(256, 256)):
        self.Packages_shape = shape  # Package Shape
        self.Packages_size = shape[0] * shape[1]  # Package size
        self.Packed_item = None  # Packed Data
        #        self.Additional_item = None                 #不符合打包条件的数据
        self.is_Compressed = False  # Whether the packacge is compressed
        self.Volume_of_Raw_Item = None  # Original size of the package
        self.Volume_of_Compressed_Item = None  # Package size after compression 
        self.Index_of_Item = {}  # Packing Index
        self.Packed_item_en = None
        self.en_shape = None
        self.Shape_of_Compressed_Item = None

    def pack_up(self, client_model):
        """Packing Method for Client Model Parameters"""
        self.Volume_of_Raw_Item = 0

        # Initialize pointers and counters for packing
        Item_left = 0   # Left pointer for data        
        Item_right = 0  # Right pointer for data
        Package_pt = 0  # Pointer for current package position
        Package_idx = 0 # Index of the package being processed
        package = torch.zeros(self.Packages_shape, dtype=torch.float32)

        for name, param in client_model.named_parameters():
            Item_left = 0 # Reset left pointer for each parameter
            data_shape = param.shape
            data = copy.deepcopy(param.data.view(-1))   # Flatten parameter data
            size = data.size()[0]                       # size of flattened data
            self.Volume_of_Raw_Item += size             # Accumulate total volume of raw data packed
            Item_right = Item_left + size

            # If the remaining capacity of the current package is 0, push the package
            if self.Packages_size - Package_pt == 0:
                self.package_push(package)
                Package_idx += 1
                Package_pt = 0
                package = torch.zeros(self.Packages_shape, dtype=torch.float32)
            
            # Store index, size, and shape information of the packed item
            self.Index_of_Item[name] = (
                Package_idx, Package_pt, size, data_shape)  # Package index, package offset, data length, original data shape
            
            if (size <= self.Packages_size - Package_pt):  
                # If it fits, pack the data into the current package
                package.view(-1)[Package_pt:Package_pt + size] = data
                Package_pt += size
            else:
                while (Item_right is not Item_left):
                    a_size = Item_right - Item_left  # Remaining size of data to be packed
                    b_size = self.Packages_size - Package_pt  #  Remaining capacity of the current package
                    sub_size = a_size if a_size <= b_size else b_size  # Determine the size to pack in this iteration
                    package.view(-1)[Package_pt:Package_pt +
                                     sub_size] = data[Item_left:Item_left + sub_size] # Pack data into package
                    # Update pointers for data
                    Item_left += sub_size
                    Package_pt += sub_size

                    if self.Packages_size - Package_pt == 0:
                        self.package_push(package)
                        Package_idx += 1
                        Package_pt = 0
                        package = torch.zeros(
                            self.Packages_shape, dtype=torch.float32)
                    if a_size == 0:
                        break

        self.package_push(package)

    def package_push(self, package):
        """Function that adds package to Packed_Item Attribute"""
        # If packed item is empty
        if self.Packed_item is None:
            self.Packed_item = copy.deepcopy(package)
        else:
            # If Packed_item is not empty concatenate the package
            # if dimension not 3 then add new dimension to Packed_item
            if self.Packed_item.dim() != 3:
                self.Packed_item = self.Packed_item.unsqueeze(0)
            self.Packed_item = torch.cat(
                (self.Packed_item, copy.deepcopy(package).unsqueeze(0)))
        return self.Packed_item.shape[0]  
    
    def unpack(self, global_model, dev):
        if self.is_Compressed:
            return 0
       
        for name, param in global_model.named_parameters():
            # for key in self.Index_of_Item:
            data = torch.tensor([], dtype=torch.float32)
            Package_idx, Package_pt, size, data_shape = self.Index_of_Item[name]

            # if size <= self.Packages_size-Package_pt:
            #     data = self.Packed_item[Package_idx][Package_pt:Package_pt+size]
            # else:
            while size != 0:
                if size <= self.Packages_size - Package_pt:
                    data = torch.cat(
                        (data, self.Packed_item[Package_idx].view(-1)[Package_pt:Package_pt + size]), 0)
                    size = 0
                else:
                    data = torch.cat(
                        (data, self.Packed_item[Package_idx].view(-1)[Package_pt:]), 0)
                    size -= self.Packages_size - Package_pt
                    Package_pt = 0
                    Package_idx += 1

            param.data += data.reshape(data_shape).to(dev)
        return global_model

    def package_compresion(self, r, transformation):
        temp = []
        self.Volume_of_Compressed_Item = 0
        for idx in range(self.Packed_item.shape[0]):        
            if idx == 0:
                # Apply Transformation, Flatten Data, and take compressed samples of (size * compression_ratio)
                if transformation == 'dct':
                    temp = dct_2d(
                        self.Packed_item[idx]).view(-1)[:math.ceil(self.Packages_size * r)]
                elif transformation == 'fft':
                    temp = fft_2d(
                        self.Packed_item[idx]).view(-1)[:math.ceil(self.Packages_size * r)]
                elif transformation == 'wav':        
                    temp = wav_2d(
                        self.Packed_item[idx], 'db1').view(-1)[:math.ceil(self.Packages_size * r)]
                elif transformation == 'haar':        
                    temp = wav_2d(
                        self.Packed_item[idx], 'haar').view(-1)[:math.ceil(self.Packages_size * r)]
                elif transformation == 'had':        
                    temp = had_2d(
                        self.Packed_item[idx]).view(-1)[:math.ceil(self.Packages_size * r)]

            else:
                if temp.dim() == 1:
                    # Add an extra dimension to a tensor:  [n] -> [1, n]
                    temp = temp.unsqueeze(0)
                # Concatenate as one single compressed tensor
                if transformation == 'dct':
                    temp = torch.cat(
                        (temp, dct_2d(
                            self.Packed_item[idx]).view(-1)[:math.ceil(self.Packages_size * r)].unsqueeze(0))) 
                elif transformation =='fft':
                    # Concatenate as one single compressed tensor
                    temp = torch.cat(
                        (temp, fft_2d(
                            self.Packed_item[idx]).view(-1)[:math.ceil(self.Packages_size * r)].unsqueeze(0)))
                elif transformation == 'wav':
                    temp = torch.cat(
                        (temp, wav_2d(
                            self.Packed_item[idx], 'db1').view(-1)[:math.ceil(self.Packages_size * r)].unsqueeze(0)))
                elif transformation == 'haar':
                    temp = torch.cat(
                        (temp, wav_2d(
                            self.Packed_item[idx], 'haar').view(-1)[:math.ceil(self.Packages_size * r)].unsqueeze(0)))
                elif transformation == 'had':  
                    temp = torch.cat(
                        (temp, had_2d(
                            self.Packed_item[idx]).view(-1)[:math.ceil(self.Packages_size * r)].unsqueeze(0)))   
                self.Volume_of_Compressed_Item += math.ceil(self.Packages_size * r)

        self.is_Compressed = True
        # temp = torch.tensor(temp).numpy()

        self.Packed_item = temp
        self.Shape_of_Compressed_Item = self.Packed_item.shape


    def package_decompresion(self, r, transformation):
        res = []
        for idx in range(self.Packed_item.shape[0]):
            temp = torch.zeros(self.Packages_shape, dtype=torch.float32)
            temp.view(-1)[:math.ceil(self.Packages_size * r)] = self.Packed_item[idx].view(-1)
            if idx == 0:
                if transformation =='dct':
                    res = idct_2d(temp)
                elif transformation == 'fft':
                    res = ifft_2d(temp)
                elif transformation == 'wav':
                    res = iwav_2d(temp, 'db1')
                elif transformation == 'haar':
                    res = iwav_2d(temp, 'haar')
                elif transformation == 'had':
                    res = ihad_2d(temp)
            else:
                if res.dim() != 3:
                    res = res.unsqueeze(0)
                if transformation =='dct':
                    res = torch.cat((res, idct_2d(temp).unsqueeze(0)), dim=0)
                elif transformation == 'fft':
                    res = torch.cat((res, ifft_2d(temp).unsqueeze(0)), dim=0)
                elif transformation == 'wav':
                    res = torch.cat((res, iwav_2d(temp, 'db1').unsqueeze(0)), dim=0) 
                elif transformation == 'haar':
                    res = torch.cat((res, iwav_2d(temp, 'haar').unsqueeze(0)), dim=0) 
                if transformation =='had':
                    res = torch.cat((res, ihad_2d(temp).unsqueeze(0)), dim=0)

        self.is_Compressed = False
        self.Packed_item = res
        
    
    
    def package_en(self, key, random):
        matrix = copy.deepcopy(self.Packed_item.cpu().numpy())
        # Flatten and encode client model to int
        flat_model = encode_floats(matrix.flatten())
        client_cipher = FeLWE.encrypt(flat_model, key, random)
        self.Packed_item_en = client_cipher
        self.Packed_item = None
        
    def package_de(self, ckks_tools):
        p1 = ckks_tools["decryptor"].decrypt(self.Packed_item_en)
        vec = ckks_tools["ckks_encoder"].decode(p1)
        # ---------------------------------------------------------
        self.Packed_item = torch.from_numpy(copy.deepcopy(
            vec[:self.en_shape[0]*self.en_shape[1]].reshape(self.en_shape)))
        self.Packed_item_en = None