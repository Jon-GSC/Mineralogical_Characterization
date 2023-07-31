'''
 Language: Python 3.8
 Coding: utf-8
 Note: 1. This code is a simplified version; this function is used for loading well logging data.
 Acknowledgements: appreciate for sharing the open source code and library.
'''
# -----------------------------------------------------------------------------------------------------------------------
import os
import sys
import re
import time
import pickle
import numpy as np
import pandas as pd
import lasio
from itools import forWeb_functions as fun
# -----------------------------------------------------------------------------------------------------------------------
str_list = ['DEPT', 'GR', 'RD', 'DTP', 'DEN', 'PE', 'WQFM', 'WCARB', 'WCLAY', 'WPYR',]
GR_index = ['GR', 'GRD', 'SGR', 'GR_EDTC',]
RD_index = ['RD', 'RM', 'RS', 'M2RX',]
DTP_index = ['DTP', 'DT', 'AC',]
DEN_index = ['DEN', 'RHOB',]
PE_index = ['PE', 'PEF', 'PEFH',]

data_folder = "/data/"
slst = ['28289.las', ]
save_folder = 'results'
training_file = 'Training_all(test).csv'
# -----------------------------------------------------------------------------------------------------------------------
t_start = time.time()
savepath0 = os.getcwd()
savepath = os.path.join(savepath0, save_folder)
if not os.path.exists(savepath):
    os.mkdir(savepath)
# -----------------------------------------------------------------------------------------------------------------------
def main():
    d_slice_all = pd.DataFrame(columns=[])
    for i, fn in enumerate(slst):
        print(i, ' ---logs name: ', fn)
        d_slice_temp = pd.DataFrame(columns=[])
        las = lasio.read(savepath0 + data_folder + fn)
        d_slice0, tops = fun.Las_slice(las)
        d_slice0['DEPT'] = d_slice0.index
        log_list = fun.idenfity_logs(d_slice0, GR_index, RD_index, DTP_index, DEN_index, PE_index)
        if len(log_list) == 10:
            d_slice_temp = d_slice0[log_list]
            d_slice_temp.rename(columns=dict(zip(log_list, str_list)), inplace=True)
        d_slice_temp.insert(loc=0, column='ID', value=fn[:-4])
        d_slice_all = pd.concat([d_slice_all, d_slice_temp], ignore_index=True, sort=False)
    d_slice_all.to_csv(savepath0 + data_folder + training_file)
    print(f'Time used:{(time.time() - t_start) / 60} minutes')

if __name__ == '__main__':
    main()
