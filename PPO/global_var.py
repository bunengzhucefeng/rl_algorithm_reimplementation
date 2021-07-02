import numpy as np
import torch
import time

np_default_dtype = np.float32
torch_default_dtype = torch.float32

def func_info(func):
    def cal_time(*args, **kw):
        start_time = time.time()
        out = func(*args, **kw)
        end_time = time.time()
        print('函数 ', func.__name__, ' 运行耗时', end_time-start_time, '秒', sep = '')
        return out
    return cal_time