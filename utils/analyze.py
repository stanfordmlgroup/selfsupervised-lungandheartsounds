import numpy as np
import h5py as h5
import os

__tasks__ = ['disease', 'heartchallenge', 'heart']

for task in __tasks__:
    if task == 'disease' or task == 'crackle' or task == 'wheeze':
        base_dir = '../data'
    else:
        base_dir = '../' + task
    h5_address = os.path.join(os.getcwd(), base_dir, 'processed', '{}_1.0.h5'.format(task))
    arr = np.array(h5.File(h5_address, 'r')['train'])
    print('task: {}, mean: {:.7f}, std: {:.7f}'.format(task, arr.mean(), arr.std()))
