import os
import numpy as np
import pandas as pd
import h5py 

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD3':
        data_path = os.path.join('./data/PeMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD4':
        data_path = os.path.join('./data/PeMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7':
        data_path = os.path.join('./data/PeMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('./data/PeMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7(L)':
        data_path = os.path.join('./data/PEMS07(L)/PEMS07L.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7(M)':
        data_path = os.path.join('./data/PEMS07(M)/V_228.csv')
        data = np.array(pd.read_csv(data_path,header=None))  #onley the first dimension, traffic flow data
    elif dataset == 'NYC-Taxi':
        data_path = os.path.join('./data/NYC-Taxi/NYC-Taxi.h5')
        df = h5py.File(data_path, 'r')
        rawdata = []
        for feature in ["pick", "drop"]:
            key = "taxi_" + feature
            data = np.array(df[key])
            rawdata.append(data)
        data = np.stack(rawdata, -1)
    elif dataset == 'CHI-Taxi':
        data_path = os.path.join('./data/CHI-Taxi/CHI-Taxi.h5')
        df = h5py.File(data_path, 'r')
        rawdata = []
        for feature in ["pick", "drop"]:
            key = "taxi_" + feature
            data = np.array(df[key])
            rawdata.append(data)
        data = np.stack(rawdata, -1)
    elif dataset == 'DC-Taxi':
        data_path = os.path.join('./data/DC-Taxi/DC-Taxi.h5')
        df = h5py.File(data_path, 'r')
        rawdata = []
        for feature in ["pick", "drop"]:
            key = "taxi_" + feature
            data = np.array(df[key])
            rawdata.append(data)
        data = np.stack(rawdata, -1)
    elif dataset == 'BOS-Bike':
        data_path = os.path.join('./data/BOS-Bike/BOS-Bike.h5')
        df = h5py.File(data_path, 'r')
        rawdata = []
        for feature in ["pick", "drop"]:
            key = "bike_" + feature
            data = np.array(df[key])
            rawdata.append(data)
        data = np.stack(rawdata, -1)
    elif dataset == 'BAY-Bike':
        data_path = os.path.join('./data/BAY-Bike/BAY-Bike.h5')
        df = h5py.File(data_path, 'r')
        rawdata = []
        for feature in ["pick", "drop"]:
            key = "bike_" + feature
            data = np.array(df[key])
            rawdata.append(data)
        data = np.stack(rawdata, -1)
    elif dataset == 'DC-Bike':
        data_path = os.path.join('./data/DC-Bike/DC-Bike.h5')
        df = h5py.File(data_path, 'r')
        rawdata = []
        for feature in ["pick", "drop"]:
            key = "bike_" + feature
            data = np.array(df[key])
            rawdata.append(data)
        data = np.stack(rawdata, -1)
    elif dataset == 'NYC-Bike':
        data_path = os.path.join('./data/NYC-Bike/NYC-Bike.h5')
        df = h5py.File(data_path, 'r')
        rawdata = []
        for feature in ["pick", "drop"]:
            key = "bike_" + feature
            data = np.array(df[key])
            rawdata.append(data)
        data = np.stack(rawdata, -1)
    elif dataset == 'DC-Taxi78':
        data_path = os.path.join('./data/DC-Taxi78/DC-Taxi78.h5')
        df = h5py.File(data_path, 'r')
        rawdata = []
        for feature in ["pick", "drop"]:
            key = "taxi_" + feature
            data = np.array(df[key])
            rawdata.append(data)
        data = np.stack(rawdata, -1)
    elif dataset == 'DC-Taxi1':
        data_path = os.path.join('./data/DC-Taxi1/DC-Taxi1.h5')
        df = h5py.File(data_path, 'r')
        rawdata = []
        for feature in ["pick", "drop"]:
            key = "taxi_" + feature
            data = np.array(df[key])
            rawdata.append(data)
        data = np.stack(rawdata, -1)
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data