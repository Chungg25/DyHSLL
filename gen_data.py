import numpy as np
import h5py

def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    length = len(data)
    end_index = length - horizon - window + 1
    X, Y = [], []
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index += 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index += 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def split_data_by_days(data, val_days, test_days, interval=30):
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days+val_days):-T*test_days]
    train_data = data[:-T*(test_days+val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def get_dataloader(data, tod=True, dow=True, single=True):
    # data shape: [T, N, F]
    T, N, F = data.shape

    t = 48
    # Time of day feature: [T, 1] then tile to [T, N, 1]
    tod_feature = (np.arange(T) % t) / t
    tod_feature = tod_feature[:, None]
    tod_feature = np.tile(tod_feature, (1, N))[:,:,None]  # [T, N, 1]

    # Day of week one-hot: [T, 7] then tile to [T, N, 7]
    dow = (np.arange(T) // t) % 7
    dow_onehot = np.eye(7)[dow]  # [T, 7]
    dow_onehot = np.tile(dow_onehot[:, None, :], (1, N, 1))  # [T, N, 7]

    # Ghép vào data gốc
    data_full = np.concatenate([data, tod_feature, dow_onehot], axis=-1)  # [T, N, F+1+7]

    x, y = Add_Window_Horizon(data_full, 12, 12, single)

    x_train, x_val, x_test = split_data_by_days(x, 14, 14)
    y_train, y_val, y_test = split_data_by_days(y, 14, 14)

    print('Train: ', x_train.shape, y_train.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    np.savez('data/NYC-Taxi/train.npz', x=x_train, y=y_train)
    np.savez('data/NYC-Taxi/val.npz', x=x_val, y=y_val)
    np.savez('data/NYC-Taxi/test.npz', x=x_test, y=y_test)

def create_data(file_path):
    df = h5py.File(file_path, 'r')
    rawdata = []
    for feature in ["pick", "drop"]:
        key = "taxi_" + feature
        data = np.array(df[key])
        rawdata.append(data)
    data = np.stack(rawdata, -1)
    get_dataloader(data)

create_data('data/NYC-Taxi/NYC-Taxi.h5')