from scipy.io import arff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

train_dataset = arff.loadarff('C:/Users/amade/Documents/GitHub/Time_series_with_lstm_and_vaes/ECG/ECG5000_TRAIN.arff')
train_data = np.array(train_dataset[0])
train_df = pd.DataFrame(train_data)

test_dataset = arff.loadarff('C:/Users/amade/Documents/GitHub/Time_series_with_lstm_and_vaes/ECG/ECG5000_TEST.arff')
test_data = np.array(test_dataset[0])
test_df = pd.DataFrame(test_data)

print(f'train_data shape is {train_df.shape} and test data shape is {test_df.shape}')

data = train_df.append(test_df)
data = data.sample(frac = 1.0)


new_columns = list(data.columns)
new_columns[-1] = 'target'
data.columns = new_columns


print(f'Number of case per class = {list(data.target.value_counts())}')

data.target.value_counts().plot.barh()
plt.show()


def dataset():
    new_class_names = ['Normal', 'R on T', 'PVC', 'SP', 'UB']
    old_class_names = list(data.target.unique())
    rename_dct = {}
    for i in range(5):
        rename_dct[old_class_names[i]] = new_class_names[i]
    data.target = data.target.replace(rename_dct)
    normal_df = data[data['target'] == 'R on T'].drop(labels = 'target', axis = 1)
    anomaly_df = data[data['target'] != 'R on T'].drop(labels = 'target', axis = 1)
    print(list(data.target.unique()))
    print(normal_df.shape)
    print(rename_dct)
    train_df, val_df = train_test_split(normal_df, test_size=0.15, random_state=42)
    val_df, test_df = train_test_split(val_df, test_size=0.15, random_state=42)

    def create_sequence(df):
        #encode target column
        sequences = df.astype(np.float32).to_numpy().tolist()
        data = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
        n_seq, seq_len, n_features = torch.stack(data).shape

        return data, seq_len, n_features
    
    train, seq_len, n_features = create_sequence(train_df)
    val, _, _ = create_sequence(val_df)
    test_normal, _, _ = create_sequence(test_df)
    test_anomaly, _, _ = create_sequence(anomaly_df)

    return train, seq_len, n_features, val, test_anomaly, test_normal




