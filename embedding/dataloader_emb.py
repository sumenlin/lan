from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
from datetime import datetime, timedelta
import pickle
import os
class self_dataset(Dataset): 
    def __init__(self, N, fileName): 
        self.win = N
        X, img_size = self.setData()
        self.imgs = X
        self.img_size = img_size
        self.fileName = fileName
        
    def setData(self):
        if os.path.exists('./data_dict.pkl'):
            with open('./data_dict.pkl','rb') as f:
                _,X_train,img_size = pickle.load(f)
            print ('X_train shape:', X_train.shape)
            return X_train,img_size

        Xheads =  [] # the list of feature names in data file
        img_size = len(Xheads)
        df = pd.read_csv(self.fileName)
        df = df.drop_duplicates().reset_index(drop=True)
        data_train = df
        data_train = data_train.fillna(0) # deal with missing value
        data_train = data_train.sort_values(by=['id', 'date']).reset_index(drop=True) # default keys are id and date
        #normalization
        data_train_x = data_train.loc[:,Xheads]
        data_train_y = data_train.loc[:,['snapshot_id','date']]
        data_train_x = preprocessDataFrame(data_train_x)
        data_train = pd.concat([data_train_y, data_train_x], axis=1) 
        
        
        data_dict = {}
        X_train = []
        idl = data_train.snapshot_id.drop_duplicates().tolist()
        print ('total',len(data_train))
        
        for iddx,idd in enumerate(idl):
            print ('idd',iddx,idd)
            data_dict[idd] = {}
            date_l = data_train[data_train.snapshot_id.isin([idd])].date.tolist()
            d_ = data_train[data_train.snapshot_id.isin([idd])]
            for date_ in date_l:
                date_0 = date_
                X_d = []
                for j in range(self.win):
                    date_ = datetime.strptime(date_0, "%Y-%m-%d")
                    date_ = date_ + timedelta(days=j)
                    date_ = datetime.strftime(date_, "%Y-%m-%d")
                    d__ = d_[d_.date.isin([date_])].reset_index(drop=True)
                    if d__.empty:
                        d__ = np.zeros([img_size,])
                    else:
                        d__ = d__.loc[:,Xheads].values[0]
                    X_d.append(d__)
                
                data_dict[idd][date_0]=np.array(X_d)
                X_train.append(X_d)
        X_train = np.array(X_train)
        print('X_train shape:', X_train.shape)
        
        with open('./data_dict.pkl','wb') as f:
             pickle.dump([data_dict,X_train,img_size], f)

        return X_train,img_size

    def __getitem__(self, index):  
        return self.imgs[index]
    
    def __len__(self):
        return len(self.imgs)
        
class self_dataloader():  
    def __init__(self, batch_size, N, data_file, num_workers=1, shuffle=True):
    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.N = N
        self.img_size = 0
        self.data_file = data_file

    def run(self):
        train_dataset = self_dataset(N=self.N, fileName=self.data_file)
        self.img_size = train_dataset.img_size
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers)
        return train_loader


def preprocessDataFrame(X_train):

    df = X_train.values
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(df)
    X_train = pd.DataFrame(x_scaled,index=X_train.index, columns=X_train.columns)
    
    return X_train
