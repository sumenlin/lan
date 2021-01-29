from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
import random
import numpy as np
# from PIL import Image
import pickle
import torch
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
class noisyDataset(Dataset): 
    def __init__(self, fileName, gt, n_gt, mode, seed, noise_level,noise_type,N=3,emb=64): 
        
        self.fileName = fileName
        self.mode = mode
        X_test,y_test,X_train_train, y_train_train, X_train_val, y_train_val = self.setData(gt, n_gt, seed, noise_level, noise_type,N=N,emb=emb)
        self.train_imgs = X_train_train
        self.train_labels = y_train_train
        self.test_imgs = X_test
        self.test_labels = y_test
        self.val_imgs = X_train_val
        self.val_labels = y_train_val
        self.img_size = emb
        self.N = N
        self.n_gt = n_gt
        self.noise_level = noise_level
        self.noise_type = noise_type
        self.true_noise_level = 0

    def setData(self, gt, n_gt, seed, noise_level,noise_type='sym',N=3,emb=64):
        with open('../embedding/tmp_weight/gru_all_epoch_N'+str(N)+'_hid'+str(emb)+'.pkl','rb') as f:
            emb_dict = pickle.load(f)
        
        df = pd.read_csv(self.fileName)
        df = df.drop_duplicates().reset_index(drop=True)
        
        Yheads = [gt]
        Xheads = [] # the list of column names of features in data file
        mccs = [gt]
        print ('gt ',gt)
        df_ = df.dropna(subset=[gt]).reset_index(drop=True)
        df_ = df_.fillna(0)
        df_[gt] = df_[gt].astype(int)
        print ('gt counts', len(df_[gt].drop_duplicates().tolist()),df_[gt].drop_duplicates().tolist())
        idl = df_.index.tolist()
        train_size = int(len(idl)*0.75)
        test_size = len(idl)-train_size
        random.seed(0)
        test_idxs = random.sample(idl,test_size)
        data_train = df_[~df_.index.isin(test_idxs)].reset_index(drop=True)
        data_test = df_[df_.index.isin(test_idxs)].reset_index(drop=True)
        

        X_train = []
        for _, row in data_train.iterrows():
            tmp_X = emb_dict[row['id']][row['date']]
            X_train.append(tmp_X)
        X_train = np.array(X_train)
        X_test = []
        for _, row in data_test.iterrows():
            tmp_X = emb_dict[row['id']][row['date']]
            X_test.append(tmp_X)
        X_test = np.array(X_test)



        Y_train = data_train.loc[:,Yheads]
        Y_test = data_test.loc[:,Yheads]
        y_train = Y_train.loc[:,gt]
        y_train = np.array(y_train.tolist())
        y_test = np.array(Y_test.loc[:,gt].tolist())
        print ('gt dist', [df_[gt].tolist().count(_) for _ in range(n_gt)])
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        if noise_level <= 0:
            noise_idx = []
        else:
            _, noise_idx = next(iter(StratifiedShuffleSplit(n_splits=1,test_size=noise_level,random_state=seed).split(X_train,y_train)))
        
        y_train_noise = y_train.copy()
        np.random.seed(seed)
        for noise_idx_ in noise_idx:
            yi = y_train_noise[noise_idx_]
            if noise_type=='asym':
                tmp_yn = []
                if yi<n_gt-1:
                    tmp_yn.append(yi+1)
                if yi>0:
                    tmp_yn.append(yi-1)
            else:   
                tmp_yn = list(set(range(n_gt))-set([yi]))
            y_train_noise[noise_idx_] = tmp_yn[np.random.randint(len(tmp_yn),size=1)[0]]
            
        y_train_noise_peak = y_train_noise
        print('NOISE: level %.4f error %.4f' % (noise_level,1. - np.mean(y_train_noise_peak == y_train)))
        self.true_noise_level = 1-np.mean(y_train_noise_peak == y_train)
                
        train_idx, val_idx = next(iter(StratifiedShuffleSplit(n_splits=1, test_size=0.1,random_state=seed).split(X_train, y_train_noise_peak)))
        X_train_train = X_train[train_idx]
        y_train_train = y_train_noise[train_idx]
        X_train_val = X_train[val_idx]
        y_train_val = y_train_noise[val_idx]

        return X_test,y_test,X_train_train, y_train_train, X_train_val, y_train_val

    def __getitem__(self, index):  
        if self.mode=='train':
            img = self.train_imgs[index]
            target = self.train_labels[index]     
        elif self.mode=='test':
            img = self.test_imgs[index]
            target = self.test_labels[index]
        elif self.mode=='val':
            img = self.val_imgs[index]
            target = self.val_labels[index]            
        return img, target
    
    def __len__(self):
        if self.mode=='train':
            return len(self.train_imgs)
        elif self.mode=='test':
            return len(self.test_imgs)      
        elif self.mode=='val':
            return len(self.val_imgs)           
        
class noisyDataloader():  
    def __init__(self, batch_size, num_workers, shuffle, gt, n_gt, seed, noise_level,noise_type='sym',N=3,emb=64):
    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.gt = gt 
        self.noise_level = noise_level
        self.true_noise_level = 0
        self.img_size = emb
        self.n_gt = n_gt
        self.noise_type = noise_type
        self.N = N

   
    def run(self):
        train_dataset = noisyDataset(mode='train', gt=self.gt, n_gt=self.n_gt, seed=self.seed, noise_level=self.noise_level, noise_type=self.noise_type,N=self.N,emb=self.img_size)
        val_dataset = noisyDataset(mode='val', gt=self.gt, n_gt=self.n_gt, seed=self.seed, noise_level=self.noise_level, noise_type=self.noise_type,N=self.N,emb=self.img_size)
        test_dataset = noisyDataset(mode='test', gt=self.gt, n_gt=self.n_gt, seed=self.seed, noise_level=self.noise_level, noise_type=self.noise_type,N=self.N,emb=self.img_size)
        self.true_noise_level = test_dataset.true_noise_level
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)        
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)            
        return train_loader, val_loader, test_loader
