from torch.utils.data import Dataset
import torch.nn as nn
import pandas as pd
import numpy as np
import torch



class GetDataSet(Dataset):
    def __init__(self,filePath):
        df = pd.read_csv(filePath,header=0)#如果csv有表头，会把表头读进去
        self.features = df.iloc[:,0:-1].values.astype(np.float32)#iloc 取具体位置df，带有表头。再取values就没有表头了
        self.labels = df.iloc[:,-1].values.astype(np.int64)
        self.seqLen =  24
        self.seqshift = 6
        self.startIdx = list(range(0,len(self.labels)- self.seqLen + 1,self.seqshift ))
    def __len__(self):
        
        return len(self.startIdx)
    def __getitem__(self, idx):#循环卷积网络的dataSet 一般 返回seq_len * feature_size

        choosewin = self.startIdx[idx]

        feature = torch.tensor(self.features[choosewin:choosewin+self.seqLen])
        win_label = self.labels[choosewin:choosewin+self.seqLen]
        label = torch.tensor(win_label[0])

        return feature,label#元组由若干逗号分隔的值组成，例如 所以这里返回的是元组
    

