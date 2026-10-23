
import torch
import torch.nn as nn
import torch.nn.functional as F


class getLSTMModel(nn.Module):
    def __init__(self):
        super(getLSTMModel,self).__init__()


        self.lstm = nn.LSTM(16,128,batch_first=True)
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,8)
        

    def forward(self,x):
        lstm_out,_ = self.lstm(x)#(batch_size,seq_len,feature_len)  返回
        x = F.relu(self.fc1(lstm_out[:,-1,:])) #batch_size,feature_len
        logits = self.fc2(x)
        return logits



class getBILSTMModel(nn.Module):
    def __init__(self):
        super(getBILSTMModel,self).__init__

        self.bilstm = nn.LSTM(16,128,batch_first=True,bidirectional=True)

        self.fc1 = nn.Linear(128*2,64)

        self.fc2 = nn.Linear(64,8)





    def forward(self,x):
        output,_ = self.bilstm(x)
        x =  F.relu(self.fc1(output[:,-1,:]))
        logits = self.fc2(x)

        return logits
    



if __name__ == '__main__':
    lstmM1 = nn.LSTM(32,128,batch_first=True,num_layers=5,bidirectional=True)
    
    x = torch.randn(100,16,32)#(batch_size,seq_len,feature_len/input_len)  
    output,(h_n,c_n) = lstmM1(x)
    c1 = output[:,-1,:]
    
    
    