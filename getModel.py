
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
        #B,S,D
        output,_ = self.bilstm(x)
        x =  F.relu(self.fc1(output[:,-1,:]))
        logits = self.fc2(x)

        return logits
    


class TfClasser(nn.Module):
    def __init__(self, seq_len,feature_dim,d_model,n_head,dim_ff, num_layers):
        super(TfClasser,self).__init__()
        self.prj_input = nn.Linear(feature_dim,d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model,n_head,dim_ff,batch_first=True,norm_first=True,activation='gelu')
        self.encoder = nn.TransformerEncoder(self.encoder_layer,num_layers)

        self.pos_emb = nn.Parameter(torch.empty(1,seq_len+1,d_model))
        self.cls_token = nn.Parameter(torch.empty(1,d_model))#cls_token Transformer编码器中常用的分类头

        self.fc = nn.Linear(d_model,8)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self,x):
        B = x.size(0)

        #x B,S,feautre_dim  
        padding_mask = self.getmask(x)#加了padding_mask 就可以处理变长序列了  #B,S

        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device) # 给 CLS 补一列 False（CLS 不是 padding，不能被 mask）

        padding_mask = torch.cat((cls_pad,padding_mask),dim=1) #B (1 + S)


        x = self.prj_input(x)#B S D
        cls_token = self.cls_token.unsqueeze(0).expand(B,-1,-1)# [1,1,D] -> [B,1,D] 
        x = torch.cat((cls_token,x),dim=1) #B (1+S) D
        
        x = x + self.pos_emb

        enc = self.encoder(x,src_key_padding_mask=padding_mask)#B (1+S) D
        logits = self.fc(enc[:,0,:])

        return logits

    def getmask(self,x):
#🟠 1. sum_feature == 0 浮点判断不稳（最值得改） zuihao 一开始就有

        sum_feature = torch.sum(x,dim=2)#B S  在某个序列上特征全为0的认为是 padding 的0

        src_padding_mask = sum_feature == 0 #B S

        return src_padding_mask





if __name__ == '__main__':
    # lstmM1 = nn.LSTM(32,128,batch_first=True,num_layers=5,bidirectional=True)
    
    # x = torch.randn(100,16,32)#(batch_size,seq_len,feature_len/input_len)  
    # output,(h_n,c_n) = lstmM1(x)
    # c1 = output[:,-1,:]

    tfClasser = TfClasser(16,32,128,4,256,1)

    y = torch.randn(100,16,32)#(batch_size,seq_len,feature_len/input_len)  

    c2 = tfClasser(y)
    
    