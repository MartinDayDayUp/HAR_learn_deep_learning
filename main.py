
from utils.init_log import Logger

from utils.init_config import getConfig

from getDataSet import GetDataSet
from getModel import getLSTMModel,getBILSTMModel,TfClasser
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import os
import matplotlib.pyplot  as plt
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix

config = getConfig('./config.yaml')
logger = Logger(config.log_path)

epochs = config.epochs
batch_size = config.batch_size
lr = config.lr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"using device: {device}")
logger.info(config)

testData = GetDataSet(config.test_file_path)
trainData = GetDataSet(config.train_file_path)

trainLoader = DataLoader(trainData,batch_size=batch_size,shuffle=True,pin_memory=True,prefetch_factor=2,num_workers=2)
testLoader = DataLoader(testData,batch_size=batch_size,pin_memory=True,prefetch_factor=2,num_workers=2)

# feature,label = dataSets[0]


def train_one_epoch(model,trainLoader,optimizer,critertion):
    model.train()

    total_loss=0
    for input,label in trainLoader:
        input = input.to(device)
        label = label.to(device)
        logits = model(input)#batch_size * input_size
        loss = critertion(logits,label)#label(batch,)一维标量 属于0~numClass-1 loss = mean( -log_softmax(logits)[range(N), label] ) 解读一下
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数
        total_loss += loss.item()*len(label)
    average_loss = total_loss /len(trainLoader.dataset)
    return average_loss

def val_one_epoch(model,valLoader,optimizer,critertion):
    model.eval()
    total_loss=0
    with torch.no_grad():
        for input,label in valLoader:
            input = input.to(device)
            label = label.to(device)
            logits = model(input)#input:(batch_size, seq_len, input_size) logits:(batch_size, num_class)
            loss = critertion(logits,label)#label_size(batch_size,) 

            total_loss += loss.item()*len(label)
            average_loss = total_loss /len(trainLoader.dataset)
        return average_loss

def save_model(checkPoint,filePath,fileName):
    torch.save(checkPoint,os.path.join(filePath,'lstm_model.pth'))



def run_train(model_name):
    if model_name == 'transFormer':
        lstmM = TfClasser(24,16,128,4,256,4).to(device)
        print('model is transformer ')
     
    else:
        lstmM = getLSTMModel().to(device)
        print('model is lstm ')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstmM.parameters(), lr=lr)

    train_loss_list = []
    test_loss_list = []
    for epId in range(epochs):
        train_loss = train_one_epoch(lstmM,trainLoader,optimizer,criterion)
        test_loss = val_one_epoch(lstmM,testLoader,optimizer,criterion)
        logger.info(f'LSTM epoch={epId} train_loss={train_loss:.2f} test_loss={test_loss:.2f}')
        test_loss_list.append(test_loss)
        train_loss_list.append(train_loss)

    checkpoint = {#只保留最后一个epoch的训练结果，实际上可以每N个epoch保存一次checkpoint
        'train_loss':train_loss_list,
        'test_loss':test_loss_list,
        'state_dict':lstmM.state_dict()
    }
    plt.figure()
    plt.plot(range(0,epochs),train_loss_list,label='train')
    plt.plot(range(0,epochs),test_loss_list,label='test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.checkpoint_path,'loss_curve.png'))
    plt.show()

    save_model(checkpoint,config.checkpoint_path,'lstm.pth')

    # BilstmM = getLSTMModel()
    # Bioptimizer = optim.Adam(BilstmM.parameters(), lr=lr)


    # for epId in range(epochs):
    #     train_loss = train_one_epoch(BilstmM,trainLoader,Bioptimizer,criterion)
    #     test_loss = val_one_epoch(BilstmM,testLoader,Bioptimizer,criterion)
    #     logger.info(f'BILSTM epoch={epId} train_loss={train_loss:.2f} test_loss={test_loss:.2f}')
    
    # save_model(checkpoint,config.checkpoint_path,'bi_lstm.pth')

def run_test(model_name):
    
    if model_name == 'transFormer':
        lstmM = TfClasser(24,16,128,4,256,4)     
        print('model is transformer ')
    else:
        lstmM = getLSTMModel().to(device)
        print('model is lstm ')


    check_path = os.path.join(config.checkpoint_path,'lstm_model.pth')
    if not os.path.exists(check_path):
        logger.info('no Model')
        return 

    check_point = torch.load(check_path)
    lstmM.load_state_dict(check_point['state_dict'])
    lstmM.eval()
    preds_list = []
    ground_list =[]

    accuracy = 0
    with torch.no_grad():
     
        for input,label in testLoader:
            input = input.to(device)
            label = label.to(device)
            logits = lstmM(input)
            preds = torch.argmax(F.softmax(logits),1)#也可以不求softmax，因为最大的值其实就是最大的softmax后的概率
            preds_list.extend(preds.cpu().numpy().tolist())
            ground_list.extend(label.cpu().numpy().tolist())
            # accuracy += torch.sum(preds == label).numpy()
            accuracy += torch.sum(preds == label).item()

    accuracy = accuracy/len(testLoader.dataset)     
    logger.info(f'total test acc = {accuracy:.2f}')
    num_classes = 8
    confusion_mat = confusion_matrix(ground_list, preds_list, labels=list(range(num_classes)))
    row_sums = confusion_mat.sum(axis=1, keepdims=True)
    normalized_confusion_mat = confusion_mat / row_sums
    normalized_confusion_mat = np.nan_to_num(normalized_confusion_mat * 100)  # Convert to percentage
        
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(normalized_confusion_mat, cmap='Blues', vmin=0, vmax=100)
    ax.set_title('Confusion Matrix (Percentage)')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(range(num_classes))
    ax.set_yticklabels(range(num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f'{normalized_confusion_mat[i, j]:.1f}',
                    ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(config.checkpoint_path,'confusion matrix.png'), dpi=150, bbox_inches='tight')
    plt.show(0)
    plt.close(fig)


if __name__ == '__main__':
    model_name = 'transFormer'
    run_train(model_name)
    run_test(model_name)


# logger.info(feature)
# print(feature)
# print(dataSets[0])






