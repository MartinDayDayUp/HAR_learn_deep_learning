import logging
from pathlib import Path
from datetime import datetime
import os

def GetFileName():

    timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fileName = f'{timeStamp}.log'


    return fileName


def Logger(filePath,logName='myapp'):
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format='%(asctime)s  -%(levelname)s - %(message)s',
    #     filename=filePath,
    #     encoding="utf-8",        # ← 中文不乱码
    # )
    
    logger = logging.getLogger(logName)
    logger.setLevel(logging.INFO)        # ← 加这一行
    formatter =  logging.Formatter('%(asctime)s  -%(levelname)s - %(message)s')
    
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    
    
    
    filePath = Path(filePath)
    filePath.mkdir(parents=True, exist_ok=True)         # 目录不存在就建,包括父目录
    fileName = GetFileName()
    filePath = os.path.join(filePath,fileName)




    console_handle = logging.StreamHandler()
    console_handle.setLevel(logging.INFO)
    console_handle.setFormatter(formatter)


    file_handle = logging.FileHandler(filePath)
    file_handle.setLevel(logging.INFO)
    file_handle.setFormatter(formatter)

    logger.addHandler(console_handle)
    logger.addHandler(file_handle)







    return logger


if __name__ =='__main__':

    logger = Logger('./out.log')
    logger.info('ccc')
    logger.info('ccc22')
    logger.info('dddd')