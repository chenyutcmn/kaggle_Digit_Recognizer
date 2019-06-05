import torch as t
import torch.nn as nn
import pandas as pd

class dataset_LeNet():
    def __init__(self , path , train):
        super(dataset_LeNet , self).__init__()
        self.csv = pd.read_csv(path)
        self.train = train
    
        if self.train == True:
            self.labels = t.Tensor(self.csv.values[: , 0]).long()
            self.data = t.Tensor(self.csv.values[: , 1:])
        
        else:
            self.data = t.Tensor(self.csv.values)
        print(self.data.shape)
        self.data = self.data.view(-1 , 1 , 28 , 28)
        print(self.data.shape)

    def __getitem__(self , index):
        if self.train == True:
            return self.data[index] , self.labels[index]
        else:
            return self.data[index]
    
    def save_res(self , res , path):
        indx = []
        ans = []
        for i in range(len(res)):
            indx.append(i + 1)
            ans.append(int(res[i]))
        DataSet = list(zip(indx,ans))
        df = pd.DataFrame(data = DataSet ,columns=['ImageId','Label'])
        df.to_csv(path, index=False, header=True )
    
    def __len__(self):
        return len(self.data)

