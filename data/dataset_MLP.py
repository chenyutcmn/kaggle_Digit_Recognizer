import numpy as np
import pandas as pd
import torch as t

class dataset_MLP():
    def __init__(self , path , train):
        super(dataset_MLP , self).__init__()
        self.csv = pd.read_csv(path)
        self.train = train
        print(self.csv.shape)

        if train == True:
            lables = self.csv.values[: , 0]
            self.data = self.csv.values[: , 1:]
            self.lable = lables
            self.data = t.Tensor(self.data).float()
            self.lable = t.Tensor(self.lable).long()
        else:
            self.data = self.csv.values
            self.data = t.Tensor(self.data).float()


    def __getitem__(self , index):
        if self.train:
            return self.data[index] , self.lable[index]
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



    

