import numpy as np
import pandas as pd

class dataset_RF():
    def __init__(self , path):
        super(dataset_RF , self).__init__
        self.data = pd.read_csv(path)
        print(self.data.shape)

    def getdata(self , train = True):
        if train == True:
            return self.data.values[: , 1:] , self.data.values[: , 0]
        else:
            return self.data.values

    def save_res(self , res , path):
        indx = []
        ans = []
        for i in range(len(res)):
            indx.append(i + 1)
            ans.append(int(res[i]))
        DataSet = list(zip(indx,ans))
        df = pd.DataFrame(data = DataSet ,columns=['ImageId','Label'])
        df.to_csv(path, index=False, header=True )



    

