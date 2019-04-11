from data import dataset_RF
from data import dataset_MLP
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import xgboost as xgb
from model import MLP
import torch as t
from torchnet import meter
import time
import argparse

#随机森林实现
def rf(root , path_train , path_test):
    data_set_train = dataset_RF(root + path_train)
    data_set_test = dataset_RF(root + path_test)


    data_train , lable_train = data_set_train.getdata(train = True)
    data_test = data_set_test.getdata(train = False)
    
    clf = RandomForestClassifier(n_estimators = 200 , min_samples_split = 5 , n_jobs = -1)
    clf = clf.fit(data_train , lable_train)
    joblib.dump(clf, './model/rf.pkl')
    #clf = joblib.load('./model/rf.pkl') 

    
    res = []
    print("running")
    print(time.asctime(time.localtime(time.time())))
    res = clf.predict(data_test)
    print("finish")
    print(time.asctime(time.localtime(time.time())))
    data_set_test.save_res(res , "./images/restest.csv")

#xgb实现
def xgb_model(root , path_train , path_test):
    data_set_train = dataset_RF(root + path_train)
    data_set_test = dataset_RF(root + path_test)

    data_train , lable_train = data_set_train.getdata(train = True)
    data_test = data_set_test.getdata(train = False)

    dataD = xgb.DMatrix(data_train[0:int(0.9*len(data_train))] , lable_train[0:int(0.9*len(lable_train))])
    dataV = xgb.DMatrix(data_train[int(0.1*len(data_train)):] , lable_train[int(0.1*len(lable_train)):])
    dataT = xgb.DMatrix(data_test)

    watchlist = [(dataV,'eval'), (dataD,'train')]
    evals_result = {}   

    params = {
        'booster':'gbtree',
        'objective':'multi:softmax',
        'num_class':10,
        'eta':0.2,
        'max_depth':8,
        'subsample':0.5,
        'min_child_weight':5,
        'colsample_bytree':0.2,
        'scale_pos_weight':0.1,
        'eval_metric':'mlogloss',
        'eval_metric':'merror',
        'gamma':0.2,            
        'lambda':300
    }
    model = xgb.train(params , dataD , 750 , watchlist , evals_result = evals_result)
    preds = model.predict(dataT)
    data_set_test.save_res(preds , './images/res_XGB.csv')

#MPL实现
def mpl(root , path_train , path_test):
    data_set_train = dataset_MLP(root + path_train , train = True)
    data_set_test = dataset_MLP(root + path_test , train = False)

    trainloader = t.utils.data.DataLoader(data_set_train,batch_size=1000,shuffle=True)
    testloader = t.utils.data.DataLoader(data_set_test,batch_size=1000)

    model = MLP()

    criterion = t.nn.CrossEntropyLoss()
    lr = 0.01
    optimizer = t.optim.SGD(model.parameters() , lr , momentum = 0.4)

    for epoch in range(240):
        for _ , (data , label) in enumerate(trainloader):
            model.train()
            optimizer.zero_grad()
            score = model(data)
            loss = criterion(score , label)
            loss.backward()
            optimizer.step()
        print("Epoch:%d loss:%f" %(epoch,loss.mean()))
    
    
    res = []
    for _ , (data) in enumerate(testloader):
        model.eval()
        predict = model(data)
        predict = predict.detach().numpy().tolist()
        res += predict
    res = np.array(res)

    ans = np.argmax(res , axis = 1)
    data_set_test.save_res(ans , "./images/res_MLP.csv")
    

    





if __name__ == "__main__":

    root = "./images/"
    path_train = "train.csv"
    path_test = "test.csv"
    #rf(root , path_train , path_test)
    #xgb_model(root , path_train , path_test)
    mpl(root , path_train , path_test)
 
    