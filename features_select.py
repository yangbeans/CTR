from sklearn.feature_selection import RFECV
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

#特征选择函数，返回筛选特征后的训练集、测试集和有用特征的columns
class SelcetFeatures():
    def __init__(self, X, y, step=1, cv=10):   ###***设置默认值，设置的参数调用时不是必要填的参数，step的默认值为1， cv的默认值为10
        self.X = X
        self.y = y
        self.step = step
        self.cv = cv
    
    #返回有用特征的columns
    def selectFeatures(self, select_model):
        selector = RFECV(estimator=select_model, step=self.step, cv=self.cv)
        select_X = selector.fit_transform(self.X, self.y)
        select_features_index = selector.get_support(True)  
        select_columns = self.X.columns[select_features_index]
        return select_X, select_columns
    
    #返回特征选择后的训练集和测试集
    def newTrainAndTest(self, train, test, select_model, train_useful_colunms, test_useful_colunms):
        print(1)
        select_X, select_columns = self.selectFeatures(select_model)
        print(2)
        train_select_columns = list(select_columns) + list(train_useful_colunms)   ###***运算时注意数据格式
        test_select_columns = list(select_columns) + list(test_useful_colunms) 
        n_train = train[train_select_columns]
        n_test = test[test_select_columns]
        return n_train, n_test


if __name__ == "__main__":
    train = pd.read_csv("data/round1_ijcai_18_train_20180301.txt", sep=" ")
    test = pd.read_csv("data/round1_ijcai_18_test_b_20180418.txt", sep=" ")
    y_train = train["is_trade"]
    X_train = train.drop(["is_trade"], axis=1, inplace=False)  ###***设置drop后原数据内存不变，inplace=False 
    
    sf = SelcetFeatures(X_train, y_train, 1, 10)
    #select_columns = sf.selectFeatures(lgb.LGBMRegressor())
    print("start")
    new_train, new_test = sf.newTrainAndTest(train, test, lgb.LGBMRegressor(), ["card_id", "target"], ["card_id"])
    
    new_train.to_csv("features_data/train_select.csv", index=False)
    new_test.to_csv("features_data/test_select.csv", index=False)
