from egg.py import *
from xgboost.sklearn import XGBClassifier


def predict(train, test):
    train = itemCategory(train)
    train = weekdayDayHour(train)
    train = extra_feature(train)
    train = zuheFeature(train)
    train = shopFenduan(train)
    train = calTimeReduce(train)
    train = calTimeReducUserItem(train)
    train = calTimeReducUserShop(train)
    train = slidingWindow(train)
    train = dataFilter(train)
    y_train = train["is_trade"]
    X_train = train.drop(["is_trade"], axis=1)
    
    test = itemCategory(test)
    test = weekdayDayHour(test)
    test = extra_feature(test)
    test = zuheFeature(test)
    test = shopFenduan(test)
    test = calTimeReduce(test)
    test = calTimeReducUserItem(test)
    test = calTimeReducUserShop(test)
    test = slidingWindow(test)
    test = dataFilter(test)
    X_test = test.drop("instance_id", axis=1)
    
    xgb_model = XGBClassifier(gamma=0,max_depth=4,min_child_weight=2,subsample=1)
    ###***lgb训练的时候如果不是数值型的数据必须是矩阵或list的形式
    
    #k折交叉验证
    skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
    lgb_avg_pred = np.zeros(X_test.shape[0])
    
    for index, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        print(index)
        print("模型训练中...")
        xgb_model.fit(np.array(X_train.iloc[train_index]), y_train.iloc[train_index])
        print("训练完毕！")
        y_predpro_lgb = xgb_model.predict_proba(np.array(X_test))[:, 1]
        lgb_avg_pred += y_predpro_lgb/5
        
    fis = xgb_model.feature_importances_
    f_columns = X_train.columns    
    
    result_df = pd.read_csv("data/submit_example.csv")
    result_df["Tag"] = lgb_avg_pred
    
    #pred_data_rule = ruleForscore1()
    #result_df = ruleForscore2(result_df, pred_data_rule)
    
    result_df.to_csv("CX_result_data/submission0.csv", index=False)
    
    return result_df, fis, f_columns

if __name__ == "__main__":
    train = pd.read_csv("features_data/train_select.csv", sep=" ")
    test = pd.read_csv("features_data/test_select.csv", sep=" ")
    result_df, fis, f_columns = predict(train, test)
    result_df.to_csv("result_data/result_df.csv", index=False)
