# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:14:24 2019
@author: 11955
"""
import numpy as np
import pandas as pd

from func_lib import *
from egg import *

from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression

real = None
pred = None

#超参数设置
up_ergodic_feas = ["is_trade","item_id","item_category_list","item_brand_id","item_city_id","item_price_level","item_sales_level",
                    "item_collected_level","item_pv_level","shop_id","shop_review_num_level","shop_review_positive_rate","shop_star_level",
                    "shop_score_service","shop_score_delivery","shop_score_description"]

up_num_feas = ["item_price_level", "item_sales_level", "item_collected_level", "item_pv_level", "shop_review_num_level", 
            "shop_review_positive_rate", "shop_star_level", "shop_score_service", "shop_score_delivery", "shop_score_description"]

sp_ergodic_feas = ["is_trade","item_id","item_category_list","item_brand_id","item_city_id","item_price_level",
                   "item_sales_level","item_collected_level","item_pv_level","user_id","user_gender_id","user_age_level",
                   "user_occupation_id"]
sp_num_feas = ["item_price_level", "item_sales_level", "item_collected_level", "item_pv_level", "user_age_level"]

usp_ergodic_feas = ["item_id", "item_city_id", "item_price_level", "item_sales_level", "item_collected_level", "shop_review_num_level",
                    "shop_review_positive_rate"]
usp_num_feas = ["item_price_level", "item_sales_level", "item_collected_level", "shop_review_num_level", "shop_review_positive_rate"]

#best_paras  调参后确定的
path = "data/" #根文件夹的路径
do_train = False
do_val = True
do_predict = True

class Run:
    def __init__(self,train, test, model):
        self.train = train
        self.test = test
        self.model = model
        self.merge_ids = []
        
    def main(self):
        #数据预处理
        self.train, self.test = abv_pre(self.train, self.test)
        
        # 生成知识库(各种画像)
        up = UserPortrait(self.train)
        up.get_all_feas(up_ergodic_feas, up_num_feas)  
        self.merge_ids.append(up.merge_id)
        
        sp = ShopPortrait(self.train)
        sp.get_all_feas(sp_ergodic_feas, sp_num_feas)
        self.merge_ids.append(sp.merge_id)
        
        usp = ZuHePortrait(self.train, ["user_id", "shop_id"], "usp_")
        usp.get_all_feas(usp_ergodic_feas, usp_num_feas, viol=True)
        self.merge_ids.append(usp.merge_id)
        
        uip = ZuHePortrait(self.train, ["user_id", "item_id"], "uip_")
        uip.get_all_feas(viol=False)
        self.merge_ids.append(uip.merge_id)
        
        sip = ZuHePortrait(self.train, ["shop_id", "item_id"], "sip_")
        sip.get_all_feas()
        self.merge_ids.append(sip.merge_id)
        
        isp = ZuHePortrait(self.train, ["item_id", "shop_id"], "isp_")
        isp.get_all_feas()
        self.merge_ids.append(isp.merge_id)
        
        icup = ZuHePortrait(self.train, ["item_city_id", "user_id"], "icup_")
        icup.get_all_feas(time=False)
        self.merge_ids.append(icup.merge_id)
        
        #每个品牌每个用户的浏览量、购买量、转化率
        ibup = ZuHePortrait(self.train, ["item_brand_id", "user_id"], "ibup_")
        ibup.get_all_feas(time=False)
        self.merge_ids.append(ibup.merge_id)
        
        # 把知识库merge到训练集和测试集中对应id或id组合，得到最终的训练集和测试集
        if self.merge_ids:
            for merge_id in self.merge_ids:
                print("merge_id=", merge_id)
                if merge_id == ["user_id"]:
                    self.train = self.train.merge(up.user_portrait, on=merge_id, how="left")
                    self.test = self.test.merge(up.user_portrait, on=merge_id, how="left")
                if merge_id == ["shop_id"]:
                    self.train = self.train.merge(sp.shop_portrait, on=merge_id, how="left")
                    self.test = self.test.merge(sp.shop_portrait, on=merge_id, how="left")
                if merge_id == ["user_id", "shop_id"]:
                    self.train = self.train.merge(usp.zuhe_portrait, on=merge_id, how="left")
                    self.test = self.test.merge(usp.zuhe_portrait, on=merge_id, how="left")
                if merge_id == ["user_id", "item_id"]:
                    self.train = self.train.merge(uip.zuhe_portrait, on=merge_id, how="left")
                    self.test = self.test.merge(uip.zuhe_portrait, on=merge_id, how="left")
                if merge_id == ["shop_id", "item_id"]:
                    self.train = self.train.merge(sip.zuhe_portrait, on=merge_id, how="left")
                    self.test = self.test.merge(sip.zuhe_portrait, on=merge_id, how="left")
                if merge_id == ["item_id", "shop_id"]:
                    self.train = self.train.merge(isp.zuhe_portrait, on=merge_id, how="left")
                    self.test = self.test.merge(isp.zuhe_portrait, on=merge_id, how="left")
                if merge_id == ["item_city_id", "user_id"]:
                    self.train = self.train.merge(icup.zuhe_portrait, on=merge_id, how="left")
                    self.test = self.test.merge(icup.zuhe_portrait, on=merge_id, how="left")
                if merge_id == ["item_brand_id", "user_id"]:
                    self.train = self.train.merge(ibup.zuhe_portrait, on=merge_id, how="left")
                    self.test = self.test.merge(ibup.zuhe_portrait, on=merge_id, how="left")
                    
        # 数据再处理、分析此时生成的训练集，剔除无用特征，生成最终的训练集和测试集
        #  ......
        drop_feas = ["item_category_list", "item_property_list", "predict_category_property", "time",
                     "context_id", "minute"]
        self.train = self.train.drop(drop_feas, axis=1).fillna(-1)
        self.test = self.test.drop(drop_feas, axis=1).fillna(-1)
        
        #特征选择：递归消除法
        select_model = XGBClassifier()
        print("特征选择中......")
        sf = SelcetFeatures(select_model, self.train, self.test, "is_trade", step=1, cv=3)
        self.train, self.test = sf.newTrainAndTest(select_model, train_useful_colunms=["instance_id", "context_timestamp", "is_trade"], test_useful_colunms=["instance_id", "context_timestamp"])
        print("特征选择完毕，最终训练集、测试集生成！")
        
        print("训练集文件写入中间文件夹中......")  #写入中间文件中调参的时候会用到
        self.train.to_csv(path+"middle/train.csv", index=False)
        print("训练集写入完毕！    最终测试集写入中间文件夹中......")
        self.test.to_csv(path+"middle/test.csv", index=False)
        print("测试集写入完毕！")
        
        #*调参：在para_adjust.py模块中完成，得到最优的参数组best_paras提供给模型训练
        
        # 训练模型
        if do_train:
            print("self.train.shape=", self.train.shape)
            y_train = self.train["is_trade"]
            X_train = self.train.drop("is_trade", axis=1)
            print("【训练】：模型训练中......")
            self.model.fit(X_train, y_train)
            print("模型训练完毕！")
            
        # 做验证：
        if do_val and not do_train:
            tra, val = tra_val_split(self.train)  #时序问题，无法用交叉验证求模型效果
            y_tra = tra["is_trade"]
            X_tra = tra.drop("is_trade", axis=1)
            y_val = val["is_trade"]
            X_val = val.drop("is_trade", axis=1)
            print("X_tra.shape[0]=", X_tra.shape[0])
            print("【验证】：模型训练中......")
            self.model.fit(X_tra, y_tra)
            print("模型训练完毕！")
            val_pred = self.model.predict_proba(X_val)[:, 1]
            count = 0
            for r in val_pred:
                if r >= 0.5:
                    count += 1
            print("count=", count)
            
            logloss = get_logloss(y_val, val_pred)
            print("验证效果：logloss=", logloss)
        
        # 做预测：把最终预测结果输入到设定的文件夹位置
        if do_predict and do_train:
            print("do_predict and do_train")
            predict = self.test[["instance_id"]]
            result = self.model.predict_proba(self.test)[:, 1]
            count = 0
            for r in result:
                if r >= 0.5:
                    count += 1
            print("count=", count)
    
            predict["proba1"] = result
            print("预测结果写入%s/output文件中......"%path)
            predict.to_csv(path+"output/predict.csv", index=False)
            print("写入完毕！")

if __name__ == "__main__":
    train = pd.read_csv(path+"input/train.txt"," ") ###*** " "
    test = pd.read_csv(path+"input/test_b.txt"," ")
    train_sample = train.sample(frac=0.0005)
    model = XGBClassifier(learning_rate=0.04,n_estimators=550,max_depth=4,min_child_weight=2,subsample=0.65,
                          colsample_bytree=0.5, gamma=0.2, reg_alpha=0.05, reg_lambda=0.45) #用到的模型(包括参数) 
    r = Run(train_sample, test, model)
    r.main()
    t_train = r.train
    t_test = r.test
        