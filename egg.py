# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:12:50 2019

@author: 11955
"""
import numpy as np
import pandas as pd

from func_lib import *
from sklearn.feature_selection import RFECV

#该类实现的功能：通过提取以user_id为基准的多个特征生成一个特征群，用一个List的全局变量装，这样在Main函数调这个类的生成特征的函数后，所有用户画像的特征群就可以被利用了
class UserPortrait:
    def __init__(self, train):
        self.train = train
        self.merge_id = []
        self.user_portrait = pd.DataFrame()
        self.user_portrait["user_id"] = np.unique(self.train["user_id"])
    
    # 暴力统计
    def viol_statis2(self, ergodic_feas, num_feas): #选合理的多个特征并遍历
        print("up:viol_statis")
        print(len(np.unique(self.train["user_id"])))
        
        for fea in ergodic_feas:
            print(fea)
            if fea in num_feas:
                uv = self.train.groupby("user_id")[fea].agg(["count", "nunique", "max", "min", "sum", "mean", "std"])  #
                uv.columns = ["uv_"+fea+"_"+ func for func in ["count", "nunique", "max", "min", "sum", "mean", "std"]]  #
                uv = uv.reset_index()
                self.user_portrait = self.user_portrait.merge(uv, on="user_id", how="left")
            else:
                uv = self.train.groupby("user_id")[fea].agg(["count", "nunique"])
                uv.columns = ["uv_"+fea+"_"+ func for func in ["count", "nunique"]]
                uv = uv.reset_index()
                self.user_portrait = self.user_portrait.merge(uv, on="user_id", how="left")
        #对缺失值处理-->用-1填充
        self.user_portrait = self.user_portrait.fillna(-1)
        #return self.user_portrait
    
    # 常规统计
    def routine_statis(self):
        print("up:routine_statis")
        #用户对商品的忠诚度
        ur1 = self.train[["user_id", "item_id", "item_city_id"]]
        ur1 = ur1.groupby(["user_id", "item_id"])["item_city_id"].count().reset_index().rename(columns={"item_city_id":"item_buy_count"})
        ur1 = ur1.groupby("user_id")["item_buy_count"].mean().reset_index().rename(columns={"item_buy_count":"ur_item_buy_mean"})
        self.user_portrait = self.user_portrait.merge(ur1, on="user_id", how="left")
        #用户对店铺的忠诚度
        ur2 = self.train[["user_id", "shop_id", "item_city_id"]]
        ur2 = ur2.groupby(["user_id", "shop_id"])["item_city_id"].count().reset_index().rename(columns={"item_city_id":"shop_buy_count"})
        ur2 = ur2.groupby("user_id")["shop_buy_count"].mean().reset_index().rename(columns={"shop_buy_count":"ur_shop_buy_mean"})
        
        #用户购买次数最多的店铺
        #print("!!!!!")
        #self.user_portrait = self.train.groupby("user_id")["shop_id"].apply(get_max_val)
        #购买的店铺平均星级
        ur3 = self.train[["user_id", "shop_id", "shop_star_level"]]
        ur3 = ur3.groupby(["user_id", "shop_id"])["shop_star_level"].mean().reset_index().rename(columns={"item_city_id":"shop_star_level"})
        ur3 = ur3.groupby("user_id")["shop_star_level"].mean().reset_index().rename(columns={"shop_star_level":"ur_shop_star_mean"})
        self.user_portrait = self.user_portrait.merge(ur3, on="user_id", how="left")
        # 用户转化率
        #return rout_feas
        
    def bayesoom_cr_rate(self):
        print("贝叶斯平滑处理中......")
        cr = self.train.groupby("user_id")["is_trade"].agg({"up_click_num":"count", "up_buy_num":"sum"}).reset_index()  ###***
        bs = BayesianSmoothing(1, 1)
        bs.update(cr["up_click_num"], cr["up_buy_num"], 100, 0.01)
        cr["cr_smooth"] = (cr["sp_buy_num"]+bs.alpha)/(cr["sp_click_num"]+bs.alpha+bs.beta)
        cr = cr[["user_id", "cr_smooth"]]
        self.user_portrait = self.user_portrait.merge(cr, on="user_id", how="left")
    
    # 时间特征
    def time_egg(self):
        print("up:time_egg")
        ut1 = self.train.groupby(["user_id", "day"])["instance_id"].count().reset_index().rename(columns={"instance_id":"day_click_count"})
        ut1 = ut1.groupby("user_id")["day_click_count"].mean().reset_index().rename(columns={"day_click_count":"ut_day_click_mean"})
        self.user_portrait = self.user_portrait.merge(ut1, on="user_id", how="left")
        
        ut2 = self.train.groupby(["user_id", "hour"])["instance_id"].count().reset_index().rename(columns={"instance_id":"hour_click_count"})
        ut2 = ut2.groupby("user_id")["hour_click_count"].mean().reset_index().rename(columns={"hour_click_count":"ut_hour_click_mean"})
        self.user_portrait = self.user_portrait.merge(ut2, on="user_id", how="left")
        
        ut3 = self.train.groupby(["user_id", "minute"])["instance_id"].count().reset_index().rename(columns={"instance_id":"minute_click_count"})
        ut3 = ut3.groupby("user_id")["minute_click_count"].mean().reset_index().rename(columns={"hour_click_count":"ut_minute_click_mean"})
        self.user_portrait = self.user_portrait.merge(ut3, on="user_id", how="left")
        
        #时间间隔特征
        #用户浏览商品的时间间隔
        ut4 = time_diff(self.train, "user_id", "context_timestamp", diff_type="sec")
        self.user_portrait = self.user_portrait.merge(ut4, on="user_id", how="left")
        #用户购买商品的时间间隔
        ut5 = time_diff(self.train[self.train["is_trade"]==1], "user_id", "context_timestamp", diff_type="sec")
        self.user_portrait = self.user_portrait.merge(ut5, on="user_id", how="left")
        #用户购买同一个商品的平均时间间隔
    
    def get_all_feas(self, ergodic_feas, num_feas):
        #self.viol_statis(ergodic_feas, num_feas)
        self.user_portrait = viol_statis(self.train, self.user_portrait, ["user_id"], ergodic_feas, num_feas, "up_")
        #用户转化率的贝叶斯平滑
        self.user_portrait = bayesoom_cr_rate(self.train, self.user_portrait, ["user_id"], 100, 0.01, "up_")
        self.routine_statis()
        self.time_egg()
        self.merge_id.append("user_id")
        pass

class ShopPortrait:
    def __init__(self, train):
        self.train = train
        self.merge_id = []
        self.shop_portrait = pd.DataFrame()
        self.shop_portrait["shop_id"] = np.unique(self.train["shop_id"])
        
    # 暴力统计
    def viol_statis2(self, ergodic_feas, num_feas): #选合理的多个特征并遍历
        print("sp:viol_statis")
        print(len(np.unique(self.train["shop_id"])))
        for fea in ergodic_feas:
            print(fea)
            if fea in num_feas:
                sv = self.train.groupby("shop_id")[fea].agg(["count", "nunique", "max", "min", "sum", "mean", "std"])  #
                sv.columns = ["sv_"+fea+"_"+ func for func in ["count", "nunique", "max", "min", "sum", "mean", "std"]]  #
                sv = sv.reset_index()
                self.shop_portrait = self.shop_portrait.merge(sv, on="shop_id", how="left")
            else:
                sv = self.train.groupby("shop_id")[fea].agg(["count", "nunique"])
                sv.columns = ["sv_"+fea+"_"+ func for func in ["count", "nunique"]]
                sv = sv.reset_index()
                self.shop_portrait = self.shop_portrait.merge(sv, on="shop_id", how="left")
        #对缺失值处理-->用-1填充
        self.shop_portrait = self.shop_portrait.fillna(-1)
    
    #对店铺转化率做贝叶斯平滑
    def bayesoom_cr_rate(self):
        print("贝叶斯平滑处理中......")
        cr = self.train.groupby("shop_id")["is_trade"].agg({"sp_click_num":"count", "sp_buy_num":"sum"}).reset_index()  ###***
        bs = BayesianSmoothing(1, 1)
        bs.update(cr["sp_click_num"], cr["sp_buy_num"], 100, 0.01)
        cr["cr_smooth"] = (cr["sp_buy_num"]+bs.alpha)/(cr["sp_click_num"]+bs.alpha+bs.beta)
        cr = cr[["shop_id", "cr_smooth"]]
        self.shop_portrait = self.shop_portrait.merge(cr, on="shop_id", how="left")
        
    # 时间特征
    def time_egg(self):
        print("sp:time_egg")
        ut1 = self.train.groupby(["shop_id", "day"])["instance_id"].count().reset_index().rename(columns={"instance_id":"day_click_count"})
        ut1 = ut1.groupby("shop_id")["day_click_count"].mean().reset_index().rename(columns={"day_click_count":"st_day_click_mean"})
        self.shop_portrait = self.shop_portrait.merge(ut1, on="shop_id", how="left")
        
        ut2 = self.train.groupby(["shop_id", "hour"])["instance_id"].count().reset_index().rename(columns={"instance_id":"hour_click_count"})
        ut2 = ut2.groupby("shop_id")["hour_click_count"].mean().reset_index().rename(columns={"hour_click_count":"st_hour_click_mean"})
        self.shop_portrait = self.shop_portrait.merge(ut2, on="shop_id", how="left")
        
        ut3 = self.train.groupby(["shop_id", "minute"])["instance_id"].count().reset_index().rename(columns={"instance_id":"minute_click_count"})
        ut3 = ut3.groupby("shop_id")["minute_click_count"].mean().reset_index().rename(columns={"hour_click_count":"st_minute_click_mean"})
        self.shop_portrait = self.shop_portrait.merge(ut3, on="shop_id", how="left")
        
        #时间间隔特征
        #用户浏览商品的时间间隔
        ut4 = time_diff(self.train, "shop_id", "context_timestamp", diff_type="sec")
        self.shop_portrait = self.shop_portrait.merge(ut4, on="shop_id", how="left")
        #用户购买商品的时间间隔
        ut5 = time_diff(self.train[self.train["is_trade"]==1], "shop_id", "context_timestamp", diff_type="sec")
        self.shop_portrait = self.shop_portrait.merge(ut5, on="shop_id", how="left")
        #用户购买同一个商品的平均时间间隔
    
    def get_all_feas(self, ergodic_feas, num_feas):
        #self.viol_statis(ergodic_feas, num_feas)
        self.shop_portrait = viol_statis(self.train, self.shop_portrait, "shop_id", ergodic_feas, num_feas, "s_")
        #self.routine_statis()
        #贝叶斯平滑店铺转化率
        self.shop_portrait = bayesoom_cr_rate(self.train, self.shop_portrait, ["shop_id"], 100, 0.01, "sp_")
        self.time_egg()
        self.merge_id.append("shop_id")
        pass

#创建一个模板，供其他组合特征调用，并在get_all_feas函数里设置相应的特征开关，便于灵活调用
class ZuHePortrait:
    def __init__(self, train, merge_id, zuhe_type):
        self.train = train
        self.merge_id = merge_id
        self.zuhe_type = zuhe_type
        self.zuhe_portrait = self.train.groupby(self.merge_id)["instance_id"].count().reset_index().rename(columns={"instance_id":self.zuhe_type+"_click_num"})
    
    def routine_statis(self):
        zuhe_rs1 = self.train[self.train["is_trade"]==1].groupby(self.merge_id)["instance_id"].count().reset_index().rename(columns={"instance_id":self.zuhe_type+"_buy_num"})
        #转化率
        #zuhe_rs2 = self.train.groupby(self.merge_id)["is_trade"].mean().reset_index().rename(columns={"is_trade":self.zuhe_type+"_cr_rate"})
        #self.zuhe_portrait = self.zuhe_portrait.merge(zuhe_rs1, on=self.merge_id, how="left")
        #self.zuhe_portrait = self.zuhe_portrait.merge(zuhe_rs2, on=self.merge_id, how="left")
        
    def time_type_statis(self, time_type):
        zuhet = self.train.groupby([self.merge_id[0],self.merge_id[1], time_type])["instance_id"].count().reset_index().rename(columns={"instance_id":"type_click_count"})
        zuhet = zuhet.groupby(self.merge_id)["type_click_count"].mean().reset_index().rename(columns={"type_click_count":self.zuhe_type+"_"+time_type+"_click_mean"})
        return zuhet
        
    def time_egg(self):
        #ust1 = self.train.groupby([self.merge_id[0],self.merge_id[1], "day"])["instance_id"].count().reset_index().rename(columns={"instance_id":"day_click_count"})
        #ust1 = ust1.groupby(["user_id", "shop_id"])["day_click_count"].mean().reset_index().rename(columns={"day_click_count":zuhe_type+"_"+time_type+"_click_mean"})
        time_types = ["day", "hour", "month"]
        for time_type in time_types:
            zuhet = self.time_type_statis(time_type)
            self.zuhe_portrait = self.zuhe_portrait.merge(zuhet, on=self.merge_id, how="left")
        
    def get_all_feas(self, ergodic_feas=None, num_feas=None, viol=False, routine=True, time=True, bayes=True):
        #self.viol_statis(ergodic_feas, num_feas)
        #self.zuhe_portrait = viol_statis(self.train, self.zuhe_portrait, self.merge_id, ergodic_feas, num_feas, self.zuhe_type)
        if viol:
            self.zuhe_portrait = viol_statis(self.train, self.zuhe_portrait, self.merge_id, ergodic_feas, num_feas, self.zuhe_type)
        if routine:
            self.routine_statis()
        if time:
            self.time_egg()
        if bayes:
            self.zuhe_portrait = bayesoom_cr_rate(self.train, self.zuhe_portrait, self.merge_id, 100, 0.01, self.zuhe_type)
        pass    
    
#特征选择函数，返回筛选特征后的训练集、测试集和有用特征的columns
class SelcetFeatures():
    def __init__(self, select_model, train, test, label, step=1, cv=10):   ###***设置默认值，设置的参数调用时不是必要填的参数，step的默认值为1， cv的默认值为10
        self.select_model = select_model
        self.train = train
        self.test = test
        self.step = step
        self.cv = cv
        self.label = label
        self.select_columns = None
    
    #返回有用特征的columns
    def selectFeatures(self, select_model):
        selector = RFECV(estimator=select_model, step=self.step, cv=self.cv)
        y = self.train[self.label]
        X = self.train.drop(self.label, axis=1)
        select_X = selector.fit_transform(X, y)
        select_features_index = selector.get_support(True)  
        select_columns = X.columns[select_features_index]
        
        return select_X, select_columns
    
    #返回特征选择后的训练集和测试集
    def newTrainAndTest(self, select_model, train_useful_colunms=[], test_useful_colunms=[]):
        print(1)
        select_X, self.select_columns = self.selectFeatures(select_model)
        print(2)
        train_select_columns = list(self.select_columns) + list(train_useful_colunms)   ###***运算时注意数据格式
        test_select_columns = list(self.select_columns) + list(test_useful_colunms) 
        n_train = self.train[train_select_columns]
        n_test = self.test[test_select_columns]
        return n_train, n_test

    