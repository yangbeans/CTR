# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:17:06 2019

@author: 11955
"""
#模块功能：非特征工程的其他函数集中地
import numpy as np
import pandas as pd
import time

import random
import scipy.special as special

#train = pd.read_csv("data/round1_ijcai_18_train_20180301.txt"," ") ###*** " "
#test = pd.read_csv("data/round1_ijcai_18_test_a_20180301.txt"," ")

def abv_pre(data):
    #剔除异常值
    data = data[data["shop_review_positive_rate"]>=0.85]
    data = data[data["shop_score_service"]>=0.85]
    data = data[data["shop_score_delivery"]>=0.85]
    data = data[data["shop_score_description"]>=0.85]
    return data

#函数功能：将时间戳变成日期格式表达
def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

def weekday_day_hour(data):
    data["time"] = data["context_timestamp"].apply(timestamp_datetime)
    data["day"] = data["time"].apply(lambda x: int(x[8:10]))
    #data["weekday"] = pd.to_datetime(data["time"]).weekday()
    data["month"] = data["time"].apply(lambda x: int(x[5:7]))
    data["hour"] = data["time"].apply(lambda x: int(x[11:13]))
    data["minute"] = data["time"].apply(lambda x: int(x[14:16]))
    return data

#函数功能：原始数据预处理
def abv_pre(train, test):
    train.drop_duplicates(subset=["instance_id"],keep='first',inplace=True)
    #剔除异常值
    train = train[train["shop_review_positive_rate"]>=0.85]
    train = train[train["shop_score_service"]>=0.85]
    train = train[train["shop_score_delivery"]>=0.85]
    train = train[train["shop_score_description"]>=0.85]
    #时间戳处理 
    train = weekday_day_hour(train)
    test = weekday_day_hour(test)
    return train, test

#函数功能：分割训练集和验证集
def tra_val_split(train, val_size=0.2): 
    time_diff = train["context_timestamp"].max() - train["context_timestamp"].min()
    split_time = train["context_timestamp"].max() - time_diff * val_size
    tra = train[train["context_timestamp"]<=split_time]
    val = train[train["context_timestamp"]>split_time]
    return tra, val

#函数功能：得到损失函数logloss
def get_logloss(y_real, y_pred):
    ones = []
    length = len(y_real)
    for i in range(length):
        ones.append(1)
    return -(sum(y_real*np.log(y_pred) + (ones-y_real)*np.log(ones-y_pred))/length)

#函数功能：求出一个特征中出现次数最多的那一个取值
def get_max_val(row):
    return row.value_counts().index[0]

#函数功能：计算表格中所有gby_id的平均时间差
def time_diff(data, gby_id, time_express, diff_type="sec"):
    print(0)
    data = data.sort_values([gby_id, time_express], ascending=[1, 1])
    print(1)
    rnColumn = data[[gby_id, time_express]].groupby(gby_id).rank(method='min')
    print(2)
    data["rn"] = rnColumn[time_express]
    print(3)
    data['rn_1'] = rnColumn[time_express] - 1
    print(4)
    data2 = data.merge(data, how='left', left_on=[gby_id, 'rn_1'], right_on=[gby_id, 'rn'])
    print(5)
    data3 = data2[[gby_id, time_express+"_x", time_express+"_y"]]
    data3 = data3.dropna()
    data3["time_diff"] = data3[time_express+"_x"] - data3[time_express+"_y"]
    if diff_type == "sec":
        pass
    if diff_type == "hour":
        data3["time_diff"] = data3["time_diff"]/3600
    if diff_type == "day":
        data3["time_diff"] = data3["time_diff"]/(3600*24)
    data3 = data3.groupby(gby_id)["time_diff"].mean().reset_index().rename(columns={"time_diff":"_time_mean_diff"})
    return data3

#函数功能：暴力统计
def viol_statis(data, portrait, gby_id_list, ergodic_feas, num_feas, port_type): #选合理的多个特征并遍历
    print(port_type+":viol_statis")
    for fea in ergodic_feas:
        print(fea)
        if fea in num_feas:
            sv = data.groupby(gby_id_list)[fea].agg(["count", "nunique", "max", "min", "sum", "mean", "std"])  #
            sv.columns = [port_type+fea+"_"+func for func in ["count", "nunique", "max", "min", "sum", "mean", "std"]]  #
            sv = sv.reset_index()
            portrait = portrait.merge(sv, on=gby_id_list, how="left")
        else:
            sv = data.groupby(gby_id_list)[fea].agg(["count", "nunique"])
            sv.columns = [port_type+fea+"_"+ func for func in ["count", "nunique"]]
            sv = sv.reset_index()
            portrait = portrait.merge(sv, on=gby_id_list, how="left")
    #对缺失值处理-->用-1填充
    portrait = portrait.fillna(-1)
    return portrait

#类功能：贝叶斯平滑，用于平滑特征中的转化率
class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for clk_rt in sample:
            imp = random.random() * imp_upperbound
            imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0

        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i]+alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i]-clks[i]+beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(numerator_alpha/denominator), beta*(numerator_beta/denominator)
    
#函数功能：贝叶斯平滑
def bayesoom_cr_rate(data, portrait, gby_id, iter_num, epsilon, port_type):
    print(port_type+"贝叶斯平滑处理中......")
    cr = data.groupby(gby_id)["is_trade"].agg({port_type+"_click_num":"count", port_type+"_buy_num":"sum"}).reset_index()  ###***
    bs = BayesianSmoothing(1, 1)
    bs.update(cr[port_type+"_click_num"], cr[port_type+"_buy_num"], 100, 0.01)
    cr[port_type+"cr_smooth"] = (cr[port_type+"_buy_num"]+bs.alpha)/(cr[port_type+"_click_num"]+bs.alpha+bs.beta)
    cr = cr[gby_id+[port_type+"cr_smooth"]]
    portrait = portrait.merge(cr, on=gby_id, how="left")
    return portrait
