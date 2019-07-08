import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import roc_curve, auc




def readData():
    train = pd.read_csv("data/round1_ijcai_18_train_20180301.txt",sep=" ")
    train = train.drop_duplicates(inplace=True)
    
    #trainLen = len(train)
    #trainLabel = train["is_trade"]
    
    test = pd.read_csv("data/round1_ijcai_18_test_a_20180301.txt"," ")
    test = test.drop_duplicates(inplace=True)
    
    #testInstanceID = test["instance_id"]
    
    #all_df = pd.concat([train,test])
    
    return train

###***获取时间
def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt
    
def weekdayDayHour(data):
    data["time"] = data["context_timestamp"].apply(timestamp_datetime)
    data["day"] = data["time"].apply(lambda x: int(x[8:10]))
    #data["weekday"] = pd.to_datetime(data["time"]).weekday()
    data["hour"] = data["time"].apply(lambda x: int(x[11:13]))
    data["minute"] = data["time"].apply(lambda x: int(x[14:16]))
    
    return data

def itemCategory(data):
    def getItemCategory1(row):
        rows = row.split(";")
        return rows[0]
    def getItemCategory2(row):
        rows = row.split(";")
        return rows[1]
    data["item_category_list1"] = data["item_category_list"].apply(getItemCategory1)
    data["item_category_list2"] = data["item_category_list"].apply(getItemCategory2)
    return data

#特征提取，统计特征
def extra_feature(data):
    #销售等级/商品展示的等级
    data["sales_div_pv"] = data["item_sales_level"]/(1+data["item_pv_level"])
    data["sales_div_pv"] = data["sales_div_pv"].map(lambda x: int(x*10), na_action="ignore") ###***
    
    #计算每天的点击量
    number_click_day = data.groupby("day").size().reset_index().rename(columns={0:'number_click_day'})
    data = pd.merge(data,number_click_day,on="day",how="left")
    
    #每个小时段的点击量
    number_click_hour = data.groupby("hour").size().reset_index().rename(columns={"item_id":"number_click_hour"})
    data = pd.merge(data,number_click_hour,on="hour",how="left")
    
    #数量
    #一个商品类目下的商品有多少个
    number_category_item = data.groupby(["item_category_list2","item_id"]).size().reset_index().rename(columns={0:"number_category_item"})###***size()
    data = pd.merge(data,number_category_item,on=["item_category_list2","item_id"],how="left")
    
    #一个商品2类别下有多少个商品 
    number_category2  = data.groupby("item_category_list2").size().reset_index().rename(columns={0:"number_category2"})
    data = pd.merge(data,number_category2,on="item_category_list2",how="left")
    #一个商品类别下的商品item所有 的个数/一个商品2类别下的所有商品个数，即单个Item在所在大类中的数量占比
    ###***特征思想：某一特征，子类与大类的关系(数量等关系)
    data["pro_item_id_category2"] = data["number_category_item"]/data["number_category2"] 
    
    #价格
    #一个商品类别2下，不同item的平均 价格
    ave_price_category_item = data.groupby(["item_category_list2","item_id"]).mean()["item_price_level"].reset_index().rename(columns={"item_price_level":"age_price_category_item"})
    data = pd.merge(data,ave_price_category_item,on=["item_category_list2","item_id"],how="left")
    
    #不同商品类别2的平均价格
    ave_price_category = data.groupby("item_category_list2").mean()["item_price_level"].reset_index().rename(columns={"item_price_level":"ave_price_category"})###***
    data = pd.merge(data,ave_price_category,on="item_category_list2",how="left")
    #单个item的价格和所在大类平均价格之比
    data['prob_item_price_to_ave_category2'] = data['item_price_level']/data['ave_price_category']
    
    #item_price_level
    #不同价格水平下的item的平均销售情况
    #ave_sales_price_category_item  = data.groupby(["item_category_list2','item_price_level",'item_id']).mean()["item_sales_level"].reset_index().rename(columns={"item_sales_level":"ave_sales_price_category_item"})
    #data = pd.merge(data,ave_sales_price_category_item,on=["item_category_list2','item_price_level",'item_id'],how="left")
    
    #不同大类的商品销售情况
    ave_sales_level_category= data.groupby("item_category_list2").mean()["item_sales_level"].reset_index().rename(columns={"item_sales_level":"ave_sales_level_category"})
    data = pd.merge(data,ave_sales_level_category,on="item_category_list2",how="left")
    
    data['prob_ave_category_sales_item_sales'] = data['item_sales_level']/data['ave_sales_level_category']
    
    #价格最高的商品大类
    max_price_category = data.groupby("item_category_list2")["item_price_level"].max().reset_index().rename(columns={"item_price_level":"max_price_category"})
    data = pd.merge(data,max_price_category,on="item_category_list2",how="left")
    data["is_max_price_category"] = data["item_price_level"]/data["max_price_category"]
    data["is_max_price_category"] = data["is_max_price_category"].map(lambda x: int(x),na_action = "ignore")
    
    #价格最低的商品大类
    min_price_category = data.groupby("item_category_list2")["item_price_level"].max().reset_index().rename(columns={"item_price_level":"min_price_category"})
    data = pd.merge(data,min_price_category,on="item_category_list2",how="left")
    data["is_min_price_category"] = data["item_price_level"]/data["min_price_category"]
    data["is_min_price_category"] = data["is_min_price_category"].map(lambda x: int(x),na_action="ignore")
    
    return data


#组合特征
def zuheFeature(data):
    #每个商品每个用户的浏览量
    number_item_user_id = data.groupby(["item_id","user_id"]).size().reset_index().rename(columns={0:"number_item_user_id"})
    number_user_id = data.groupby("user_id").size().reset_index().rename(columns={0:"number_user_id"})
    data = pd.merge(data,number_item_user_id,on=["item_id","user_id"],how="left")
    data = pd.merge(data,number_user_id,on="user_id",how="left")
    data["item_id_user_prob"] = data["number_item_user_id"]/data["number_user_id"] ###***
    data["sale_price"] = data["item_sales_level"] + data["item_price_level"]
    data["gender_star"] = data["user_gender_id"] + data["user_star_level"]
    
    #每个品牌每个用户的浏览量
    item_brand_id_user_cnt = data.groupby(["item_brand_id","user_id"]).size().reset_index().rename(columns={0:"item_brand_id_user_cnt"})
    data = pd.merge(data,item_brand_id_user_cnt,on=["item_brand_id","user_id"],how="left")
    data["item_brand_id_user_prob"] = data["item_brand_id_user_cnt"]/data["number_user_id"]
    
    #每个商品每家店的浏览量
    item_id_shop_cnt = data.groupby(["item_id","shop_id"]).size().reset_index().rename(columns={0:"item_id_shop_cnt"})
    number_shop_id = data.groupby("shop_id").size().reset_index().rename(columns={0:"number_shop_id"})
    data = pd.merge(data,item_id_shop_cnt,on=["item_id","shop_id"],how="left")
    data = pd.merge(data,number_shop_id,on="shop_id",how="left")
    data["item_id_shop_prob"] = data["item_id_shop_cnt"]/data["number_shop_id"]
    
    #店铺不同好评率对应的销量情况
    shop_rev_cnt = data.groupby("shop_review_positive_rate").size().reset_index().rename(columns={0:"shop_rev_cnt"})
    data = pd.merge(data,shop_rev_cnt,on="shop_review_positive_rate",how="left")
    
    #不同店铺好评率与每一种商品价格的销量
    item_price_level_shop_rev_cnt  = data.groupby(["shop_review_positive_rate","item_price_level"]).size().reset_index().rename(columns={0:"item_price_level_shop_rev_cnt"})
    data = pd.merge(data,item_price_level_shop_rev_cnt,on=["shop_review_positive_rate","item_price_level"],how="left")
    data["item_price_level_shop_rev_prob"] = data["item_price_level_shop_rev_cnt"]/data["shop_rev_cnt"]
    
    #不同商品售卖价格每一个用户的购买量
    item_sales_level_user_cnt = data.groupby(["item_sales_level","user_id"]).size().reset_index().rename(columns={0:"item_sales_level_user_cnt"})
    data = pd.merge(data,item_sales_level_user_cnt,on=["item_sales_level","user_id"],how="left")
    data["item_sales_level_user_prob"] = data["item_sales_level_user_cnt"]/data["number_user_id"]
    
    data["sale_collect"] = data["item_sales_level"] + data["item_collected_level"]
    
    #不同展现下每个商店流量情况
    item_pv_level_shop_cnt = data.groupby(['item_pv_level', 'shop_id']).size().reset_index().rename(columns={0:"item_pv_level_shop_cnt"})
    data = pd.merge(data,item_pv_level_shop_cnt,on=['item_pv_level', 'shop_id'],how="left")
    data["item_pv_level_shop_prob"] = data["item_pv_level_shop_cnt"]/data["number_shop_id"]
    
    #不同城市每种商店展现的流量
    item_city_id_shop_rev_cnt = data.groupby(['item_city_id', 'shop_review_positive_rate']).size().reset_index().rename(columns={0:"item_city_id_shop_rev_cnt"})
    data = pd.merge(data,item_city_id_shop_rev_cnt,on=['item_city_id', 'shop_review_positive_rate'],how="left")
    data["item_city_id_shop_rev_prob"] = data["item_city_id_shop_rev_cnt"]/data["shop_rev_cnt"]
    
    return data

#店铺分类    这个函数 data的shape变了.. 
def shopFenduan(data):
    def deliver(row):
        ###***用for循环对数值做分类
        jiange = 0.1
        for i in range(0,10):
            if (row >= (4.0 + jiange * i)) & (row < (4.0 + jiange * (i + 1))):###***划分是根据探索相应字段的数据分布情况而定的
                return i
        if row == 5:
            return 10
        else:
            return 11
    
    def service(x):
        # x=round(x,6)
        jiange = 0.1
        for i in range(1, 20):
            if (x >= 3.93 + jiange * (i - 1)) & (x <= 3.93 + jiange * i):
                return i + 1
            if x == -1:
                return 1
    def describe(x):
        # x=round(x,6)
        jiange = 0.1
        for i in range(1, 30):
            if (x >= 3.93 + jiange * (i - 1)) & (x <= 3.93 + jiange * i):
                return i + 1
            if x == -1:
                return 1
    
    def review(x):
        # x=round(x,6)
        jiange = 0.02
        for i in range(1, 30):
            if (x >= 0.714 + jiange * (i - 1)) & (x <= 0.714 + jiange * i):
                return i + 1
            if x == -1:
                return 1

        
    #根据物流评分对店铺分段
    data["shop_score_delivery"] = data["shop_score_delivery"] * 5 ###***扩大目标值
    data = data[data["shop_score_delivery"]!=-5]
    data["shop_deliver"] = data["shop_score_delivery"].apply(deliver)
    
    #根据店铺服务态度对店铺分段
    data["shop_score_service"] = data["shop_score_service"] * 5
    data = data[data["shop_score_service"]!=5]
    data["shop_service"] = data["shop_score_service"].apply(service)
    
    #店铺描述相符对店铺分段
    data["shop_score_description"] = data["shop_score_description"] * 5
    data = data[data["shop_score_description"]!=-5]
    data["shop_describe"] = data["shop_score_description"].apply(describe)
    
    #根据店铺好评率对店铺分段
    data = data[data["shop_review_positive_rate"]!=-1]
    data["shop_review"] = data["shop_review_positive_rate"].apply(review)
    
    return data
  
#?
def userCntPre30min(data):
    
    return data

###***对于本条记录(操作)，用户距离上一次浏览的时间差
def calTimeReduce(data):
    """
    求距离上一次浏览的时间差的思路：
    <1>找出所需的字段，生成df 如：user_id,context_timestamp
    <2>按user_id对时间进行排序(如时间越早排名越靠前)
        train1 = train1.sort_values(['user_id', 'context_timestamp'], ascending=[1, 1])
        rnColumn = train1.groupby('user_id').rank(method='min')
        train1['rn'] = rnColumn['context_timestamp']
    <3>对时间排序分别减1，生成新的排序字段
        train1['rn_1'] = rnColumn['context_timestamp'] - 1
    <4>合并 on=["user_id","timestamp"],并让合并后的train["rn_1"] - train["rn"] 得到该条记录下该用户距离上一次浏览的时间差
        train2 = train1.merge(train1, how='left', left_on=['user_id', 'rn_1'], right_on=['user_id', 'rn'])
    <5>删除重复和无用的字段
    <6>将train2与data merge
    """
    train_origin = data
    train1 = train_origin[["user_id", "context_timestamp", "instance_id"]]
    
    train1 = train1.sort_values(["user_id", "context_timestamp"],ascending=[1,1])
    rnColumn = train1.groupby("user_id").rank(method="min")
    train1["rn"] = rnColumn["context_timestamp"]
    
    train1["rn_1"] = rnColumn["context_timestamp"] - 1
    
    train2 = train1.merge(train1,left_on=["user_id", "rn_1"],right_on=["user_id", "rn"],how="left")
    train2["time_redc"] = train2['context_timestamp_x'] - train2['context_timestamp_y']
    train2 = train2.fillna(-1).astype("int64")
    
    train2 = train2.rename(columns={"context_timestamp_x":"context_timestamp"})
    train2 = train2.rename(columns={"instance_id_x":"instance_id"})
    train2 = train2.rename(columns={"rn_x":"rn"})
    train2 = train2.rename(columns={"rn_1_x":"rn_1"})
    train2 = train2.drop(["context_timestamp_y","instance_id_y","rn_y","rn_1_y"],axis=1)
    
    data = pd.merge(data,train2,on=["instance_id", "user_id", "context_timestamp"],how="left") ###***on=["instance_id", "user_id","context_timestamp"]
    
    return data

###***对于该条记录，用户浏览同样商品距离上次浏览的时间差
def calTimeReducUserItem(data):
    train_origin = data
    train1 = train_origin[['context_timestamp', 'user_id', 'instance_id', 'item_id']]
    
    train1 = train1.sort_values(["user_id", "item_id", "context_timestamp"],ascending=[1,1,1]) ###***函数sort_value(by=[],ascending=[])的使用
    rnColumn = train1.groupby(["user_id", "item_id"]).rank(method="min") ###***rank()函数的用法：显示该数的排名
                                        ###***df.groupby(["a","b"]).fun() 会对除"a","b"外的其他字段做fun()操作，返回的是被操作的字段和操作结果
    train1["rnui"] = rnColumn["context_timestamp"] 
    
    train1["rnui_1"] = rnColumn["context_timestamp"] - 1 
    
    train2 = train1.merge(train1,left_on=["user_id", "item_id", "rnui_1"], right_on=["user_id", "item_id", "rnui"], how="left")
    train2["time_redc_user_item"] = train2["context_timestamp_x"] - train2["context_timestamp_y"]
    train2 = train2.fillna(-1).astype("int64")
    
    train2 = train2.rename(columns={"instance_id_x":"instance_id"})
    train2 = train2.rename(columns={"context_timestamp_x":"context_timestamp"})
    train2 = train2.rename(columns={"rnui_1_x":"rnui_1"})
    train2 = train2.rename(columns={"rnui_x":"rnui"})
    train2 = train2.drop(["instance_id_y","context_timestamp_y","rnui_1_y","rnui_y"], axis=1)
    
    data = pd.merge(data, train2, on=["instance_id", "user_id", "item_id", "context_timestamp"], how="left")
    
    return data

#对于该条记录，用户浏览同一家店铺距离上次浏览的时间差
def calTimeReducUserShop(data):
    train_origin = data
    train1 = train_origin[['context_timestamp', 'user_id', 'instance_id', 'shop_id']]
    
    train1 = train1.sort_values(["user_id", "shop_id", "context_timestamp"],ascending=[1,1,1]) ###***函数sort_value(by=[],ascending=[])的使用
    rnColumn = train1.groupby(["user_id", "shop_id"]).rank(method="min") ###***rank()函数的用法：显示该数的排名
                                        ###***df.groupby(["a","b"]).fun() 会对除"a","b"外的其他字段做fun()操作，返回的是被操作的字段和操作结果
    train1["rnus"] = rnColumn["context_timestamp"] 
    
    train1["rnus_1"] = rnColumn["context_timestamp"] - 1 
    
    train2 = train1.merge(train1,left_on=["user_id", "shop_id", "rnus_1"], right_on=["user_id", "shop_id", "rnus"], how="left")
    train2["time_redc_user_shop"] = train2["context_timestamp_x"] - train2["context_timestamp_y"]
    train2 = train2.fillna(-1).astype("int64")
    
    train2 = train2.rename(columns={"instance_id_x":"instance_id"})
    train2 = train2.rename(columns={"context_timestamp_x":"context_timestamp"})
    train2 = train2.rename(columns={"rnus_1_x":"rnus_1"})
    train2 = train2.rename(columns={"rnus_x":"rnus"})
    train2 = train2.drop(["instance_id_y","context_timestamp_y","rnus_1_y","rnus_y"], axis=1)
    
    data = pd.merge(data, train2, on=["instance_id", "user_id", "shop_id", "context_timestamp"], how="left")
    
    return data

#*用户当天浏览商品的时间顺序特征，历史浏览商品的时间顺序等重要特征  内存原因，跑不动啊  
def slideCnt2(data):
    df = pd.DataFrame()
    for d in range(18, 26): #取一周
        df1 = data[data["day"]==d]
        df1 = df1[["instance_id", "user_id", "shop_id", "item_id", "context_timestamp"]]
        
        #对于该条记录，用户当天浏览商品的时间顺序
        rnColumn_user = df1.groupby("user_id").rank(method="min") ###***注意剩余的字段要可rank
        
        df1["user_id_order"] = rnColumn_user["context_timestamp"] #因为两者的列数相同，所以可以
        
        #对于该条记录，用户浏览同一件商品的时间顺序
        rnColumn_user_item = df1.groupby(["user_id", "item_id"]).rank(method="min")
        df1["user_item_id_order"] = rnColumn_user_item["context_timestamp"]
        
        #对于该条记录，用户浏览同一家店的时间顺序
        rnColumn_user_shop_id = df1.groupby(["user_id", "shop_id"]).rank(method="min")
        df1["user_shop_id_order"] = rnColumn_user_shop_id["context_timestamp"]
        
        #将一周的顺序特征df1拼接起来得到df
        df = pd.concat([df, df1])
    #把data与df merge on=["instance_id", "user_id", "shop_id"] , 得到data中每条记录的三种浏览时间顺序特征
    data = pd.merge(data, df, on=["instance_id", "user_id", "shop_id"], how="left")
    #下面那些代码是什么意思？？？
    return data


#滑窗,对浏览量、（购买量）做滑窗。    时序问题经常用的手段
def slidingWindow(data):
    #将索引设置为时间
    data = data.set_index(pd.to_datetime(data["time"])).sort_index()
    print("debug1")
    fmean = lambda x: x.rolling(window="7D", min_periods=3, closed="left").mean()
    fstd = lambda x: x.rolling(window="7D", min_periods=3, closed="left").std()
    #每个用户的浏览量7天滑窗
    data["7Dmean_user_id"] = data.groupby("user_id")["number_user_id"].transform(fmean).fillna(0) ###***data.groupby("user_id")["number_user_id"].transform(fmean).fillna(0)
    data["7Dstd_user_id"] = data.groupby("user_id")["number_user_id"].transform(fstd).fillna(0)
    print("debug2")
    #每家店的浏览量7天滑窗
    data["7Dmean_shop_id"] = data.groupby("shop_id")["number_shop_id"].transform(fmean).fillna(0)
    data["7Dstd_shop_id"] = data.groupby("shop_id")["number_shop_id"].transform(fstd).fillna(0)
    print("debug3")
    #每个商品的浏览量7天滑窗
    number_item_id = data.groupby("item_id").size().reset_index().rename(columns={0:"number_item_id"})
    data = pd.merge(data, number_item_id, on="item_id", how="left")
    data = data.set_index(pd.to_datetime(data["time"])).sort_index()
    data["7Dmean_item_id"] = data.groupby("item_id")["number_item_id"].transform(fmean).fillna(0)
    data["7Dstd_item_id"] = data.groupby("item_id")["number_item_id"].transform(fstd).fillna(0)
    print("debug4")
    #item, user_id 7天滑窗
    data["7Dmean_item_user_id"] = data.groupby(["item_id", "user_id"])["number_item_user_id"].transform(fmean).fillna(0)
    data["7Dstd_item_user_id"] = data.groupby(["item_id", "user_id"])["number_item_user_id"].transform(fstd).fillna(0)
    print("debug5")
    #item_brand_id, user_id 7天滑窗
    data["7Dmean_item_brand_user_id"] = data.groupby(["item_id", "item_brand_id"])["item_brand_id_user_cnt"].transform(fmean).fillna(0)
    data["7Dstd_item_brand_user_id"] = data.groupby(["item_id", "item_brand_id"])["item_brand_id_user_cnt"].transform(fstd).fillna(0)
    print("debug6")
    #item_id, shop_id
    data["7Dmean_item_shop_id"] = data.groupby(["item_id", "shop_id"])["item_id_shop_cnt"].transform(fmean).fillna(0)
    data["7Dstd_item_shop_id"] = data.groupby(["item_id", "shop_id"])["item_id_shop_cnt"].transform(fstd).fillna(0)
    
    #*统计了前一天，前三天以及历史的用户浏览/购买记录作为特征
    return data


#贝叶斯平滑



#*组合特征统计， 数据转换离散化编码，
def convertData(data):
    ###***数据离散化编码
    lbl = LabelEncoder()
    for col in ["item_id", "item_brand_id", "item_city_id", "shop_id", "user_id"]:
        data[col] = lbl.fit_transform(data[col])
    
    user_query_day = data.groupby(["user_id", "day"]).size().reset_index().rename(columns={0:"user_query_day"})
    data = pd.merge(data, user_query_day, on=["user_id", "day"], how="left")
    user_query_day_hour  = data.groupby(["user_id", "day", "hour"]).size().reset_index().rename(columns={0:"user_query_day_hour"})
    data = pd.merge(data, user_query_day_hour, on=["user_id", "day", "hour"], how="left")
    
    day_user_item_id = data.groupby(["day", "user_id", "item_id"]).size().reset_index().rename(
            columns={0:"day_user_item_id"})
    data = pd.merge(data, day_user_item_id, on=["day", "user_id", "item_id"], how="left")
    
    day_hour_minute_user_item_id = data.groupby(
            ['day', 'hour', 'minute', 'user_id', 'item_id']).size().reset_index().rename(columns={0:"day_hour_minute_user_item_id"})
    data = pd.merge(data, day_hour_minute_user_item_id, on=['day', 'hour', 'minute', 'user_id', 'item_id'], how="left")
    
    number_day_hour_item_id  = data.groupby(['day', 'hour', 'item_id']).size().reset_index().rename(
            columns={0:"number_day_hour_item_id"})
    data = pd.merge(data, number_day_hour_item_id, on=['day', 'hour', 'item_id'], how="left")
    
    item_user_id = data.groupby(['item_id', 'user_id']).size().reset_index().rename(columns={0: 'item_user_id'})
    data = pd.merge(data, item_user_id, 'left', on=['item_id', 'user_id'])
    
    #...
    
    return data


#数据整合，无用字段删除等。 没什么东西。。。
def dataFilter(data):
    drop_cloumns = ["item_category_list",
                    "item_property_list",
                    "context_timestamp",
                    "predict_category_property",
                    "time",
                    "item_id",
                    "item_brand_id",
                    "item_city_id",
                    "user_id",
                    "user_occupation_id",
                    "context_id",
                    "context_page_id",
                    "shop_id",
                    ]
    encoder_columns = ["item_category_list1",
                       "item_category_list2"
                       ]
    #编码
    lbl = LabelEncoder()
    for col in encoder_columns:
        data[col] = lbl.fit_transform(data[col])
    
    #删除无用字段
    data = data.drop(drop_cloumns, axis=1)
    #prepare_data = data[["instance_id"]]
    return data 

  
def fAuc(y_true, y_predict):
    fpr, tpr, thresholds = roc_curve(y_true, y_predict, pos_label=1)
    return auc(fpr,tpr)

def logLoss(y_true, y_predpro):
    y_true = np.array(y_true)
    y_predpro = np.array(y_predpro[:,1])
    s1 = -sum(y_true * np.log(y_predpro) + (1 - y_true) * np.log(1 - y_predpro))
    return s1/len(y_true)

def getAuc(train):
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
    X_tra, X_val, y_tra, y_val = train_test_split(X_train, y_train, test_size=0.2)
    
    X_tra = X_tra.drop("instance_id", axis=1)
    prepare_X_val = X_val[["instance_id"]]
    X_val = X_val.drop("instance_id", axis=1)
    
    #训练模型
    print("训练模型中.....")
    model_xgb = XGBClassifier()
    model_xgb.fit(X_tra, y_tra)
    print("训练完毕！")
    
    #预测
    y_pred = model_xgb.predict(X_val)
    y_predpro = model_xgb.predict_proba(X_val)
    #auc = fAuc(y_val, y_pred)
    #logLo = logLoss(y_val, y_predpro)
    
    return y_val, y_pred, y_predpro
                                                                    
    
    
if __name__ == "__main__":
    train = pd.read_csv("data/round1_ijcai_18_train_20180301.txt", sep=" ")
    #test = pd.read_csv("data/round1_ijcai_18_test_a_20180301.txt", sep=" ")
    test = pd.read_csv("data/round1_ijcai_18_test_b_20180418.txt", sep=" ")
    
    y_val, y_pred, y_predpro = getAuc(train)
    #print("auc = ", auc)
    #print("logLoss = ", logLo)
    auc = fAuc(y_val, y_pred)
    logLo = logLoss(y_val, y_predpro)
