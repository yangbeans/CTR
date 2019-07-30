# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 23:59:11 2019

@author: 11955
"""

#调参
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV

train = pd.read_csv("data/middle/train.csv")
y_train = train["is_trade"]
X_train = train.drop("is_trade", axis=1)

#初始化
learning_rate: 0.1
n_estimators: 500
max_depth: 5
min_child_weight: 1
subsample: 0.8
colsample_bytree:0.8
gamma: 0
reg_alpha: 0
reg_lambda: 1

"""
#1 调n_estimators
cv_params = {'n_estimators': [550, 575, 600, 650, 675]}  #550
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, y_train)

print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

#2 调max_depth和min_child_weight
cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}  #4, 2
other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, y_train)

print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

#3 调gamma
cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}   #0.2
other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 2, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, y_train)

print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))


#4 调subsample和colsample_bytree
cv_params = {'subsample': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], 'colsample_bytree': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]}   #0.65,0.5
other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 2, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.2, 'reg_alpha': 0, 'reg_lambda': 1}

model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, y_train)

print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

#5 调reg_alpha和reg_lambda
cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}  #0.05， 0.45
other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 2, 'seed': 0,
                'subsample': 0.65, 'colsample_bytree': 0.5, 'gamma': 0.2, 'reg_alpha': 0, 'reg_lambda': 1}

model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, y_train)

print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
"""
#6 调学习率learning_rate
cv_params = cv_params = {'learning_rate': [0.01, 0.02, 0.04, 0.05, 0.07, 0.1, 0.2]}  #0.04
other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 2, 'seed': 0,
                'subsample': 0.65, 'colsample_bytree': 0.5, 'gamma': 0.2, 'reg_alpha': 0.05, 'reg_lambda': 0.45}

model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, y_train)

print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

#最佳参数组合：
learning_rate: 0.04
n_estimators: 550
max_depth: 4
min_child_weight: 2
subsample: 0.65
colsample_bytree:0.5
gamma: 0.2
reg_alpha: 0.05
reg_lambda: 0.45


