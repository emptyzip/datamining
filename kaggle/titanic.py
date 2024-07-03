# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
#from sre_constants import NOT_LITERAL_IGNORE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

train_df = pd.read_csv(sys.argv[1])
test_df = pd.read_csv(sys.argv[2])

#train_df = pd.read_csv("/content/drive/MyDrive/train.csv")
#test_df = pd.read_csv("/content/drive/MyDrive/test.csv")

train_df_pre = train_df.drop(['PassengerId', 'Name'], axis=1)
test_df_pre = test_df.drop(['PassengerId', 'Name'], axis=1)

#숫자형 데이터 평균값 넣기
def encoding_feature(features):
  for feature in features:
    mode = train_df_pre[feature].mode()[0]
    null_feature = train_df_pre[feature].isnull()
    train_df_pre.loc[null_feature, feature] = mode

features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']
encoding_feature(features)

#나이 카테고리화 시키기
def ageTransform(age) :
    category = ''
    if age <= -1 : category = 'Unknown'
    elif age <= 5 : category = 'Baby'
    elif age <= 12 : category = 'Child'
    elif age <= 18 : category = 'Teenager'
    elif age <= 25 : category = 'Student'
    elif age <= 35 : category = 'Young Child'
    elif age <= 60 : category = 'Adult'
    else : category = 'Elderly'

    return category

train_df_pre['Age'] = train_df_pre['Age'].apply(lambda x : ageTransform(x))

#범주형 최빈값으로
features = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
encoding_feature(features)

encoding_cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP' ]
not_encoding_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

#데이터들 합쳐서
enc_cls = {}
def encoding_labels(x):
    le = LabelEncoder()
    le.fit(x)
    label = le.transform(x)
    enc_cls[x.name] = le.classes_
    return label

dt1 = train_df_pre[encoding_cols].apply(encoding_labels)
dt2 = train_df_pre[not_encoding_cols]
train_df_pre = dt1.join(dt2)
train_df_pre['Transported'] = train_df['Transported']

train_df_pre.isnull().sum()

from sklearn.model_selection import train_test_split
y_df = train_df_pre['Transported']
X_df = train_df_pre.drop('Transported', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X_df, y_df,
                                                  test_size = 0.3, random_state = 0)
"""
from sklearn.model_selection import GridSearchCV

model = RandomForestClassifier()

params = { 'n_estimators' : [200],
           'max_depth' : [14, 15, 16, 17, 18],
           'min_samples_leaf' : [ 8, 9, 10, 11, 12],
           'min_samples_split' : [1, 2, 4]
            }

rf_clf = RandomForestClassifier(random_state = 0, n_jobs = -1)
grid_cv = GridSearchCV(rf_clf, param_grid = params, cv = 3, n_jobs = -1)
grid_cv.fit(X_train, y_train)

print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))
#최적 하이퍼 파라미터:  {'max_depth': 15, 'min_samples_leaf': 12, 'min_samples_split': 2, 'n_estimators': 200}
#최고 예측 정확도: 0.7947
#최적 하이퍼 파라미터:  {'max_depth': 17, 'min_samples_leaf': 8, 'min_samples_split': 2, 'n_estimators': 200}
#최고 예측 정확도: 0.7956
#최적 하이퍼 파라미터:  {'max_depth': 15, 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 200}
#최고 예측 정확도: 0.7959
"""
rf_clf = RandomForestClassifier(n_estimators= 200,
                                  max_depth= 17,
                                  min_samples_leaf= 12,
                                  min_samples_split= 2,
                                random_state = 11)

rf_clf.fit(X_train, y_train)
"""
scores = cross_val_score(rf_clf, X_df, y_df, cv = 10)
print("scores : ", scores)
for iter_count, accuracy in enumerate(scores) :
    print("교차 검증 {0} 정확도: {1:.4f}".format(iter_count+1, accuracy))

print("평균 정확도: {0:.4f}".format(np.mean(scores)))
"""
#테스트 데이터 전처리
def encoding_feature_test(features):
  for feature in features:
    mode = test_df_pre[feature].mode()[0]
    null_feature = test_df_pre[feature].isnull()
    test_df_pre.loc[null_feature, feature] = mode

features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']
encoding_feature_test(features)

test_df_pre['Age'] = test_df_pre['Age'].apply(lambda x : ageTransform(x))

#범주형 최빈값으로
features = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
encoding_feature(features)

encoding_cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP' ]
not_encoding_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

enc_cls = {}
def encoding_labels(x):
    le = LabelEncoder()
    le.fit(x)
    label = le.transform(x)
    enc_cls[x.name] = le.classes_
    return label
dt1 = test_df_pre[encoding_cols].apply(encoding_labels)
dt2 = test_df_pre[not_encoding_cols]
test_df_pre = dt1.join(dt2)

test_df_pre.isnull().sum()

from sklearn.metrics import accuracy_score

#pred = rf_clf.predict(X_test)
#accuracy = accuracy_score(y_test, pred)
#print('랜덤 포레스트 정확도: {:.4f}'.format(accuracy))

pred = rf_clf.predict(test_df_pre)

pred_df = pd.DataFrame(pred, columns=['Transported'])

pred_df['PassengerId'] = test_df['PassengerId']

cols = pred_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
pred_df = pred_df[cols]

pred_df.to_csv('2114899.csv', index=False)
