import warnings

warnings.simplefilter(action='ignore', category=Warning)
import gc
from sklearn.model_selection import train_test_split, LeaveOneOut, RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, GroupKFold, GroupShuffleSplit
import pandas as pd
import numpy as np
import matplotlib
import time

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
import seaborn as sns
import lightgbm as lgb

pd.set_option('display.max_columns', 10)

sub = pd.read_csv('sample_submission.csv')

train = pd.read_csv('X_train.csv')
train_df = train[['series_id']].drop_duplicates().reset_index(drop=True)

train_sub = train.iloc[:, 3:]
# Mean , Std , Quartiles plot
train_desc = train_sub.describe()
train_desc = train_desc.drop(['count'])
train_desc.plot.bar()
# plt.show()

for col in train_sub.columns:
    train_df[col + '_mean'] = train.groupby(['series_id'])[col].mean()
    train_df[col + '_std'] = train.groupby(['series_id'])[col].std()
    train_df[col + '_max'] = train.groupby(['series_id'])[col].max()
    train_df[col + '_min'] = train.groupby(['series_id'])[col].min()
    train_df[col + '_max_to_min'] = train_df[col + '_max'] / train_df[col + '_min']

# --------------------------Test------------------------------------------------

test = pd.read_csv('X_test.csv')
test_df = test[['series_id']].drop_duplicates().reset_index(drop=True)
test_sub = test.iloc[:, 3:]

# Mean , Std , Quartiles plot
test_desc = test_sub.describe()
test_desc = test_desc.drop(['count'])
test_desc.plot.bar()
# plt.show()

for col in test_sub.columns:
    test_df[col + '_mean'] = test.groupby(['series_id'])[col].mean()
    test_df[col + '_std'] = test.groupby(['series_id'])[col].std()
    test_df[col + '_max'] = test.groupby(['series_id'])[col].max()
    test_df[col + '_min'] = test.groupby(['series_id'])[col].min()
    test_df[col + '_max_to_min'] = test_df[col + '_max'] / test_df[col + '_min']

y_df = pd.read_csv('y_train.csv')

le = LabelEncoder()
le.fit(y_df['surface'])
y_df['surface'] = le.transform(y_df['surface'])

train_df = train_df.drop(['series_id'], axis=1)
scaler = StandardScaler().fit(train_df)
scaler_train = scaler.transform(train_df)
scaler_train_df = pd.DataFrame(scaler_train)
train_df = scaler_train_df

test_df = test_df.drop(['series_id'], axis=1)
scaler_test = scaler.transform(test_df)
scaler_test_df = pd.DataFrame(scaler_test)
test_df = scaler_test_df

kf = StratifiedKFold(n_splits=20, random_state=123, shuffle=False)
# kf = GroupKFold( n_splits=2)
# kf = GroupShuffleSplit(n_splits=4, test_size=0.2, random_state=0)
# kf = LeaveOneOut()
# kf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=None)

print kf.get_n_splits(train_df)
oof_preds = np.zeros((len(train_df),))
sub_preds = np.zeros((len(test_df), 9))

predicts = []
score = []
features = [f_ for f_ in train_df.columns]
groups = y_df['group_id']
y_df = y_df['surface']

print groups.shape
print y_df.shape
print train_df.shape

for fold_n, (train_index, valid_index) in enumerate(kf.split(train_df, y_df, groups=groups)):
    print('Fold', fold_n, 'started at', time.ctime())
    X_train, X_valid = train_df.iloc[train_index], train_df.iloc[valid_index]
    y_train, y_valid = y_df.iloc[train_index], y_df.iloc[valid_index]
    # print train_index, valid_index

    # 'num_leaves': 1024
    params = {'num_leaves': 51,
              'min_data_in_leaf': 10,
              'objective': 'multiclass',
              'max_depth': -1,
              'learning_rate': 0.001,
              'bagging_freq': 5,
              'max_bin': 200,
              'feature_fraction': 0.91,
              'random_state': 42
              }

    best_grid = {'bootstrap': False,
                 'min_samples_leaf': 4,
                 'n_estimators': 600,
                 'max_features': 'sqrt',
                 'min_samples_split': 10,
                 'max_depth': 90}

    print "lgb Classifier ...."
    model_lgb = lgb.LGBMClassifier(boosting_type='gbdt',
                                   objective='multiclass',
                                   metric='multi_logloss',
                                   n_jobs=3,
                                   silent=True,
                                   max_depth=params['max_depth'],
                                   num_leaves=params['num_leaves'],
                                   min_data_in_leaf=params['min_data_in_leaf'],
                                   bagging_freq=params['bagging_freq'],
                                   max_bin=params['max_bin'],
                                   feature_fraction=params['feature_fraction'],
                                   random_state=params['random_state']
                                   )

    model_lgb.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='multi_logloss',
                  verbose=1000, early_stopping_rounds=200)

    y_pred_valid = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration_)
    y_pred = model_lgb.predict_proba(test_df, num_iteration=model_lgb.best_iteration_)
    oof_preds[valid_index] = y_pred_valid
    score.append(f1_score(y_valid, y_pred_valid, average='weighted'))
    sub_preds += y_pred

    print('Precision Score : ' + str(precision_score(y_valid, y_pred_valid, average='weighted')))
    print('Recall Score : ' + str(recall_score(y_valid, y_pred_valid, average='weighted')))
    print('F1 Score : ' + str(f1_score(y_valid, y_pred_valid, average='weighted')))

sub_preds /= kf.get_n_splits(test_df)
print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(score), np.std(score)))
# print ('oof: {}, sub: {}.'.format(oof_preds, sub_preds))
sub['surface'] = le.inverse_transform(sub_preds.argmax(1))
sub.to_csv('lgb_sub.csv', index=False)

#
#
#     oof_preds[val_idx] = model_lgb.predict_proba(val_x, num_iteration=model_lgb.best_iteration_)
#     #sub_preds += model_lgb.predict_proba(test_df[features], num_iteration=model_lgb.best_iteration_) / kf.n_splits
#     y_pred_valid = model_lgb.predict_proba(val_x)
#     y_pred = model_lgb.predict_proba(test_df, num_iteration=model_lgb.best_iteration_)
#     #oof_preds[val_idx] = y_pred_valid
#     score.append(accuracy_score(val_y, oof_preds[val_idx]))
#     sub_preds += y_pred
#     sub_preds /= kf.n_splits
#     print val_y
#     print y_pred_valid.argmax(1)
#     print accuracy_score(val_y, oof_preds[val_idx])
#
#     sub_preds += y_pred
#
#     #print ('oof: {}, sub: {}.'.format(oof_preds, sub_preds))
#     #print('Fold %2d AC : %.6f' % (n_fold + 1, accuracy_score(val_y, y_pred_valid)))
#     del model_lgb, trn_x, trn_y, val_x, val_y
#     gc.collect()
#
# print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(score), np.std(score)))


# for (train_index, test_index), i in zip(kf.split(train_df, y_df['surface']), range(5)):
#     le[i] = LabelEncoder()
#     le[i].fit(y_df['surface'])
#     y_df['surface'] = le[i].transform(y_df['surface'])
#
#     model_lgb.fit(train_df.iloc[train_index], y_df['surface'].iloc[train_index],
#                   eval_metric='auc', verbose=250, early_stopping_rounds=150)
#     y_pred = model_lgb.predict(train_df)
#     predicts.append(accuracy_score(y_df['surface'], y_pred))
#     y_df['surface'] = le[i].inverse_transform(y_pred)
#     #print y_df['surface']


# Building Model -------------------------------------------------------------------
#
# print "SVM ..."
# model = SVC(probability=True)
# model.fit(X_train, y_train['surface'])
# y_pred = model.predict(X_test)
# print accuracy_score(y_test['surface'], y_pred)
#
# print "Decision Tree Classifier ...."
# model_decisiontree = DecisionTreeClassifier(criterion="entropy",
#                                             random_state=100, max_depth=3, min_samples_leaf=5)
# model_decisiontree.fit(X_train, y_train['surface'])
# y_pred = model.predict(X_test)
# print accuracy_score(y_test['surface'], y_pred)
#
# print "Knn Classifier ...."
# model_knn = KNeighborsClassifier(n_neighbors=5)
# model_knn.fit(X_train, y_train['surface'])
# y_pred = model.predict(X_test)
# print accuracy_score(y_test['surface'], y_pred)
#
# params = {'num_leaves': 150,
#           'min_data_in_leaf': 12,
#           'objective': 'multiclass',
#           'max_depth': 20,
#           'learning_rate': 0.04680350949723872,
#           "boosting": "gbdt",
#           "bagging_freq": 5,
#           }
#
# print "lgb Classifier ...."
# model_lgb = lgb.LGBMClassifier(boosting_type='gbdt',
#                                objective='multiclass',
#                                n_jobs=3,
#                                silent=True,
#                                max_depth=params['max_depth'],
#                                num_leaves=params['num_leaves'],
#                                learning_rate=params['learning_rate']
#                                )
#
#
# model_lgb.fit(X_train, y_train['surface'])
# y_pred = model_lgb.predict(X_test)
# print accuracy_score(y_test['surface'], y_pred)
#
# # sorted(zip(clf.feature_importances_, X.columns), reverse=True)
# feature_imp = pd.DataFrame(sorted(zip(model_lgb.feature_importances_, X_train.columns)), columns=['Value', 'Feature'])
#
# plt.figure(figsize=(20, 10))
# sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
# plt.title('LightGBM Features (avg over folds)')
# plt.tight_layout()
# plt.show()
