#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""Emotion model of video features.
https://tynbl.github.io/docs/python-xxxy-3/06-04/06-04.html
https://medium.com/@yehjames/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-5%E8%AC%9B-%E6%B1%BA%E7%AD%96%E6%A8%B9-decision-tree-%E4%BB%A5%E5%8F%8A%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97-random-forest-%E4%BB%8B%E7%B4%B9-7079b0ddfbda
https://martychen920.blogspot.com/2017/09/ml-gridsearchcv.html
https://stackoverflow.com/questions/50290273/gridsearchcv-representation-of-each-class-in-each-part-of-the-dataframe
https://blog.csdn.net/u012969412/article/details/72973055
http://sklearn.apachecn.org/cn/0.19.0/modules/model_evaluation.html
https://www.jianshu.com/p/4439913ed73f
https://zhuanlan.zhihu.com/p/37652854
"""
import copy
import datetime
import time

import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.utils import shuffle
from sklearn import metrics
import pickle


def read_form(form_path, form_type='csv'):
    """讀取保存特征的csv
    """
    if form_type == 'csv':
        return pd.read_csv(form_path)
    elif form_type == 'xlsx':
        return pd.read_excel(form_path)


def z_score_df_column(dataframe):
    """對每一種數值做z-score標準化
    """
    if 'Unnamed: 0' in dataframe.columns:
        dataframe = dataframe.drop('Unnamed: 0', axis=1)
    for col in dataframe.columns:
        if col == 'emotion_type':
            continue
        else:
            dataframe[col] = (dataframe[col] - dataframe[col].mean()) / dataframe[col].std(ddof=0)
    return dataframe


def recreate_binary_dateset(dataframe, key='emotion_type', key_base='happy'):
    """binary classfier之前要先對樣本影片做平均處理
    先改變類型成兩類，再從not的類型隨機抽出和yes同樣數量的樣本
    """
    # Change emotion type.
    for index, clip_df in dataframe.iterrows():
        if clip_df[key] != key_base:
            # Cannot use clip_df[1][key] to change. It will be invalid.
            # SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame. See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
            dataframe[key][index] = 'Not_' + key_base

    # Create balance dataset.
    N_df = dataframe[dataframe[key] != key_base]
    Y_df = dataframe[dataframe[key] == key_base]
    # https://zhuanlan.zhihu.com/p/34191885
    random_N_df = N_df.sample(n=len(Y_df))
    balance_dataset = pd.concat([Y_df, random_N_df], ignore_index=True)

    return balance_dataset


def train_and_verify(x_train, y_train, x_test, y_test, model_config, cv_val=5, scoring_type=['accuracy'], is_xgboost=False):
    """訓練某種模型, 并計算評估分數
    scoring_type：評估分數的類型
    """
    scoring_dict = {}
    for score_type in scoring_type:
        model = model_config[0]
        parameters = model_config[1]
        if parameters is not None:
            # which models need parameters.
            # https://martychen920.blogspot.com/2017/09/ml-gridsearchcv.html
            clf = GridSearchCV(model, parameters, cv=cv_val,
                               scoring=score_type)
            clf.fit(x_train, y_train)
            print('Best parameter:', clf.best_params_)
            print('Best score (%s): %.3f' % (score_type, clf.best_score_))
        else:
            # which models do not need parameters eg. Naive Bayes.
            model.fit(x_train, y_train)
            clf = model
        
        if is_xgboost:
            # plot feature importance
            # fig,ax = plt.subplots(figsize=(15, 15))
            ax = plt.subplots(figsize=(15, 15))[1]
            plot_importance(
                model,
                height=0.5,
                ax=ax,
                max_num_features=64
            )
            plt.show()
            plt.clf()

        test_score = clf.score(x_test, y_test)
        print('Test %s: %.3f' % (score_type, test_score))
        
        scoring_dict[score_type] = {
            'model': clf,
            'score': test_score
        }

    return scoring_dict


def verify_confusion_matrix(y_test, y_pred, labels=None):
    """計算confusion matrix
    """
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    if labels:
        confusion_matrix = metrics.confusion_matrix(
            y_test, y_pred, labels)
    else:
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    return confusion_matrix


def calculate_confusion_matrix(confusion_matrix, is_binary=True):
    if is_binary:
        tp = confusion_matrix[1, 1]
        tn = confusion_matrix[0, 0]
        fp = confusion_matrix[0, 1]
        fn = confusion_matrix[1, 0]


def plot_confusion_matrix(confusion_matrix, labels, title, output_path=None):
    """畫出confusion matrix
    """
    cm_df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    ax = plt.subplots()[1]
    # sn.heatmap(cm_df, cmap='Greys', annot=True, ax=ax, fmt='g')
    sn.heatmap(cm_df, annot=True, ax=ax, fmt='g')
    ax.set_title(title, fontsize='large', fontweight='bold', position=(0.5, 1.1))
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.xaxis.tick_top()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.clf()
    return cm_df


def train_model(x_train, y_train, x_test, y_test, scoring_type=['accuracy']):
    # GB vs GBDT vs XGBoost: https://bigdatafinance.tw/index.php/tech/methodology/572-gb-gbdt-xgboost
    # https://hk.saowen.com/a/49aaf0f2412a4cf09d91ceec03a4106cc315ff6ca3a1f3a50134d1fefedba4cf
    # ValueError: Parameter values for parameter (learning_rate) need to be a sequence(but not a string) or np.ndarray.
    # 因为等等要进 GridSearchCV 且 cv_val=5， 所以parameters要变可迭代的，而且看cv_val是多少就要几个元素(不确定...)
    # xgboost_parameters = {
    #     'learning_rate': 0.1,
    #     'n_estimators': 1000,         # 樹的個數--1000棵樹創建xgboost
    #     'max_depth': 6,               # 樹的深度
    #     'min_child_weight': 1,      # 葉子節點最小權重
    #     'gamma': 0.,                  # 懲罰項中葉子結點個數前的參數
    #     'subsample': 0.8,             # 隨機選擇80%樣本創建樹
    #     'colsample_btree': 0.8,       # 隨機算哦80%樣本選擇特徵
    #     'objective': 'multi:softmax',  # 指定損失函數
    #     'scale_pos_weight': 1,        # 解決樣本個數不平衡的問題
    #     'random_state': 27            # 隨機數
    # }
    # xgboost_parameters = {
    #     'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05],
    #     'n_estimators': range(30, 151, 20),
    #     'max_depth': range(3, 10, 2),
    #     'min_child_weight': range(1, 6, 2),
    #     'gamma': [0., 0., 0., 0., 0.],
    #     'subsample': [0.8, 0.8, 0.8, 0.8, 0.8],
    #     'colsample_btree': [0.8, 0.8, 0.8, 0.8, 0.8],
    #     'objective': ['multi:softmax', 'multi:softmax', 'multi:softmax', 'multi:softmax', 'multi:softmax'],
    #     'scale_pos_weight': [1, 1, 1, 1, 1],
    #     'random_state': [27, 27, 27, 27, 27],
    #     # xgboost.core.XGBoostError: b'value 0 for Parameter num_class should be greater equal to 1'
    #     # https://stackoverrun.com/cn/q/11048264
    #     'num_class': [5, 5, 5, 5, 5]
    # }
    model_dict = {
        'kNN':    (KNeighborsClassifier(),               {'n_neighbors': [5, 10, 15]}),
        'LR':     (LogisticRegression(),                    {'C': [0.01, 1, 100]}),
        'SVM':    (SVC(),                                                 {'C': [100, 1000, 10000]}),
        'DT':     (DecisionTreeClassifier(),            {'max_depth': [50, 100, 150]}),
        'GNB':    (GaussianNB(), None),
        # https://medium.com/@yehjames/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-5%E8%AC%9B-%E6%B1%BA%E7%AD%96%E6%A8%B9-decision-tree-%E4%BB%A5%E5%8F%8A%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97-random-forest-%E4%BB%8B%E7%B4%B9-7079b0ddfbda
        'RF':     (RandomForestClassifier(),            {'n_estimators': [100, 150, 200]}),
        # https://blog.csdn.net/zwqjoy/article/details/80424783
        'GBDT':   (GradientBoostingClassifier(),  {'learning_rate': [0.1, 1, 10]})
        # # XGBoost Memory error: https://www.kaggle.com/c/grupo-bimbo-inventory-demand/discussion/22321
        # 'XGBoost': (XGBClassifier(),                         xgboost_parameters)
    }

    # Multiple scoring.
    columns = []
    for score_type in scoring_type:
        columns.append(score_type)
    results_df = pd.DataFrame(
        columns=columns, index=list(model_dict.keys()))

    results_df.index.name = 'Model'
    trained_models_dict = {}

    for model_name, model_config in model_dict.items():
        print('Train model:', model_name)
        if model_name == 'XGBoost':
            scoring_dict = train_and_verify(x_train, y_train, x_test, y_test,
                                            model_config,
                                            scoring_type=scoring_type,
                                            is_xgboost=True)
        else:
            scoring_dict = train_and_verify(x_train, y_train, x_test, y_test,
                                            model_config,
                                            scoring_type=scoring_type)
        for score_type, data in scoring_dict.items():
            if score_type not in trained_models_dict:
                trained_models_dict[score_type] = []
            trained_models_dict[score_type].append(data['model'])
            results_df.loc[model_name, score_type] = data['score']
        print()

    return results_df, trained_models_dict


def emotion_classifier_model(dataframe, result_store_folder=None, model_result_plot_folder=None, best_model_pkl_folder=None, matrix_folder=None, normal='original', scoring_type=['accuracy'], best_score_base=None, is_binary=False, key_base=None):
    feature_df = dataframe.drop('emotion_type', axis=1)
    if 'Unnamed: 0' in feature_df.columns:
        feature_df = feature_df.drop('Unnamed: 0', axis=1)

    if normal == 'z-score':
        feature_df = z_score_df_column(feature_df)

    feat_dim = len(feature_df.columns)
    print('Feature Dim:', feat_dim)
    samples_length = len(feature_df)
    print('Samples Length:', samples_length)

    # Fill feature matrix.
    feature_matrix = np.zeros((samples_length, feat_dim))
    for index, row in feature_df.iterrows():
        feature_matrix[index, :] = copy.deepcopy(np.array(row.values))
    print(feature_matrix.shape)

    # Label fit array.
    target_names = dataframe['emotion_type'].values
    print(target_names)
    label_enc = LabelEncoder()
    label_fit_transform = label_enc.fit_transform(target_names)
    print('Video Type:', label_enc.classes_)
    print('Label Fit Array:', label_fit_transform)

    # Split dataset.
    # random_state: https://blog.csdn.net/Tony_Stark_Wang/article/details/80407923
    # 这里的random_state就是为了保证程序每次运行都分割一样的训练集合测试集。否则，同样的算法模型在不同的训练集和测试集上的效果不一样。
    # 当你用sklearn分割完测试集和训练集，确定模型和促初始参数以后，你会发现程序每运行一次，都会得到不同的准确率，无法调参。这个时候就是因为没有加random_state。加上以后就可以调参了。
    # https://www.datacamp.com/community/tutorials/scikit-learn-python
    x_train, x_test, y_train, y_test = train_test_split(
        feature_matrix, label_fit_transform, test_size=1 / 4, random_state=0)
    print('Train Dataset Length:', len(x_train))
    print('Test Dataset Length:', len(x_test))

    # Train and verify.
    results_df, trained_models_dict = train_model(
        x_train, y_train, x_test, y_test, scoring_type=scoring_type)

    # confusion matrix
    # model_types = ['RF']
    model_types = ['kNN', 'LR', 'SVM', 'DT', 'GNB', 'RF', 'GBDT']
    labels = label_enc.classes_.tolist()
    labels_number = list(set(label_fit_transform.tolist()))
    if matrix_folder:
        if matrix_folder[-1] != '/':
            matrix_folder += '/'
    for model_type in model_types:
        # predict
        trained_model = trained_models_dict[scoring_type[0]][model_types.index(
                model_type)]
        y_pred = trained_model.predict(x_test)
        y_pred_labeled = label_enc.inverse_transform(y_pred)
        print('Predict:', y_pred)
        print('Predict (Label):', y_pred_labeled)

        print('【%s】Confusion_matrix' % model_type)
        confusion_matrix = verify_confusion_matrix(y_test, y_pred)
        print(confusion_matrix)
        if matrix_folder:
            current_matrix_path = matrix_folder + \
                model_type + '_multiclass_confusion_matrix'
            title = 'Confusion matrix '
            if is_binary:
                current_matrix_path = matrix_folder + \
                    model_type + '_binary_' + key_base + '_confusion_matrix'
                title += 'of binary '
            title += key_base + ' (' + model_type + ' model)'
            cm_df = plot_confusion_matrix(
                confusion_matrix, labels_number, title, output_path=current_matrix_path + '.jpg')
            cm_df.to_csv(current_matrix_path + '.csv')

        print('【%s】Confusion_matrix_labeled' % model_type)
        y_test_labeled = label_enc.inverse_transform(y_test)
        confusion_matrix_labeled = verify_confusion_matrix(
            y_test_labeled, y_pred_labeled, labels=labels)
        print(confusion_matrix_labeled)
        if matrix_folder:
            current_matrix_path = matrix_folder + \
                model_type + '_multiclass_labeled_confusion_matrix'
            title = 'Confusion matrix '
            if is_binary:
                current_matrix_path = matrix_folder + \
                    model_type + '_binary_labeled_' + key_base + '_confusion_matrix'
                title += 'of binary '
            title += key_base + ' (with labels) (' + model_type + ' model)'
            cm_df = plot_confusion_matrix(
                confusion_matrix_labeled, labels, title, output_path=current_matrix_path + '.jpg')
            cm_df.to_csv(current_matrix_path + '.csv')
    
    has_multiple_scoring = True
    if len(scoring_type) == 1:
        has_multiple_scoring = False

    # Store result.
    if result_store_folder:
        if result_store_folder[-1] != '/':
            result_store_folder += '/'
        current_result_store_path = result_store_folder + 'multi_score_pred_results.csv'
        if is_binary:
            current_result_store_path = result_store_folder + \
                'multi_score_binary_' + key_base + '_pred_results.csv'
            if not has_multiple_scoring:
                current_result_store_path = result_store_folder + \
                    best_score_base + '_score_binary_' + key_base + '_pred_results.csv'
        else:
            if not has_multiple_scoring:
                current_result_store_path = result_store_folder + best_score_base + '_score_pred_results.csv'
        results_df.to_csv(current_result_store_path)
        # https://blog.csdn.net/brucewong0516/article/details/80524442
        results_df.plot(kind='bar', figsize=(15, 10))
        plt.ylabel('score')
        plt.tight_layout()
        if model_result_plot_folder:
            if model_result_plot_folder[-1] != '/':
                model_result_plot_folder += '/'
            current_path = model_result_plot_folder + 'multi_score_pred_results.png'
            if is_binary:
                current_path = model_result_plot_folder + \
                    'multi_score_binary_' + key_base + '_pred_results.png'
                if not has_multiple_scoring:
                    current_path = model_result_plot_folder + best_score_base + \
                        '_score_binary_' + key_base + '_pred_results.png'
            else:
                if not has_multiple_scoring:
                    current_path = model_result_plot_folder + \
                        best_score_base + '_score_pred_results.png'
            plt.savefig(current_path)
            # plt.show()
            plt.clf()

    # Store best model.
    if best_model_pkl_folder:
        if best_model_pkl_folder[-1] != '/':
            best_model_pkl_folder += '/'
        if best_score_base:
            current_best_model_pkl_path = best_model_pkl_folder + best_score_base + '_predictor.pkl'
            if is_binary:
                current_best_model_pkl_path = best_model_pkl_folder + \
                    best_score_base + '_binary_' + key_base + '_predictor.pkl'
            # https://blog.csdn.net/jingyi130705008/article/details/78162758
            best_model_idx = results_df.reset_index()[best_score_base].argmax()
            best_model = trained_models_dict[best_score_base][best_model_idx]
            with open(current_best_model_pkl_path, 'wb') as f:
                pickle.dump(best_model, f)
        else:
            for score_type in scoring_type:
                current_best_model_pkl_path = best_model_pkl_folder + score_type + '_predictor.pkl'
                if is_binary:
                    current_best_model_pkl_path = best_model_pkl_folder + \
                        score_type + '_binary_' + key_base + '_predictor.pkl'
                best_model_idx = results_df.reset_index()[score_type].argmax()
                best_model = trained_models_dict[score_type][best_model_idx]
                with open(current_best_model_pkl_path, 'wb') as f:
                    pickle.dump(best_model, f)


def predict(model_path, dataframe):
    with open(model_path, 'rb') as f:
        predictor = pickle.load(f)

    target_names = dataframe['emotion_type'].values
    test_df = dataframe.drop('emotion_type', axis=1)
    if 'Unnamed: 0' in test_df.columns:
        test_df = test_df.drop('Unnamed: 0', axis=1)
    feature_array = np.array(test_df.head(1))

    label_enc = LabelEncoder()
    label_enc.fit_transform(target_names)
    pred_result = predictor.predict(feature_array.reshape(1, -1))
    pred_genre = label_enc.inverse_transform(pred_result)
    print('Prediction type:', pred_genre)
    print('Actual type:', dataframe.head(1)['emotion_type'].values)


if __name__ == '__main__':
    form_path = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/video_feature.csv'
    form_type = 'csv'
    normal = 'z-score'
    result_store_folder = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/pred_results'
    model_result_plot_folder = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/chart_output/model_result_plot'
    best_model_pkl_folder = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/best_predictor_model'
    matrix_folder = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/confusion_matrix'
    # http://sklearn.apachecn.org/cn/0.19.0/modules/model_evaluation.html
    # ValueError: Target is multiclass but average='binary' solv.: https://stackoverflow.com/questions/50290273/gridsearchcv-representation-of-each-class-in-each-part-of-the-dataframe

    # # Multiclass.
    # scoring_type = [
    #     'accuracy',
    #     'f1_micro',
    #     'precision_micro',
    #     'recall_micro',
    #     'f1_macro',
    #     'precision_macro',
    #     'recall_macro',
    #     'f1_weighted',
    #     'precision_weighted',
    #     'recall_weighted'
    #     # # ValueError: Sample-based precision, recall, fscore is not meaningful outside multilabel classification. See the accuracy_score instead.
    #     # 'f1_samples',
    #     # 'precision_samples',
    #     # 'recall_samples',
    #     # # ValueError: multiclass format is not supported
    #     # 'average_precision',
    #     # 网格搜索scoring=’roc_auc’只支持二分类，多分类需要修改scoring(默认支持多分类)
    #     # 'roc_auc'
    # ]
    # best_score_base = None

    # start = time.clock()
    # print('Start time:', datetime.datetime.now(), '\n')

    dataframe = read_form(form_path, form_type=form_type)
    # emotion_classifier_model(
    #     dataframe,
    #     result_store_folder=result_store_folder,
    #     model_result_plot_folder=model_result_plot_folder,
    #     best_model_pkl_folder=best_model_pkl_folder,
    #     matrix_folder=matrix_folder,
    #     normal=normal,
    #     scoring_type=scoring_type,
    #     best_score_base=best_score_base
    # )
    # print('\nTraining Finish!\n')
    
    # for score_type in scoring_type:
    #     print('Predict with %s_predictor model' % score_type)
    #     current_best_model_pkl_path = best_model_pkl_folder + \
    #         '/' + score_type + '_predictor.pkl'
    #     predict(current_best_model_pkl_path, dataframe)
    
    # print('\nStop time:', datetime.datetime.now())
    # elapsed = (time.clock() - start)
    # print("Time used:", elapsed)

    # print('\n======\n')

    # Binary class.
    key_base_list = [
        'happy',
        'sad',
        'angry',
        'fear',
        'disgust',
        'surprise'
    ]
    scoring_type = [
        'accuracy',
        'f1',
        'precision',
        'recall',
        'roc_auc'
    ]
    best_score_base = None

    for key_base in key_base_list:
        print('【%s】' % key_base)
        start = time.clock()
        print('Start time:', datetime.datetime.now(), '\n')
        binary_dataset = recreate_binary_dateset(copy.deepcopy(
            dataframe), key='emotion_type', key_base=key_base)
        binary_dataset = shuffle(binary_dataset)
        emotion_classifier_model(
            binary_dataset,
            result_store_folder=result_store_folder,
            model_result_plot_folder=model_result_plot_folder,
            best_model_pkl_folder=best_model_pkl_folder,
            matrix_folder=matrix_folder,
            normal=normal,
            scoring_type=scoring_type,
            best_score_base=best_score_base,
            is_binary=True,
            key_base=key_base
        )
        print('\nTraining Finish!\n')

        # for score_type in scoring_type:
        #     print('Predict with %s_predictor model' % score_type)
        #     current_best_model_pkl_path = best_model_pkl_folder + \
        #         '/' + score_type + '_binary_' + key_base + '_predictor.pkl'
        #     predict(current_best_model_pkl_path, binary_dataset)

        print('\nStop time:', datetime.datetime.now())
        elapsed = (time.clock() - start)
        print("Time used:", elapsed)

        print('\n======\n')
