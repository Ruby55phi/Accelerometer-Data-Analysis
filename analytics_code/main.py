import os
from analytics_code.utils import transform,make_MA,make_behave_dict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
class accele_predictor():
    """
    これは加速度センサーに基づき行動を分類するモデルを作成するクラスです。
    はじめに、データは以下の規則で用意される必要があります。
    
    ディレクトリ配置やファイル名について
    メインディレクトリ
    ├── 1_行動名
    │   ├── ファイル名_個体名_1.csv
    │   ├── ファイル名_個体名_1.csv
    ├── 2_行動名
    │   ├── ファイル名_個体名_2.csv
    │   ├── ファイル名_個体名_2.csv

    mainとutilは同じディレクトリに配置してください。

    使い方は以下の通りです。
    1.instance = accele_predictor(input_root,output_root)
    2.instance.data_transformer(target_col=[0,10,11,12]) #
    3.instance.data_to_MA(gap=1,term=10) #移動平均的に特徴量を算出
    4.instance.fit(test_size=0.2,n_estimators=500, max_features=7,random_state=42) #ランダムフォレストで学習
    """

    def __init__(self,input_root,output_root,target_extentions=['csv']):
        self.input_root,self.output_root = input_root,output_root
        self.features = ['x','y','z','SMA','SVM','energy','Movement variation','Entropy','Pitch','Roll','Inclination']
        self.target_extentions = target_extentions
        return
    
    def dir_process(self,input_root,output_root,processor,topic,save = True):
        #ディレクトリ内のファイルを処理する関数
        target_extentions = self.target_extentions
        df = []
        behave_dict = {}
        for root, dirs, files in os.walk(input_root):
            behave_dict = make_behave_dict(dirs,behave_dict)
            for file in files:
                if file.split('.')[-1] in target_extentions:
                    try:
                        input_path = os.path.join(root,file)
                        output_path = os.path.join(output_root,os.path.relpath(input_path,input_root))
                        os.makedirs(os.path.dirname(output_path),exist_ok=True)
                        df = processor(input_path,df,topic,behave_dict)
                        if save:
                            df.to_csv(output_path,index=False)
                    except Exception as e:
                        print(f"error occured in {input_path}:{e}")
        return df

    def data_transformer(self,target_col=[0,10,11,12]):
        #データファイルを特徴量へ変換する関数　dir構造は保たれる
        self.dir_process(self.input_root,self.output_root,processor=transform,topic=[['time','x','y','z'],target_col])
        self.input_root = self.output_root
        return
    
    def data_to_MA(self,gap=1,term=10):
        #移動平均的に特徴量を再計算
        df_list = self.dir_process(self.input_root,self.output_root,processor=make_MA,topic=[gap,term],save=False)
        df = pd.concat(df_list,ignore_index=True)
        df.columns = [f'{i}_{j}' for i in self.features for j in ['min','max','mean','var']]+['label']
        self.df = df
        return
    
    def estimate_rf_time(self,X_train, n_estimators=500, max_features=7):
        # ランダムフォレストの計算量を推定する関数
        n_samples = X_train.shape[0]
        d = max_features
        estimate = n_estimators * n_samples * np.log2(n_samples) * d
        estimate_scaled = estimate / 1e9  # 任意のスケール（例：G単位）
        return f"Estimated computation scale (arb. units): {estimate_scaled:.2f}"

    def fit(self,test_size=0.2,n_estimators=500, max_features=7,random_state=42):
        # 特徴量とラベルに分ける
        X = self.df.drop('label', axis=1)
        y = self.df['label']

        # 学習データとテストデータに分割（例：80%訓練、20%テスト）
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print(self.estimate_rf_time(X_train,n_estimators=n_estimators,max_features=max_features))
        # ランダムフォレストモデル作成
        clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, random_state=random_state)
        clf.fit(X_train, y_train)

        # 予測と評価
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))