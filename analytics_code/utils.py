import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
関数定義
1.回転行列
2.特長量計算
3.まとめて処理する関数
"""
#1-1ロール角、ピッチ角を計算する関数
def rotate_angle(df):
    df_angle = pd.concat([np.arctan(df['y']/df['z']),np.arctan(-df['x']/np.sqrt(df['y']**2+df['z']**2))],axis=1)
    return df_angle
#1-2回転行列を計算する関数
def rotate_matrix_3d(row_df,row_amgle):
    rx_matrix = np.array([[1,0,0],
                          [0,np.cos(row_amgle[0]),-np.sin(row_amgle[0])],
                          [0,np.sin(row_amgle[0]),np.cos(row_amgle[0])]])
    ry_matrix = np.array([[np.cos(row_amgle[1]), 0, np.sin(row_amgle[1])],
                          [0, 1, 0],
                          [-np.sin(row_amgle[1]), 0, np.cos(row_amgle[1])]])
    return (ry_matrix@rx_matrix@row_df.T)
#1-3データを回転する関数
def rotate_transform(df):
    df_angle = rotate_angle(df)
    df_transformed = df.copy()
    df_transformed[['x', 'y', 'z']] = df_transformed[['x', 'y', 'z']].astype(float)
    for i, row in df[['x','y','z']].iterrows():
        df_transformed.iloc[i,1:] = rotate_matrix_3d(row,df_angle.iloc[i,:])
    return df_transformed

#2特長量を計算する関数
def features(df):
    df = df.dropna()
    df['SMA'] = np.abs(df['x'])+np.abs(df['y'])+np.abs(df['z'])
    df['SVM'] = np.sqrt(df['x']**2+df['y']**2+df['z']**2)
    df['energy'] = df['SVM']**2
    df['Movement variation'] = np.abs(df['x'].diff())+np.abs(df['y'].diff())+np.abs(df['z'].diff())
    df['Entropy'] = ((1+df['x']+df['y']+df['z'])**2)*np.log(1+(df['x']+df['y']+df['z'])**2)
    df['Pitch'] = np.arctan(-df['x']/np.sqrt(df['y']**2+df['z']**2))*180/np.pi
    df['Roll'] = np.atan2(df['y'],df['z'])*180/np.pi
    df['Inclination'] = np.arctan(np.sqrt(df['x']**2+df['y']**2)/df['z'])*180/np.pi
    return df
#3ファイルを処理する関数
def transform(file_path,df,topic=[['time','x','y','z'],[0,10,11,12]], dict={}):
    col_list = topic
    df = pd.read_csv(file_path)
    df = df.dropna()    
    df_motion = df.iloc[:,col_list[1]]
    df_motion.columns = col_list[0]
    df_motion = df_motion.dropna()
    df_motion['time'] = pd.date_range(start=df_motion['time'][0], periods=len(df_motion), freq='100ms')
    df_transformed = rotate_transform(df_motion)
    df_features = features(df_transformed)
    return df_features

"""
リストや辞書を作る関数
"""
def make_behave_dict(dirs,dict):
    #behave_dict = {1:'横臥睡眠'...}
    if len(dirs)>0:
        for dir in dirs:
            if dir.split('_')[1] not in dict.values() and dir.split('_')[0] not in dict.keys():
                dict[dir.split('_')[0]] = dir.split('_')[1]
    return dict

def behave_classifier(ind):
    return ind.split('.')[0].split('-')[0]

"""
移動平均的に特徴量を算出し直す関数
"""
def make_MA(file_path,df,topic=[1,10], behave_dict={}):
    file_name = behave_classifier(file_path.split('_')[-1])
    df_list = df
    gap, term = topic
    df_act = pd.read_csv(file_path).iloc[:,1:]
    data_np = df_act.to_numpy()
    label_code = behave_classifier(file_name)
    label = behave_dict[label_code]
    for i in range(len(df_act)//gap-term):
        data_i = data_np[i*gap:i*gap+term,:]
        features_i = np.concatenate([np.min(data_i,axis=0),np.max(data_i,axis=0),np.mean(data_i,axis=0),np.var(data_i,axis=0)])
        row = pd.DataFrame(features_i).T
        row['label'] = label
        df_list.append(row)
    return df_list