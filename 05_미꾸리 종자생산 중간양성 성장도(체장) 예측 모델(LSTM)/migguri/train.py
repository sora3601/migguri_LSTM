import os
import pandas as pd
import json
import numpy as np
from migguri.utils import *

def sorted_df_by_gsi(df):
    maturation_period = df['maturation_period'].copy()
    df = df.sort_values(by='gsi').reset_index(drop=True)
    df['maturation_period'] = maturation_period
    df['gsi'] = df['gsi'].apply(lambda x: 30 if x >= 30 else x)
    return df

def range_limit(df):
    # ★★★★★★ 범위 확인해 ★★★★★★
    df = df.copy()

    if 'water_temp' in df:
        df['water_temp'] = df['water_temp'].apply(lambda x: 5 if x <= 5 else x)
        df['water_temp'] = df['water_temp'].apply(lambda x: 28.5 if x > 28.5 else x)

    if 'water_do' in df:
        df['water_do'] = df['water_do'].apply(lambda x: 1.5 if x <= 1.5 else x)
        df['water_do'] = df['water_do'].apply(lambda x: 6.5 if x > 6.5 else x)

    if 'water_ph' in df:
        df['water_ph'] = df['water_ph'].apply(lambda x: 2 if x <= 2 else x)
        df['water_ph'] = df['water_ph'].apply(lambda x: 7.2 if x > 7.2 else x)

    if 'water_orp' in df:
        df['water_orp'] = df['water_orp'].apply(lambda x: 100 if x <= 100 else x)
        df['water_orp'] = df['water_orp'].apply(lambda x: 401 if x > 401 else x)

    if 'tank_lux' in df:
        df['tank_lux'] = df['tank_lux'].apply(lambda x: 200 if x <= 200 else x)
        df['tank_lux'] = df['tank_lux'].apply(lambda x: 800 if x > 800 else x)

    if 'water_nh3' in df:
        df['water_nh3'] = df['water_nh3'].apply(lambda x: 0 if x <= 0 else x)
        df['water_nh3'] = df['water_nh3'].apply(lambda x: 0.25 if x > 0.25 else x)

    if 'water_no2-' in df:
        df['water_no2'] = df['water_no2'].apply(lambda x: 0 if x <= 0 else x)
        df['water_no2-'] = df['water_no2-'].apply(lambda x: 0.21 if x > 0.21 else x)

    if 'feed_frequency' in df:
        df['feed_frequency'] = df['feed_frequency'].apply(lambda x: 1 if x <= 1 else x)
        df['feed_frequency'] = df['feed_frequency'].apply(lambda x: 4 if x > 4 else x)

    if 'feed_cap' in df:
        df['feed_cap'] = df['feed_cap'].apply(lambda x: 10 if x <= 10 else x)
        df['feed_cap'] = df['feed_cap'].apply(lambda x: 300 if x > 300 else x)

    if 'photoperiod' in df:
        df['photoperiod'] = df['photoperiod'].apply(lambda x: 0 if x <= 0 else x)
        df['photoperiod'] = df['photoperiod'].apply(lambda x: 1 if x > 1 else x)

    if 'body_length' in df:
        df['body_length'] = df['body_length'].apply(lambda x: 0.5 if x <= 0.5 else x)
        df['body_length'] = df['body_length'].apply(lambda x: 16 if x > 16 else x)

    if 'body_weight' in df:
        df['body_weight'] = df['body_weight'].apply(lambda x: 5 if x <= 5 else x)
        df['body_weight'] = df['body_weight'].apply(lambda x: 12.1 if x > 12.1 else x)

    if 'gsi' in df:
        df['gsi'] = df['gsi'].apply(lambda x: 0 if x <= 0 else x)
        df['gsi'] = df['gsi'].apply(lambda x: 15 if x > 15 else x)
        
    return df
        
def train(CONFIG, sorted_by_gsi=False):
    
    if sorted_by_gsi:
        df_train = pd.concat([sorted_df_by_gsi(pd.read_csv(CONFIG.csv_dataset_train_folder + '/' + i)) for i in os.listdir(CONFIG.csv_dataset_train_folder)], ignore_index=True)
        df_valid = pd.concat([sorted_df_by_gsi(pd.read_csv(CONFIG.csv_dataset_valid_folder + '/' + i)) for i in os.listdir(CONFIG.csv_dataset_valid_folder)], ignore_index=True)
    else:        
        df_train = pd.concat([pd.read_csv(CONFIG.csv_dataset_train_folder + '/' + i) for i in os.listdir(CONFIG.csv_dataset_train_folder)], ignore_index=True)
        df_valid = pd.concat([pd.read_csv(CONFIG.csv_dataset_valid_folder + '/' + i) for i in os.listdir(CONFIG.csv_dataset_valid_folder)], ignore_index=True)
    
    CONFIG.TRAIN_SPLIT = len(df_train)
    
    df = pd.concat([df_train, df_valid], ignore_index=True)
    df.fillna(0, inplace=True)
    df.replace(CONFIG.features_considered_change, inplace=True)

    # 관심 있는 특성들을 선택합니다.
    features = df[CONFIG.features_considered]
    features = range_limit(features)
    features.plot(subplots=True)
    # plt.show()
    features.index = df[CONFIG.INDEX_COLUMN]

    # 데이터를 numpy 배열로 변환합니다.
    dataset = features.values

    # 훈련 데이터의 평균과 표준 편차를 계산합니다.
    data_mean = dataset[:CONFIG.TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:CONFIG.TRAIN_SPLIT].std(axis=0)
    CONFIG.data_mean = data_mean.tolist()
    CONFIG.data_std = data_std.tolist()

    # 데이터를 표준화(정규화)합니다.
    dataset = (dataset - data_mean) / data_std

    # multivariate_data 함수를 사용하여 입력 및 출력 데이터를 생성합니다.
    x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, CONFIG.TARGET_INDEX],
                                                    0, CONFIG.TRAIN_SPLIT,CONFIG.past_history,
                                                    CONFIG.future_target, CONFIG.STEP, CONFIG.STEP_TO_FUTURE)
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, CONFIG.TARGET_INDEX],
                                                CONFIG.TRAIN_SPLIT, None, CONFIG.past_history,
                                                CONFIG.future_target, CONFIG.STEP, CONFIG.STEP_TO_FUTURE)

    # 생성된 데이터의 모양을 출력합니다.
    print('------- train --------')
    print('Single window of past history : {}'.format(x_train_multi.shape))
    print('Target temperature to predict : {}'.format(y_train_multi.shape))
    CONFIG.x_train_multi_shape = x_train_multi.shape
    CONFIG.y_train_multi_shape = y_train_multi.shape

    print('-------- val --------')
    print('Single window of past history : {}'.format(x_val_multi.shape))
    print('Target temperature to predict : {}'.format(y_val_multi.shape))
    CONFIG.x_val_multi_shape = x_val_multi.shape
    CONFIG.y_val_multi_shape = y_val_multi.shape

    # TensorFlow를 사용하여 데이터를 처리하기 위한 데이터셋을 생성합니다.
    import tensorflow as tf

    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data_multi = train_data_multi.cache().shuffle(CONFIG.BUFFER_SIZE).batch(CONFIG.BATCH_SIZE).repeat()

    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(CONFIG.BATCH_SIZE).repeat()

    # 훈련 데이터에서 하나의 배치를 가져와서 그래프로 시각화합니다.
    for x, y in train_data_multi.take(0):
        multi_step_plot(x[0], y[0], np.array([0]),
                        CONFIG.TARGET, CONFIG.TARGET_INDEX, CONFIG.STEP)

    # 다층 LSTM 모델 생성 및 컴파일
    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.LSTM(32,
                                            return_sequences=True,
                                            input_shape=x_train_multi.shape[-2:]))
    multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
    multi_step_model.add(tf.keras.layers.Dense(CONFIG.future_target/CONFIG.STEP_TO_FUTURE))

    multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

    multi_step_history = multi_step_model.fit(train_data_multi, epochs=CONFIG.EPOCHS,
                                            steps_per_epoch=CONFIG.EVALUATION_INTERVAL,
                                            validation_data=val_data_multi,
                                            validation_steps=50)

    # 모델 저장
    multi_step_model.save(CONFIG.model_save_path)

    # 학습 결과 시각화
    plot_train_history(multi_step_history, 'Multi-Step Training and validation loss', save_path=CONFIG.model_save_path[:-3]+'.jpg')

    # 메타 데이터를 저장
    CONFIG.loss = multi_step_history.history['loss']
    CONFIG.val_loss = multi_step_history.history['val_loss']
    json.dump(class2dict(CONFIG), open(CONFIG.model_save_path[:-3] + '.json', 'w', encoding='utf-8 sig'), indent=4, ensure_ascii=False)

    # 예측 결과 시각화
    for x, y in val_data_multi.take(0):
        multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0],
                        CONFIG.TARGET, CONFIG.TARGET_INDEX, CONFIG.STEP)