import json
import tensorflow as tf
from migguri.utils import *
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error
from migguri.test_log import Logger_LSTM

def test(LoadModelConfig, graph=True, rangeY=None):
    test_save_graphs = LoadModelConfig.test_save + '/graphs'
    test_save_log = LoadModelConfig.test_save + '/log.txt'
    os.makedirs(test_save_graphs)
    
    loaded_model  = tf.keras.models.load_model(LoadModelConfig.model_path)
    CONFIG = dict2class(json.load(open(LoadModelConfig.model_path[:-3] + '.json', 'r', encoding='utf-8 sig')))
    print("테스트데이터셋:", CONFIG.csv_dataset_test_folder)
    
    if LoadModelConfig.test_dataset_setting:
        CONFIG.csv_dataset_test_folder = LoadModelConfig.test_dataset_setting
    
    df = pd.concat([pd.read_csv(CONFIG.csv_dataset_test_folder + '/' + i) for i in os.listdir(CONFIG.csv_dataset_test_folder)], ignore_index=True)
    df.replace(CONFIG.features_considered_change, inplace=True)

    features = df[CONFIG.features_considered]
    features.index = df[CONFIG.INDEX_COLUMN]

    dataset = features.values
    dataset = (dataset - CONFIG.data_mean) / CONFIG.data_std

    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, CONFIG.TARGET_INDEX],
                                                0, None, CONFIG.past_history,
                                                CONFIG.future_target, CONFIG.STEP, CONFIG.STEP_TO_FUTURE)


    # log set
    logger = Logger_LSTM(test_save_log)
    logger.writemodelinfo('LSTM',
                          LoadModelConfig.cmd,
                          LoadModelConfig.model_path,
                          CONFIG.BATCH_SIZE,
                          CONFIG.BUFFER_SIZE,
                          CONFIG.EPOCHS,
                          CONFIG.EVALUATION_INTERVAL,
                          CONFIG.features_considered, CONFIG.TARGET,
                          CONFIG.past_history, CONFIG.future_target,
                          CONFIG.STEP, CONFIG.STEP_TO_FUTURE,
                          CONFIG.csv_dataset_test_folder, LoadModelConfig.test_save)
    
    logger.write(f"Single window of past history : {x_val_multi.shape}")
    logger.write(f"Target temperature to predict : {y_val_multi.shape}")
    
    # print('-------- test --------')
    # print('Single window of past history : {}'.format(x_val_multi.shape))
    # print('Target temperature to predict : {}'.format(y_val_multi.shape))

    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(CONFIG.BATCH_SIZE)
    
    MAE_VALUES = []

    TRUE_YS = np.array([]).reshape((0, int(CONFIG.future_target/CONFIG.STEP_TO_FUTURE)))
    PRED_YS = np.array([]).reshape((0, int(CONFIG.future_target/CONFIG.STEP_TO_FUTURE)))

    TRUE_YS_ORIGINAL = np.array([]).reshape((0, int(CONFIG.future_target/CONFIG.STEP_TO_FUTURE)))
    PRED_YS_ORIGINAL = np.array([]).reshape((0, int(CONFIG.future_target/CONFIG.STEP_TO_FUTURE)))
    
    
    logger.setallnumber(len(val_data_multi))    
    for idx, (x, y) in enumerate(val_data_multi, start=1):
        logger.setheader(f"배치사이즈: {CONFIG.BATCH_SIZE}")
        # print(f'\n■ 배치 {idx}/{len(val_data_multi)} (배치사이즈: {CONFIG.BATCH_SIZE})')

        batch_pred = loaded_model.predict(x)
        
        original_x = x * CONFIG.data_std + CONFIG.data_mean
        original_y =  y * CONFIG.data_std[CONFIG.TARGET_INDEX] + CONFIG.data_mean[CONFIG.TARGET_INDEX]
        original_pred = batch_pred * CONFIG.data_std[CONFIG.TARGET_INDEX] + CONFIG.data_mean[CONFIG.TARGET_INDEX]
        
        MAE = mean_absolute_error(original_y, original_pred)
        MAE_VALUES.append(MAE)
        logger.write(f"배치 {idx}의 MAE: {MAE}")
        # print(f'배치 {idx}의 MAE(복구후): {MAE}')
        
        TRUE_YS = np.concatenate((TRUE_YS, y), axis=0)
        PRED_YS = np.concatenate((PRED_YS, batch_pred), axis=0)

        TRUE_YS_ORIGINAL = np.concatenate((TRUE_YS_ORIGINAL, original_y), axis=0)
        PRED_YS_ORIGINAL = np.concatenate((PRED_YS_ORIGINAL, original_pred), axis=0)
        
        ################################ 그리기
        
        if graph:
            check_step = CONFIG.BATCH_SIZE / 8
            
            for batch in range(CONFIG.BATCH_SIZE):
                if batch % check_step: continue
                if batch >= len(original_x): break
                
                # 복구 후 그래프
                multi_step_plot(original_x[batch], original_y[batch], original_pred[batch],
                                CONFIG.TARGET, CONFIG.TARGET_INDEX, CONFIG.STEP,
                                f"{test_save_graphs}/{idx}-{batch}.jpg",
                                rangeY=rangeY)
                
    
    RESULT_MAE = mean_absolute_error(TRUE_YS_ORIGINAL, PRED_YS_ORIGINAL)
    logger.write("■ 테스트 결과")
    logger.write(f"MAE: {RESULT_MAE}")
    logger.end()