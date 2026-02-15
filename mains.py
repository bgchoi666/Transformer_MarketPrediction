from systems import *
import math
import time
import os
import sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='2, 3'   #use gpu number
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 종목과 종가 변수 리스트
li_item_name = ['메리츠화재']
li_price_column = ['수정주가(원)']

# 데이터 전처리 파라미터
nan_thresh = 70
fill_na = 'interpolation'
target_type = 'trend_ratio'

#lookback_days = [20,22,40,42,60,80,120,240,300]
lookback_days = [20,60,120,240]
prediction_day = 20
time_intervals = [1,2]

# 차원축소 모델 파라미터
li_coding_size = 256
# 예측 모델 파라미터
model_name = 'AutoEncoderLSTM'
lstm_unit = 800
#window_size = math.ceil(lookback_days/time_interval)
# 학습과정 파라미터
test_period = 20
train_period = 500

batch_size = 32
learing_rate = 0.0005

begin_pred_date = '2015-01-02' # first test prediction day
begin_iteration = 1000
iteration = 500

for item_name, price_column in zip(li_item_name, li_price_column):
    for lookback_day in lookback_days:
        for time_interval in time_intervals:
            print(f"종목명 : {item_name}       타겟 : {price_column}")
            market_prediction = AutoEncoderLSTM(model_name, item_name, price_column, target_type, lookback_day,
                       prediction_day, time_interval, begin_pred_date, li_coding_size)

            market_prediction.data_load()
            market_prediction.preprocessing()
            market_prediction.create_model(lstm_unit)

            start = time.time() # 시작 시간 저장

            market_prediction.moving_transfer_learning(train_period, test_period, batch_size, learing_rate, begin_iteration,iteration)

            learing_time = time.time() - start
            print(f"time : {learing_time}") # 현재시각 - 시작시간 = 실행 시간

            market_prediction.evaluation(learing_time)