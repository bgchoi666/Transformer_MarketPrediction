"""
Transformer 모델을 사용한 주가 예측 메인 스크립트
기존 mains.py와 동일한 구조로 작성
"""

from systems import *
from transformer_systems import TransformerPredictor, HybridTransformerLSTMPredictor
import math
import time
import os
import sys

# GPU 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'  # 사용할 GPU 번호
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# ==================== 실험 설정 ====================

# 종목과 종가 변수 리스트
li_item_name = ['메리츠화재']
li_price_column = ['수정주가(원)']

# 데이터 전처리 파라미터
nan_thresh = 70
fill_na = 'interpolation'
target_type = 'trend_ratio'

# 시계열 파라미터
lookback_days = [60, 120]  # Transformer는 더 긴 시퀀스도 효과적으로 처리 가능
prediction_day = 20
time_intervals = [1]  # Transformer는 보통 time_interval=1 사용

# Transformer 모델 파라미터
d_model = 256           # Transformer 내부 차원
num_heads = 8           # Multi-head attention 헤드 개수
num_layers = 4          # Transformer 레이어 개수
dff = 1024             # Feed-forward 네트워크 차원
dropout_rate = 0.1     # 드롭아웃 비율

# 학습과정 파라미터
test_period = 20
train_period = 500
batch_size = 32
learning_rate = 0.0005
begin_pred_date = '2015-01-02'  # 첫 예측 날짜
begin_iteration = 1000
iteration = 500


# ==================== 실험 1: VanillaTransformer ====================
print("=" * 100)
print("실험 1: Vanilla Transformer 모델")
print("=" * 100)

for item_name, price_column in zip(li_item_name, li_price_column):
    for lookback_day in lookback_days:
        for time_interval in time_intervals:
            print(f"\n종목명: {item_name}       타겟: {price_column}")
            print(f"Lookback: {lookback_day}일, Interval: {time_interval}일")
            
            # Transformer 모델 초기화
            market_prediction = TransformerPredictor(
                model_name='VanillaTransformer',
                item_name=item_name,
                price_column=price_column,
                target_type=target_type,
                lookback_days=lookback_day,
                prediction_day=prediction_day,
                time_interval=time_interval,
                begin_pred_date=begin_pred_date,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers
            )
            
            # 데이터 로드 및 전처리
            market_prediction.data_load()
            market_prediction.preprocessing()
            
            # 모델 생성
            market_prediction.create_model(
                dff=dff,
                dropout_rate=dropout_rate,
                gpu=0
            )
            
            # 학습 시작
            start = time.time()
            
            market_prediction.moving_transfer_learning(
                train_period=train_period,
                test_period=test_period,
                batch_size=batch_size,
                learning_rate=learning_rate,
                begin_iteration=begin_iteration,
                iteration=iteration
            )
            
            learning_time = time.time() - start
            print(f"\n학습 완료! 소요 시간: {learning_time:.2f}초 ({learning_time/60:.2f}분)")
            
            # 평가 및 결과 저장
            market_prediction.evaluation(learning_time)
            
            print("\n" + "=" * 100 + "\n")


# ==================== 실험 2: Hybrid Transformer-LSTM ====================
print("=" * 100)
print("실험 2: Hybrid Transformer-LSTM 모델")
print("=" * 100)

# 하이브리드 모델 파라미터
num_transformer_layers = 2  # Transformer 레이어 수 (적게)
lstm_units = 256           # LSTM 유닛 수

for item_name, price_column in zip(li_item_name, li_price_column):
    for lookback_day in lookback_days:
        for time_interval in time_intervals:
            print(f"\n종목명: {item_name}       타겟: {price_column}")
            print(f"Lookback: {lookback_day}일, Interval: {time_interval}일")
            
            # 하이브리드 모델 초기화
            market_prediction = HybridTransformerLSTMPredictor(
                model_name='HybridTransformerLSTM',
                item_name=item_name,
                price_column=price_column,
                target_type=target_type,
                lookback_days=lookback_day,
                prediction_day=prediction_day,
                time_interval=time_interval,
                begin_pred_date=begin_pred_date,
                d_model=d_model,
                num_heads=num_heads,
                num_transformer_layers=num_transformer_layers,
                lstm_units=lstm_units
            )
            
            # 데이터 로드 및 전처리
            market_prediction.data_load()
            market_prediction.preprocessing()
            
            # 모델 생성
            market_prediction.create_model(
                dropout_rate=dropout_rate,
                gpu=0
            )
            
            # 학습 시작
            start = time.time()
            
            market_prediction.moving_transfer_learning(
                train_period=train_period,
                test_period=test_period,
                batch_size=batch_size,
                learning_rate=learning_rate,
                begin_iteration=begin_iteration,
                iteration=iteration
            )
            
            learning_time = time.time() - start
            print(f"\n학습 완료! 소요 시간: {learning_time:.2f}초 ({learning_time/60:.2f}분)")
            
            # 평가 및 결과 저장
            market_prediction.evaluation(learning_time)
            
            print("\n" + "=" * 100 + "\n")


# ==================== 실험 3: 여러 모델 비교 ====================
print("=" * 100)
print("실험 3: Transformer vs LSTM 성능 비교")
print("=" * 100)

# 비교할 모델 리스트
models_to_compare = [
    {
        'name': 'VanillaTransformer',
        'class': TransformerPredictor,
        'params': {
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 4
        }
    },
    {
        'name': 'HybridTransformerLSTM',
        'class': HybridTransformerLSTMPredictor,
        'params': {
            'd_model': 256,
            'num_heads': 8,
            'num_transformer_layers': 2,
            'lstm_units': 256
        }
    }
]

# 고정 파라미터
fixed_lookback = 60
fixed_interval = 1

for model_config in models_to_compare:
    print(f"\n{'='*50}")
    print(f"모델: {model_config['name']}")
    print(f"{'='*50}")
    
    for item_name, price_column in zip(li_item_name, li_price_column):
        # 모델 초기화
        market_prediction = model_config['class'](
            model_name=model_config['name'],
            item_name=item_name,
            price_column=price_column,
            target_type=target_type,
            lookback_days=fixed_lookback,
            prediction_day=prediction_day,
            time_interval=fixed_interval,
            begin_pred_date=begin_pred_date,
            **model_config['params']
        )
        
        # 데이터 로드 및 전처리
        market_prediction.data_load()
        market_prediction.preprocessing()
        
        # 모델 생성
        if model_config['name'] == 'VanillaTransformer':
            market_prediction.create_model(dff=dff, dropout_rate=dropout_rate, gpu=0)
        else:
            market_prediction.create_model(dropout_rate=dropout_rate, gpu=0)
        
        # 학습
        start = time.time()
        market_prediction.moving_transfer_learning(
            train_period=train_period,
            test_period=test_period,
            batch_size=batch_size,
            learning_rate=learning_rate,
            begin_iteration=begin_iteration,
            iteration=iteration
        )
        learning_time = time.time() - start
        
        # 평가
        market_prediction.evaluation(learning_time)
        
        print(f"✓ {model_config['name']} 완료 (소요 시간: {learning_time/60:.2f}분)")


print("\n" + "=" * 100)
print("모든 실험 완료!")
print("=" * 100)
print(f"\n결과 확인:")
print(f"  - 엑셀 파일: ./experiment/result_excel/")
print(f"  - 그래프: ./experiment/result_pyplot/")
print(f"  - 모델: ./experiment/models/")
