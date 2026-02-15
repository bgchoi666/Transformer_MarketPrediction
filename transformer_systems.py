# systems.py에 추가할 Transformer 클래스
# 기존 AutoEncoderLSTM, VanillaLSTM 클래스와 동일한 패턴으로 작성

import models
from models import training
import FnLearn
import math
from tensorflow import keras
import tensorflow as tf
import numpy as np


class TransformerPredictor(MarketPrediction):
    """
    Transformer 기반 주가 예측 클래스
    기존 LSTM 클래스들과 동일한 인터페이스 유지
    """
    def __init__(self, model_name, item_name, price_column, target_type, 
                 lookback_days, prediction_day, time_interval, begin_pred_date, 
                 d_model=256, num_heads=8, num_layers=4):
        """
        Args:
            model_name: 모델 이름 (예: 'VanillaTransformer')
            item_name: 종목명
            price_column: 가격 컬럼명
            target_type: 타겟 타입 ('trend_ratio', 'log_ratio' 등)
            lookback_days: 과거 참조 일수
            prediction_day: 미래 예측 일수
            time_interval: 시간 간격 (Transformer는 보통 1 사용)
            begin_pred_date: 예측 시작 날짜
            d_model: Transformer 모델 차원 (기본값: 256)
            num_heads: 어텐션 헤드 개수 (기본값: 8)
            num_layers: Transformer 레이어 개수 (기본값: 4)
        """
        super().__init__(model_name, item_name, price_column, target_type, 
                        lookback_days, prediction_day, time_interval, 
                        begin_pred_date, None)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = math.ceil(lookback_days / time_interval)
        
    def preprocessing(self):
        """데이터 전처리 - Transformer용으로 수정"""
        print(f"\nData Preprocessing for Transformer...\n")
        
        self.date = self.dataframe['date']
        self.price = self.dataframe[self.price_column]
        
        # 필요없는 컬럼 제거
        self.dataframe = FnLearn.remove_columns(self.dataframe, nan_thresh=70)
        self.dataframe = self.dataframe.interpolate(method='linear', limit_direction='both')
        
        self.input_size = self.dataframe.shape[1]
        
        # 예측값의 날짜
        self.output_date = self.date[self.lookback_days + self.prediction_day - 1:].reset_index(drop=True)
        self.date_idx = self.output_date.index[self.output_date == self.begin_pred_date][0]
        
        # Transformer용 입력 생성 (time_interval 대신 연속 데이터 사용)
        input3D = FnLearn.CreateInputOutput.transformer_input_data(
            self.dataframe.values, 
            self.lookback_days, 
            self.prediction_day,
            stride=self.time_interval
        )
        
        # 정규화
        self.input3D = FnLearn.TimeSeriesNormalization.std_scaler(input3D)
        
        # 타겟 생성
        self.ratio = FnLearn.CreateTarget.trend_ratio(self.price, self.prediction_day)
        
        # Future target 방식 사용 (현재에서 prediction_day 후 예측)
        self.target = FnLearn.CreateInputOutput.transformer_future_target(
            self.ratio, 
            self.lookback_days, 
            self.prediction_day,
            stride=self.time_interval
        )
        
        print(f"Input shape: {self.input3D.shape}")
        print(f"Target shape: {self.target.shape}")
        
    def create_model(self, dff=1024, dropout_rate=0.1, gpu=0):
        """
        Transformer 모델 생성
        
        Args:
            dff: 피드포워드 네트워크 차원
            dropout_rate: 드롭아웃 비율
            gpu: 사용할 GPU 번호
        """
        # transformer_models.py에서 임포트
        from transformer_models import VanillaTransformer
        
        self.predictor = VanillaTransformer(
            n_timestep=self.window_size,
            n_inputs=self.input_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dff=dff,
            dropout_rate=dropout_rate,
            gpu=gpu
        )
        
        print(f"\nTransformer Model Created:")
        print(f"  - Model dimension (d_model): {self.d_model}")
        print(f"  - Attention heads: {self.num_heads}")
        print(f"  - Encoder layers: {self.num_layers}")
        print(f"  - Parameters: {self.predictor.count_params():,}")
        
    def moving_transfer_learning(self, train_period, test_period, batch_size, 
                                 learning_rate, begin_iteration, iteration):
        """
        Moving window 방식으로 학습 및 예측
        
        Args:
            train_period: 학습 데이터 기간
            test_period: 테스트 데이터 기간
            batch_size: 배치 크기
            learning_rate: 학습률
            begin_iteration: 첫 번째 학습 반복 횟수
            iteration: 이후 학습 반복 횟수
        """
        self.batch_size = batch_size
        self.moving_count = int(round((len(self.input3D) - self.date_idx) / test_period))
        self.test_prediction = np.empty(shape=[0, 1])  # Future target이므로 (샘플, 1)
        
        for i in range(self.moving_count):
            # 학습/테스트 인덱스 계산
            train_begin_idx, train_end_idx, test_begin_idx, test_end_idx = train_test_idx(
                i, self.moving_count, self.date_idx, train_period, test_period, len(self.target)
            )
            
            print(f'{i}번 training index: {train_begin_idx} ~ {train_end_idx}', 
                  f'test index: {test_begin_idx} ~ {test_end_idx}')
            print(f'예측 구간: {self.output_date.iloc[test_begin_idx]} ~ {self.output_date.iloc[test_end_idx-1]}')
            print('#' * 100)
            
            # 데이터 분할
            train_input = self.input3D[train_begin_idx:train_end_idx]
            test_input = self.input3D[test_begin_idx:test_end_idx]
            train_output = self.target[train_begin_idx:train_end_idx]
            test_output = self.target[test_begin_idx:test_end_idx]
            
            # 학습 반복 횟수 결정
            if i == 0:
                n_iteration = begin_iteration
            else:
                n_iteration = iteration
            
            # 학습률 스케줄링
            global_step = tf.Variable(0, trainable=False)
            lr_decay = tf.compat.v1.train.exponential_decay(
                learning_rate, global_step, 
                train_input.shape[0] / batch_size * 5, 
                0.5, staircase=True
            )
            
            # Transformer 모델 학습
            self.predictor = training(
                self.predictor, 
                train_input, 
                train_output, 
                test_input, 
                test_output, 
                n_iteration, 
                batch_size, 
                batch_size, 
                lr_decay
            )
            
            # 예측값 추출
            batch_test_prediction = FnLearn.predict_batch_test(
                self.predictor, 
                test_input, 
                test_input.shape[0]
            )
            
            # Future target이므로 마지막 타임스텝 추출
            if len(batch_test_prediction.shape) == 3:
                # (batch, timestep, 1) → (batch, 1)
                batch_test_prediction = batch_test_prediction[:, -1, :]
            
            self.test_prediction = np.append(self.test_prediction, batch_test_prediction, axis=0)
    
    def evaluation(self, learning_time):
        """후처리 및 평가 - Future target용으로 수정"""
        test_price = self.price[self.date_idx + self.lookback_days - 1:]
        test_date = self.output_date[self.date_idx:].astype('str')
        
        # Future target을 위한 결과 처리
        self.result = FnLearn.ExperimentResult(
            self.test_prediction, 
            self.target[self.date_idx:],  # Future target
            self.lookback_days, 
            self.prediction_day
        )
        
        # 가격 변환
        self.result.convert_price(test_price)
        
        # 평가
        self.result.evaluation()
        
        # 결과 저장
        self.result.table(test_date)
        self.result.save_result(
            self.model_name, self.item_name, self.input_size, 
            self.time_interval, self.batch_size, self.target_type, 
            self.window_size, learning_time
        )
        self.result.save_visualization()
        self.result.save_model(self.predictor)
        self.result.upload_result(
            self.model_name, self.item_name, self.window_size, 
            self.time_interval, self.prediction_day, self.batch_size
        )


class HybridTransformerLSTMPredictor(MarketPrediction):
    """
    Transformer + LSTM 하이브리드 모델
    """
    def __init__(self, model_name, item_name, price_column, target_type,
                 lookback_days, prediction_day, time_interval, begin_pred_date,
                 d_model=256, num_heads=8, num_transformer_layers=2, lstm_units=256):
        super().__init__(model_name, item_name, price_column, target_type,
                        lookback_days, prediction_day, time_interval,
                        begin_pred_date, None)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.lstm_units = lstm_units
        self.window_size = math.ceil(lookback_days / time_interval)
    
    def preprocessing(self):
        """TransformerPredictor와 동일"""
        print(f"\nData Preprocessing for Hybrid Transformer-LSTM...\n")
        
        self.date = self.dataframe['date']
        self.price = self.dataframe[self.price_column]
        
        self.dataframe = FnLearn.remove_columns(self.dataframe, nan_thresh=70)
        self.dataframe = self.dataframe.interpolate(method='linear', limit_direction='both')
        
        self.input_size = self.dataframe.shape[1]
        
        self.output_date = self.date[self.lookback_days + self.prediction_day - 1:].reset_index(drop=True)
        self.date_idx = self.output_date.index[self.output_date == self.begin_pred_date][0]
        
        # Transformer용 입력 생성
        input3D = FnLearn.CreateInputOutput.transformer_input_data(
            self.dataframe.values,
            self.lookback_days,
            self.prediction_day,
            stride=self.time_interval
        )
        
        self.input3D = FnLearn.TimeSeriesNormalization.std_scaler(input3D)
        
        self.ratio = FnLearn.CreateTarget.trend_ratio(self.price, self.prediction_day)
        
        # Many-to-many 출력 (LSTM 부분을 위해)
        self.target = FnLearn.CreateInputOutput.many_to_many_output(
            self.ratio,
            self.lookback_days,
            self.time_interval
        )
        
        print(f"Input shape: {self.input3D.shape}")
        print(f"Target shape: {self.target.shape}")
    
    def create_model(self, dropout_rate=0.1, gpu=0):
        """하이브리드 모델 생성"""
        from transformer_models import HybridTransformerLSTM
        
        self.predictor = HybridTransformerLSTM(
            n_timestep=self.window_size,
            n_inputs=self.input_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_transformer_layers=self.num_transformer_layers,
            lstm_units=self.lstm_units,
            dropout_rate=dropout_rate,
            gpu=gpu
        )
        
        print(f"\nHybrid Transformer-LSTM Model Created:")
        print(f"  - Transformer dimension: {self.d_model}")
        print(f"  - Attention heads: {self.num_heads}")
        print(f"  - Transformer layers: {self.num_transformer_layers}")
        print(f"  - LSTM units: {self.lstm_units}")
        print(f"  - Parameters: {self.predictor.count_params():,}")
    
    def moving_transfer_learning(self, train_period, test_period, batch_size,
                                 learning_rate, begin_iteration, iteration):
        """VanillaLSTM과 동일한 방식으로 학습"""
        self.batch_size = batch_size
        self.moving_count = int(round((len(self.input3D) - self.date_idx) / test_period))
        self.test_prediction = np.empty(shape=[0, self.window_size, 1])
        
        for i in range(self.moving_count):
            train_begin_idx, train_end_idx, test_begin_idx, test_end_idx = train_test_idx(
                i, self.moving_count, self.date_idx, train_period, test_period, len(self.target)
            )
            
            print(f'{i}번 training index: {train_begin_idx} ~ {train_end_idx}',
                  f'test index: {test_begin_idx} ~ {test_end_idx}')
            print(f'예측 구간: {self.output_date.iloc[test_begin_idx]} ~ {self.output_date.iloc[test_end_idx-1]}')
            print('#' * 100)
            
            train_input = self.input3D[train_begin_idx:train_end_idx]
            test_input = self.input3D[test_begin_idx:test_end_idx]
            train_output = self.target[train_begin_idx:train_end_idx]
            test_output = self.target[test_begin_idx:test_end_idx]
            
            if i == 0:
                n_iteration = begin_iteration
            else:
                n_iteration = iteration
            
            global_step = tf.Variable(0, trainable=False)
            lr_decay = tf.compat.v1.train.exponential_decay(
                learning_rate, global_step,
                train_input.shape[0] / batch_size * 5,
                0.5, staircase=True
            )
            
            self.predictor = training(
                self.predictor,
                train_input,
                train_output,
                test_input,
                test_output,
                n_iteration,
                batch_size,
                batch_size,
                lr_decay
            )
            
            batch_test_prediction = FnLearn.predict_batch_test(
                self.predictor,
                test_input,
                test_input.shape[0]
            )
            
            self.test_prediction = np.append(self.test_prediction, batch_test_prediction, axis=0)


# Helper function (systems.py에 이미 있음)
def train_test_idx(times, moving_count, date_idx, train_period, test_period, size):
    """학습/테스트 인덱스 추출"""
    if times == 0:
        train_begin_idx = 0
    else:
        train_begin_idx = date_idx + test_period * times - train_period
    
    test_split_idx = date_idx + test_period * times
    train_end_idx = test_split_idx
    test_begin_idx = test_split_idx
    
    if times == moving_count - 1:
        test_end_idx = size
    else:
        test_end_idx = test_begin_idx + test_period
    
    return train_begin_idx, train_end_idx, test_begin_idx, test_end_idx
