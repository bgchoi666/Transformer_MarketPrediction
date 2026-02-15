# FnLearn.py에 추가할 Transformer용 데이터 처리 메서드들

import numpy as np
import math


class CreateInputOutput(object):
    """
    기존 메서드들 (input_2d_data, many_to_many_output, many_to_one_output)은 그대로 유지하고
    아래 메서드들을 추가합니다.
    """
    
    # ==================== 기존 메서드들 (변경 없음) ====================
    @staticmethod
    def input_2d_data(data, n_timestep, future_day, time_interval=1):
        """기존 LSTM용 입력 데이터 생성 (변경 없음)"""
        data = data[:-future_day]
        input_2d_data = np.zeros((data.shape[0] + 1 - n_timestep, math.ceil(n_timestep / time_interval), data.shape[1]),
                                 dtype=np.float32)
        for i in range(len(input_2d_data)):
            input_2d_data[i] = data[i:n_timestep + i:time_interval]
        return input_2d_data
    
    @staticmethod
    def many_to_many_output(target, n_timestep, time_interval=1):
        """기존 LSTM용 출력 데이터 생성 (변경 없음)"""
        if type(target) != np.ndarray:
            target = np.array(target)
        output = np.zeros((target.shape[0] + 1 - n_timestep, math.ceil(n_timestep / time_interval), 1), dtype=np.float32)
        target = target.reshape(-1, 1)
        if time_interval == 1:
            for i in range(len(output)):
                output[i] = target[i:n_timestep + i:time_interval]
        else:
            for i in range(len(output)):
                temp = target[n_timestep + i - 1:i:-time_interval]
                output[i] = temp[::-1]
        return output
    
    @staticmethod
    def many_to_one_output(target, n_timestep):
        """기존 단일 출력 생성 (변경 없음)"""
        if type(target) != np.ndarray:
            target = np.array(target)
        target = target[n_timestep - 1:]
        output = np.expand_dims(target, axis=-1)
        return output
    
    
    # ==================== Transformer용 신규 메서드들 ====================
    
    @staticmethod
    def transformer_input_data(data, n_timestep, future_day, stride=1):
        """
        Transformer용 입력 데이터 생성
        time_interval 없이 연속된 시계열 데이터 사용
        
        Args:
            data: 원본 데이터 (2D array: [날짜 수, 특징 수])
            n_timestep: 시계열 길이 (예: 60일)
            future_day: 미래 예측 일수 (예: 20일)
            stride: 윈도우 이동 간격 (기본값 1 = 하루씩 이동)
        
        Returns:
            3D array: (샘플 수, n_timestep, 특징 수)
            
        예시:
            data.shape = (1000, 300)  # 1000일, 300개 특징
            n_timestep = 60
            future_day = 20
            stride = 1
            → output.shape = (920, 60, 300)  # 1000 - 60 - 20 + 1 = 921 샘플
        """
        # future_day만큼 앞의 데이터 제거 (예측할 데이터가 있어야 하므로)
        data = data[:-future_day]
        
        # 생성할 샘플 개수 계산
        n_samples = (data.shape[0] - n_timestep) // stride + 1
        
        # 출력 배열 초기화
        input_data = np.zeros((n_samples, n_timestep, data.shape[1]), dtype=np.float32)
        
        # 슬라이딩 윈도우로 데이터 생성
        for i in range(n_samples):
            start_idx = i * stride
            end_idx = start_idx + n_timestep
            input_data[i] = data[start_idx:end_idx]
        
        return input_data
    
    
    @staticmethod
    def transformer_output_sequence(target, n_timestep, future_day, stride=1, output_type='many_to_many'):
        """
        Transformer용 출력 데이터 생성
        
        Args:
            target: 타겟 값 (1D array: 수익률 등)
            n_timestep: 입력 시계열 길이
            future_day: 미래 예측 일수
            stride: 윈도우 이동 간격
            output_type: 'many_to_many' 또는 'many_to_one'
        
        Returns:
            many_to_many: (샘플 수, n_timestep, 1) - 각 시간스텝마다 예측값
            many_to_one: (샘플 수, 1) - 마지막 예측값만
            
        예시:
            target = [0.5, -0.3, 1.2, ...] (1000개)
            n_timestep = 60
            output_type = 'many_to_many'
            → output.shape = (921, 60, 1)
        """
        if type(target) != np.ndarray:
            target = np.array(target)
        
        n_samples = (len(target) - n_timestep - future_day + 1) // stride
        
        if output_type == 'many_to_many':
            # 각 시간스텝마다 예측값 생성 (LSTM과 동일)
            output = np.zeros((n_samples, n_timestep, 1), dtype=np.float32)
            
            for i in range(n_samples):
                start_idx = i * stride
                end_idx = start_idx + n_timestep
                output[i] = target[start_idx:end_idx].reshape(-1, 1)
            
        elif output_type == 'many_to_one':
            # 마지막 예측값만 생성
            output = np.zeros((n_samples, 1), dtype=np.float32)
            
            for i in range(n_samples):
                target_idx = i * stride + n_timestep - 1
                output[i] = target[target_idx]
        
        else:
            raise ValueError("output_type must be 'many_to_many' or 'many_to_one'")
        
        return output
    
    
    @staticmethod
    def transformer_future_target(target, n_timestep, future_day, stride=1):
        """
        미래 예측값을 타겟으로 하는 출력 생성
        현재 시점에서 future_day 이후의 값을 예측
        
        Args:
            target: 타겟 값 (수익률 등)
            n_timestep: 입력 시계열 길이
            future_day: 예측할 미래 일수
            stride: 윈도우 이동 간격
        
        Returns:
            (샘플 수, 1) - future_day 이후의 값
            
        예시:
            입력 윈도우: [Day 1 ~ Day 60]
            출력: Day 80의 수익률 (future_day=20)
        """
        if type(target) != np.ndarray:
            target = np.array(target)
        
        n_samples = (len(target) - n_timestep - future_day + 1) // stride
        output = np.zeros((n_samples, 1), dtype=np.float32)
        
        for i in range(n_samples):
            # 입력 윈도우의 마지막 시점 + future_day
            future_idx = i * stride + n_timestep + future_day - 1
            output[i] = target[future_idx]
        
        return output
    
    
    @staticmethod
    def transformer_multi_horizon_output(target, n_timestep, future_days_list, stride=1):
        """
        여러 미래 시점을 동시에 예측하는 출력 생성
        
        Args:
            target: 타겟 값
            n_timestep: 입력 시계열 길이
            future_days_list: 예측할 미래 일수 리스트 (예: [5, 10, 20])
            stride: 윈도우 이동 간격
        
        Returns:
            (샘플 수, len(future_days_list)) - 여러 미래 시점의 값
            
        예시:
            future_days_list = [5, 10, 20]
            → 5일 후, 10일 후, 20일 후를 동시에 예측
        """
        if type(target) != np.ndarray:
            target = np.array(target)
        
        max_future = max(future_days_list)
        n_samples = (len(target) - n_timestep - max_future + 1) // stride
        output = np.zeros((n_samples, len(future_days_list)), dtype=np.float32)
        
        for i in range(n_samples):
            for j, future_day in enumerate(future_days_list):
                future_idx = i * stride + n_timestep + future_day - 1
                output[i, j] = target[future_idx]
        
        return output
    
    
    @staticmethod
    def transformer_attention_mask(n_timestep, mask_future=True):
        """
        Transformer용 어텐션 마스크 생성
        
        Args:
            n_timestep: 시계열 길이
            mask_future: True면 미래 시점 마스킹 (causal mask)
        
        Returns:
            마스크 배열: (n_timestep, n_timestep)
            
        예시:
            mask_future=True일 때:
            [[1, 0, 0, 0],    # 1번째 시점은 자기 자신만 볼 수 있음
             [1, 1, 0, 0],    # 2번째 시점은 1~2번만 볼 수 있음
             [1, 1, 1, 0],    # 3번째 시점은 1~3번만 볼 수 있음
             [1, 1, 1, 1]]    # 4번째 시점은 1~4번 모두 볼 수 있음
        """
        if mask_future:
            # Causal mask (미래를 볼 수 없음)
            mask = np.tril(np.ones((n_timestep, n_timestep)))
        else:
            # No mask (모든 시점을 볼 수 있음)
            mask = np.ones((n_timestep, n_timestep))
        
        return mask.astype(np.float32)
    
    
    @staticmethod
    def create_transformer_dataset(data, target, n_timestep, future_day, 
                                   stride=1, output_type='future_target'):
        """
        Transformer 학습을 위한 전체 데이터셋 생성 (원스톱 함수)
        
        Args:
            data: 원본 특징 데이터
            target: 타겟 값 (수익률 등)
            n_timestep: 시계열 길이
            future_day: 예측할 미래 일수
            stride: 윈도우 이동 간격
            output_type: 'many_to_many', 'many_to_one', 'future_target'
        
        Returns:
            input_data, output_data
            
        사용 예시:
            X, y = CreateInputOutput.create_transformer_dataset(
                data=df_scaled, 
                target=ratio,
                n_timestep=60,
                future_day=20,
                output_type='future_target'
            )
        """
        # 입력 데이터 생성
        input_data = CreateInputOutput.transformer_input_data(
            data, n_timestep, future_day, stride
        )
        
        # 출력 데이터 생성
        if output_type == 'future_target':
            output_data = CreateInputOutput.transformer_future_target(
                target, n_timestep, future_day, stride
            )
        elif output_type in ['many_to_many', 'many_to_one']:
            output_data = CreateInputOutput.transformer_output_sequence(
                target, n_timestep, future_day, stride, output_type
            )
        else:
            raise ValueError("output_type must be 'many_to_many', 'many_to_one', or 'future_target'")
        
        return input_data, output_data


# ==================== 사용 예시 ====================
if __name__ == '__main__':
    """
    Transformer용 데이터 처리 예시
    """
    print("=" * 60)
    print("Transformer 데이터 처리 예시")
    print("=" * 60)
    
    # 가상 데이터 생성
    n_days = 1000
    n_features = 300
    np.random.seed(42)
    
    # 원본 데이터 (1000일, 300개 특징)
    data = np.random.randn(n_days, n_features).astype(np.float32)
    
    # 타겟 값 (수익률)
    target = np.random.randn(n_days).astype(np.float32) * 5  # -5% ~ +5% 범위
    
    print(f"\n원본 데이터 shape: {data.shape}")
    print(f"타겟 데이터 shape: {target.shape}")
    
    # 파라미터 설정
    n_timestep = 60
    future_day = 20
    stride = 1
    
    print(f"\n파라미터:")
    print(f"  - 시계열 길이 (n_timestep): {n_timestep}일")
    print(f"  - 예측 일수 (future_day): {future_day}일")
    print(f"  - 윈도우 이동 간격 (stride): {stride}일")
    
    # 1. 기본 입력 데이터 생성
    print("\n" + "=" * 60)
    print("1. Transformer 입력 데이터 생성")
    print("=" * 60)
    input_data = CreateInputOutput.transformer_input_data(data, n_timestep, future_day, stride)
    print(f"입력 데이터 shape: {input_data.shape}")
    print(f"예상 샘플 수: {n_days - n_timestep - future_day + 1} = {1000 - 60 - 20 + 1} = 921")
    
    # 2. Many-to-Many 출력
    print("\n" + "=" * 60)
    print("2. Many-to-Many 출력 (각 시간스텝마다 예측)")
    print("=" * 60)
    output_m2m = CreateInputOutput.transformer_output_sequence(
        target, n_timestep, future_day, stride, 'many_to_many'
    )
    print(f"출력 데이터 shape: {output_m2m.shape}")
    print(f"설명: 60개 시간스텝 각각에 대한 예측값")
    
    # 3. Many-to-One 출력
    print("\n" + "=" * 60)
    print("3. Many-to-One 출력 (마지막 예측값만)")
    print("=" * 60)
    output_m2o = CreateInputOutput.transformer_output_sequence(
        target, n_timestep, future_day, stride, 'many_to_one'
    )
    print(f"출력 데이터 shape: {output_m2o.shape}")
    print(f"설명: 마지막 시간스텝의 예측값만")
    
    # 4. Future Target 출력 (가장 일반적)
    print("\n" + "=" * 60)
    print("4. Future Target 출력 (20일 후 예측)")
    print("=" * 60)
    output_future = CreateInputOutput.transformer_future_target(
        target, n_timestep, future_day, stride
    )
    print(f"출력 데이터 shape: {output_future.shape}")
    print(f"설명: 현재 시점에서 {future_day}일 후의 값 예측")
    
    # 5. Multi-Horizon 출력
    print("\n" + "=" * 60)
    print("5. Multi-Horizon 출력 (여러 시점 동시 예측)")
    print("=" * 60)
    future_days_list = [5, 10, 20]
    output_multi = CreateInputOutput.transformer_multi_horizon_output(
        target, n_timestep, future_days_list, stride
    )
    print(f"출력 데이터 shape: {output_multi.shape}")
    print(f"설명: {future_days_list}일 후를 동시에 예측")
    
    # 6. Attention Mask
    print("\n" + "=" * 60)
    print("6. Attention Mask 생성")
    print("=" * 60)
    mask_causal = CreateInputOutput.transformer_attention_mask(n_timestep, mask_future=True)
    mask_full = CreateInputOutput.transformer_attention_mask(n_timestep, mask_future=False)
    print(f"Causal Mask shape: {mask_causal.shape}")
    print(f"Causal Mask (처음 5x5):\n{mask_causal[:5, :5]}")
    print(f"\nFull Mask: 모든 값이 1 (모든 시점 참조 가능)")
    
    # 7. 원스톱 데이터셋 생성
    print("\n" + "=" * 60)
    print("7. 원스톱 데이터셋 생성")
    print("=" * 60)
    X, y = CreateInputOutput.create_transformer_dataset(
        data, target, n_timestep, future_day, 
        stride=1, output_type='future_target'
    )
    print(f"입력 X shape: {X.shape}")
    print(f"출력 y shape: {y.shape}")
    print(f"준비 완료! 바로 모델 학습에 사용 가능")
    
    # 8. 실제 사용 시나리오
    print("\n" + "=" * 60)
    print("8. 실제 사용 시나리오")
    print("=" * 60)
    print("""
    # 데이터 로드 및 전처리
    dataframe = FnLearn.DataLoad.fndata_dataframe('삼성전자', '수정주가(원)', True)
    dataframe = FnLearn.remove_columns(dataframe, nan_thresh=70)
    dataframe = dataframe.interpolate(method='linear', limit_direction='both')
    
    # 가격과 수익률 추출
    price = dataframe['수정주가(원)']
    ratio = FnLearn.CreateTarget.trend_ratio(price, future_day=20)
    
    # 정규화
    input3D = FnLearn.CreateInputOutput.transformer_input_data(dataframe.values, 60, 20)
    input3D_normalized = FnLearn.TimeSeriesNormalization.std_scaler(input3D)
    
    # 타겟 생성
    target = FnLearn.CreateInputOutput.transformer_future_target(ratio, 60, 20)
    
    # 또는 원스톱으로
    X, y = FnLearn.CreateInputOutput.create_transformer_dataset(
        dataframe.values, ratio, 60, 20, output_type='future_target'
    )
    
    # Transformer 모델 생성
    from transformer_models import VanillaTransformer
    model = VanillaTransformer(
        n_timestep=60,
        n_inputs=X.shape[2],
        d_model=256,
        num_heads=8,
        num_layers=4
    )
    
    # 학습
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mse'
    )
    model.fit(X, y, epochs=100, batch_size=32)
    """)
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
