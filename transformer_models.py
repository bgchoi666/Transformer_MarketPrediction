import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np


# ==================== Transformer Components ====================

class PositionalEncoding(layers.Layer):
    """위치 인코딩 레이어 - 시계열 데이터의 순서 정보를 추가"""
    def __init__(self, max_sequence_length=300, d_model=256):
        super(PositionalEncoding, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        
        # 위치 인덱스 생성
        position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(self.d_model, dtype=tf.float32)[tf.newaxis, :]
        
        # 각도 계산
        angle_rads = self.get_angles(position, i, self.d_model)
        
        # sin을 짝수 인덱스에, cos를 홀수 인덱스에 적용
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        
        # 교차로 합치기
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return inputs + pos_encoding


class MultiHeadAttention(layers.Layer):
    """멀티 헤드 어텐션 레이어"""
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.depth = d_model // num_heads
        
        # Query, Key, Value를 위한 Dense 레이어
        self.wq = layers.Dense(d_model, kernel_initializer='glorot_uniform')
        self.wk = layers.Dense(d_model, kernel_initializer='glorot_uniform')
        self.wv = layers.Dense(d_model, kernel_initializer='glorot_uniform')
        
        self.dense = layers.Dense(d_model, kernel_initializer='glorot_uniform')
        self.dropout = layers.Dropout(dropout_rate)
        
    def split_heads(self, x, batch_size):
        """마지막 차원을 (num_heads, depth)로 분할"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, query, key, value, mask=None, training=False):
        batch_size = tf.shape(query)[0]
        
        # Query, Key, Value 생성
        q = self.wq(query)  # (batch_size, seq_len, d_model)
        k = self.wk(key)
        v = self.wv(value)
        
        # 헤드 분할
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled Dot-Product Attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # 스케일링
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # 마스크 적용 (선택적)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax로 어텐션 가중치 계산
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Value에 가중치 적용
        output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth)
        
        # 헤드 병합
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        
        # 최종 선형 변환
        output = self.dense(concat_attention)
        
        return output, attention_weights


class FeedForwardNetwork(layers.Layer):
    """피드포워드 네트워크"""
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = layers.Dense(dff, activation='relu', kernel_initializer='he_normal')
        self.dense2 = layers.Dense(d_model, kernel_initializer='he_normal')
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x


class TransformerEncoderLayer(layers.Layer):
    """Transformer 인코더 레이어"""
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ffn = FeedForwardNetwork(d_model, dff, dropout_rate)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, mask=None, training=False):
        # 멀티 헤드 어텐션
        attn_output, _ = self.mha(x, x, x, mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Residual connection
        
        # 피드포워드 네트워크
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection
        
        return out2


# ==================== Transformer Models ====================

def VanillaTransformer(n_timestep, n_inputs, d_model=256, num_heads=8, num_layers=4, 
                       dff=1024, dropout_rate=0.1, gpu=0):
    """
    기본 Transformer 모델
    
    Args:
        n_timestep: 시계열 길이 (예: 60일)
        n_inputs: 입력 특징 개수
        d_model: 모델 차원 (256)
        num_heads: 어텐션 헤드 개수 (8)
        num_layers: 인코더 레이어 개수 (4)
        dff: 피드포워드 네트워크 차원 (1024)
        dropout_rate: 드롭아웃 비율 (0.1)
        gpu: 사용할 GPU 번호
    
    Returns:
        Keras Model
    """
    with tf.device('/gpu:' + str(gpu)):
        # 입력
        inputs = keras.Input(shape=(n_timestep, n_inputs))
        
        # 입력 차원을 d_model로 변환
        x = layers.Dense(d_model, kernel_initializer='he_normal')(inputs)
        
        # 위치 인코딩 추가
        pos_encoding = PositionalEncoding(max_sequence_length=n_timestep, d_model=d_model)
        x = pos_encoding(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Transformer 인코더 레이어들
        for _ in range(num_layers):
            x = TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)(x)
        
        # 출력 레이어 - 각 시간스텝마다 예측값 생성
        outputs = layers.Dense(1, kernel_initializer='he_normal')(x)
        
        return Model(inputs=inputs, outputs=outputs, name='VanillaTransformer')


def TransformerRegressor(n_timestep, n_inputs, d_model=256, num_heads=8, num_layers=4,
                         dff=1024, dropout_rate=0.1, gpu=0):
    """
    시계열 회귀를 위한 Transformer 모델 (GlobalAveragePooling 사용)
    마지막 출력만 사용하는 경우에 적합
    
    Args:
        n_timestep: 시계열 길이
        n_inputs: 입력 특징 개수
        d_model: 모델 차원
        num_heads: 어텐션 헤드 개수
        num_layers: 인코더 레이어 개수
        dff: 피드포워드 네트워크 차원
        dropout_rate: 드롭아웃 비율
        gpu: 사용할 GPU 번호
    
    Returns:
        Keras Model
    """
    with tf.device('/gpu:' + str(gpu)):
        inputs = keras.Input(shape=(n_timestep, n_inputs))
        
        # 입력 임베딩
        x = layers.Dense(d_model, kernel_initializer='he_normal')(inputs)
        
        # 위치 인코딩
        pos_encoding = PositionalEncoding(max_sequence_length=n_timestep, d_model=d_model)
        x = pos_encoding(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Transformer 인코더
        for _ in range(num_layers):
            x = TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)(x)
        
        # Global Average Pooling - 모든 시간스텝의 평균
        x = layers.GlobalAveragePooling1D()(x)
        
        # 출력 레이어
        x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, kernel_initializer='he_normal')(x)
        
        return Model(inputs=inputs, outputs=outputs, name='TransformerRegressor')


def AutoEncoderTransformer(n_timestep, n_inputs, coding_size=256, num_heads=8, 
                           num_encoder_layers=3, num_decoder_layers=3,
                           dff=1024, dropout_rate=0.1, gpu=0):
    """
    AutoEncoder 구조의 Transformer
    입력 데이터를 압축 후 복원
    
    Args:
        n_timestep: 시계열 길이
        n_inputs: 입력 특징 개수
        coding_size: 압축 차원
        num_heads: 어텐션 헤드 개수
        num_encoder_layers: 인코더 레이어 개수
        num_decoder_layers: 디코더 레이어 개수
        dff: 피드포워드 네트워크 차원
        dropout_rate: 드롭아웃 비율
        gpu: 사용할 GPU 번호
    
    Returns:
        Keras Model
    """
    with tf.device('/gpu:' + str(gpu)):
        inputs = keras.Input(shape=(n_timestep, n_inputs))
        
        # === Encoder ===
        x = layers.Dense(512, kernel_initializer='he_normal')(inputs)
        pos_encoding = PositionalEncoding(max_sequence_length=n_timestep, d_model=512)
        x = pos_encoding(x)
        x = layers.Dropout(dropout_rate)(x)
        
        for _ in range(num_encoder_layers):
            x = TransformerEncoderLayer(512, num_heads, dff, dropout_rate)(x)
        
        # 압축 레이어 (Bottleneck)
        encoded = layers.Dense(coding_size, kernel_initializer='he_normal', name='HiddenLayer')(x)
        
        # === Decoder ===
        x = layers.Dense(512, kernel_initializer='he_normal')(encoded)
        
        for _ in range(num_decoder_layers):
            x = TransformerEncoderLayer(512, num_heads, dff, dropout_rate)(x)
        
        # 복원 레이어
        outputs = layers.Dense(n_inputs, kernel_initializer='he_normal')(x)
        
        return Model(inputs=inputs, outputs=outputs, name='AutoEncoderTransformer')


def HybridTransformerLSTM(n_timestep, n_inputs, d_model=256, num_heads=8, 
                          num_transformer_layers=2, lstm_units=256,
                          dropout_rate=0.1, gpu=0):
    """
    Transformer + LSTM 하이브리드 모델
    Transformer로 장기 의존성을 포착하고 LSTM으로 시계열 패턴 학습
    
    Args:
        n_timestep: 시계열 길이
        n_inputs: 입력 특징 개수
        d_model: Transformer 모델 차원
        num_heads: 어텐션 헤드 개수
        num_transformer_layers: Transformer 레이어 개수
        lstm_units: LSTM 유닛 개수
        dropout_rate: 드롭아웃 비율
        gpu: 사용할 GPU 번호
    
    Returns:
        Keras Model
    """
    with tf.device('/gpu:' + str(gpu)):
        inputs = keras.Input(shape=(n_timestep, n_inputs))
        
        # Transformer 레이어
        x = layers.Dense(d_model, kernel_initializer='he_normal')(inputs)
        pos_encoding = PositionalEncoding(max_sequence_length=n_timestep, d_model=d_model)
        x = pos_encoding(x)
        x = layers.Dropout(dropout_rate)(x)
        
        for _ in range(num_transformer_layers):
            x = TransformerEncoderLayer(d_model, num_heads, d_model * 4, dropout_rate)(x)
        
        # LSTM 레이어
        x = layers.LSTM(lstm_units, return_sequences=True, 
                       kernel_initializer='he_normal',
                       kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.LSTM(lstm_units, return_sequences=True,
                       kernel_initializer='he_normal',
                       kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # 출력 레이어
        outputs = layers.Dense(1, kernel_initializer='he_normal')(x)
        
        return Model(inputs=inputs, outputs=outputs, name='HybridTransformerLSTM')


def ResTransformer(n_timestep, n_inputs, d_model=256, num_heads=8, num_layers=6,
                   dff=1024, dropout_rate=0.1, gpu=0):
    """
    깊은 Residual Connection을 가진 Transformer
    더 깊은 네트워크 학습 가능
    
    Args:
        n_timestep: 시계열 길이
        n_inputs: 입력 특징 개수
        d_model: 모델 차원
        num_heads: 어텐션 헤드 개수
        num_layers: 인코더 레이어 개수
        dff: 피드포워드 네트워크 차원
        dropout_rate: 드롭아웃 비율
        gpu: 사용할 GPU 번호
    
    Returns:
        Keras Model
    """
    with tf.device('/gpu:' + str(gpu)):
        inputs = keras.Input(shape=(n_timestep, n_inputs))
        
        # 입력 임베딩
        x = layers.Dense(d_model, kernel_initializer='he_normal')(inputs)
        initial = x  # Residual connection을 위해 저장
        
        # 위치 인코딩
        pos_encoding = PositionalEncoding(max_sequence_length=n_timestep, d_model=d_model)
        x = pos_encoding(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Transformer 레이어들 with skip connections
        for i in range(num_layers):
            x_prev = x
            x = TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)(x)
            
            # 매 2개 레이어마다 skip connection 추가
            if i % 2 == 1:
                x = layers.Add()([x, x_prev])
        
        # 초기 입력과의 residual connection
        x = layers.Add()([x, initial])
        
        # 출력 레이어
        outputs = layers.Dense(1, kernel_initializer='he_normal', name='logits')(x)
        
        return Model(inputs=inputs, outputs=outputs, name='ResTransformer')


# ==================== 사용 예시 ====================
if __name__ == '__main__':
    """
    사용 예시:
    
    # 1. 기본 Transformer
    model = VanillaTransformer(n_timestep=60, n_inputs=300, d_model=256, num_heads=8, num_layers=4)
    
    # 2. AutoEncoder Transformer
    autoencoder = AutoEncoderTransformer(n_timestep=60, n_inputs=300, coding_size=256)
    
    # 3. Hybrid Transformer-LSTM
    hybrid_model = HybridTransformerLSTM(n_timestep=60, n_inputs=300, d_model=256, lstm_units=256)
    
    # 4. Residual Transformer
    res_model = ResTransformer(n_timestep=60, n_inputs=300, d_model=256, num_layers=6)
    """
    
    # 테스트
    print("Testing Transformer Models...")
    
    # 테스트 데이터
    batch_size = 32
    n_timestep = 60
    n_inputs = 300
    
    test_input = tf.random.normal((batch_size, n_timestep, n_inputs))
    
    # 1. VanillaTransformer 테스트
    print("\n1. VanillaTransformer")
    model1 = VanillaTransformer(n_timestep, n_inputs, d_model=256, num_heads=8, num_layers=4, gpu=0)
    output1 = model1(test_input, training=False)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output1.shape}")
    print(f"Parameters: {model1.count_params():,}")
    
    # 2. AutoEncoderTransformer 테스트
    print("\n2. AutoEncoderTransformer")
    model2 = AutoEncoderTransformer(n_timestep, n_inputs, coding_size=256, gpu=0)
    output2 = model2(test_input, training=False)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output2.shape}")
    print(f"Parameters: {model2.count_params():,}")
    
    # 3. HybridTransformerLSTM 테스트
    print("\n3. HybridTransformerLSTM")
    model3 = HybridTransformerLSTM(n_timestep, n_inputs, d_model=256, lstm_units=256, gpu=0)
    output3 = model3(test_input, training=False)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output3.shape}")
    print(f"Parameters: {model3.count_params():,}")
    
    print("\nAll models tested successfully!")
