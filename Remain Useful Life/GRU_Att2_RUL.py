import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

"""
Docker를 통해 접속한 서버의 GPU 자원을 할당받기 위한 구문
"""
# GPU 수동 할당
gpus = tf.config.experimental.list_physical_devices('GPU')

# GPU 수동 할당
# 0번 GPU 20000mb 사용
tf.config.experimental.set_virtual_device_configuration(gpus[0],
[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20000)])

# 데이터 불러오기
# 데이터는 각각 수집해야 된다.
data = pd.read_csv("./RUL/Data/Gear/x0_gear.csv")

# 데이터를 넘파이 배열로 변환
# 외부 파일을 불러오는 방법은 넘파이도 제공하지만 판다스가 속도가 빠르다.
data_np = np.array(data0.values, np.float32())

# 일정 간격으로 평균을 나누어 계산
# 해당 부분은 데이터의 불규칙적인 움직임을 평균으로 예측하기 위한 전처리 과정
data_mean = []
mean_step = 1
for i in range(data_np.size//mean_step):
    data_mean.append(np.mean(data_np[i*mean_step:(i+1)*mean_step, 1]))

# 데이터 정규화
data_set = data_mean
data_set = np.array(data_set, np.float32)
data_set = data_set.reshape((data_set.size, -1))
data_set = MinMaxScaler().fit_transform(data_set)

# 학습 데이터 만들기 - HI그래프 예측
# 시퀸스 생성 메소드 선언
# GRU 모델을 사용하기 위해서는 입력 데이터가 3차원 데이터이어야 한다.
def create_sequence(data, timestep, pred, input_dim):
    # 빈 리스트 선언
    X_data, y_data = [], []

    # 반복문을 통해 데이터를 저장
    # timestep 개수만큼 input_dim의 간격으로 리스트에 저장
    for i in range(len(data)-timestep-pred+1):
        temp = []
        for j in range(i, i+timestep, input_dim):
            temp.append([data[j]])
        X_data.append(temp)

        # 예측값 저장
        y_data.append(data[i+timestep:i+timestep+pred])
    return X_data, y_data

# 타임스텝 , 출력갯수, 속성 수  정의 후 시퀸스와 예측값 저장
timestep = 10
pred = 1
dim = 1
X_data, y_data = create_sequence(np.ravel(data_set, order="C"), timestep, pred, dim)

# 학습데이터와 테스트 데이터로 구분하는 메소드 선언
def train_test_split(X_data, y_data, test_size):
    X_train = X_data[0:int(len(X_data)*(1-test_size))]
    X_test = X_data[int(len(X_data)*(1-test_size)):]
    y_train = y_data[0:int(len(y_data)*(1-test_size))]
    y_test = y_data[int(len(y_data)*(1-test_size)):]
    return X_train, X_test, y_train, y_test

# 학습데이터와 테스트 데이터로 구분(본 예제에서는 0.9비율로 구분)
# 넘파이 배열로 변환 및 데이터 저장
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 0.9)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# 어텐션 메커니즘 모델 선언
# GRU와 호환되도록 3차원 블록을 이용하는 방식
def attention_3d_block(inputs):
    
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    
    a = tf.keras.layers.Permute((2, 1))(inputs) # same transpose
    #a = tf.keras.layers.Reshape((input_dim, TIME_STEPS))(a) 
    # this line is not useful. It's just to know which dimension is what.
    a = tf.keras.layers.Dense(timestep, activation='softmax')(a)
    
    a_probs = tf.keras.layers.Permute((2, 1), name='attention_vec')(a)
    
    output_attention_mul  = tf.keras.layers.multiply([inputs, a_probs])
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

#  어텐션 GRU 모델 설계
def model_attention_applied_after_gru():
    # GRU 구문
    inputs        = tf.keras.Input(shape=(timestep, dim,))
    gru_units    = 32    
    gru_out      = tf.keras.layers.GRU(gru_units, return_sequences=True)(inputs)
    
    # 어텐션 메커니즘 구문
    attention_mul = attention_3d_block(gru_out)
    attention_mul = tf.keras.layers.Flatten()(attention_mul)
    
    # 시그모이드 출력
    output        = tf.keras.layers.Dense(1, activation='sigmoid')(attention_mul)    
    model         = tf.keras.Model(inputs=[inputs], outputs=output)
    
    return model

# 모델 컴파일
modelman = model_attention_applied_after_gru()
modelman.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
try:
    with tf.device('/device:GPU:0'):
        modelman.fit([X_train], y_train, epochs=100, batch_size=128, validation_split=0.005)
        layer_outputs    = [layer.output for layer in modelman.layers if layer.name == 'attention_vec']
        activation_model = tf.keras.models.Model(inputs=modelman.input, outputs=layer_outputs)
except RuntimeError as e:
    print(e)

# 예측 데이터 저장
y_hat = modelman.predict(X_test)

# 모델 저장
modelman.save("GRU_Att.h5")

# 그래프를 통한 각 결과 비교
plt.plot(y_hat, "r-")
plt.plot(y_test, "b-")
plt.show()

"""
지금부터의 과정은 HI그래프를 예측하는 모델을 활용하여 그래프의 추이를 예측한다.
예측한 결과를 이용하여 RUL그래프를 예측하는 입력데이터로 활용한다.
아래 과정은 HI 그래프 예측으로부터 RUL 그래프를 예측하는 과정이다.
"""
# 모델을 이용하여 예측 데이터 부풀리기
# HI그래프의 현 시점이 아니라 추후 시점을 그리기 위한 과정
list_hat =  modelman.predict(X_test[-timestep:].reshape(timestep,timestep,1))
print(list_hat)
for i in range(10000):
    list_hat = np.append(list_hat, modelman.predict(list_hat[-timestep:].reshape(1,timestep,1)), axis=0)

# 기존 데이터와 합치기
HI_data = np.concatenate((y_hat, list_hat))

# 결과 출력
plt.plot(HI_data)

# 평균을 이용하여 데이터 전처리
# 그래프의 추이를 명확하게 하기위한 과정
ave = []
m = 3000
for i in range(len(HI_data)):
    ave.append(HI_data[i:m+i].mean())

# 건강상태 그래프(HI 그래프)
plt.plot(ave[25500:27000])

# RUL 그래프 라벨링 데이터 생성
max_time = 100000
X_data_len = 16000-13000 + 27000 - 25500
#y_data = [i for i in range(0.0, X_data_len/(max_time*60*60), 1/(max_time*60*60))]
y_data_temp = np.arange(1, 0 , -1/(max_time)/30)
temp = []
for i in range(len(y_data_temp)):   
    temp.append(y_data_temp[i])
y_data = temp[2200000:X_data_len+2200000]

# RUL 확인용 그래프 출력
plt.plot(y_data)

# RUL 예측을 위한 학습 데이터와 테스트 데이터 구분
mid = 16000
last = 27000
rul_X_train = np.array(ave[13000:mid], np.float32)
rul_X_test = np.array(ave[25500:last], np.float32)
rul_y_train = np.array(y_data[0:mid-13000], np.float32)
rul_y_test = np.array(y_data[mid-13000:], np.float32)

# RUL 예측을 위한 선형 모델 생성
rul_model = tf.keras.Sequential()

# 모델 층 구성
rul_model.add(tf.keras.layers.Dense(64, input_dim=1, activation="relu"))
rul_model.add(tf.keras.layers.Dense(1, input_dim=1))

# 모델 컴파일
rul_model.compile(optimizer="rmsprop", loss="mse", metrics=['accuracy'])

# 학습
hist = rul_model.fit(rul_X_train, rul_y_train, epochs=100, batch_size=128)

# 예측 데이터 저장
y_hat_rul = rul_model.predict(rul_X_test)

# 그래프를 통한 각 결과 비교
plt.plot(y_hat_rul-0.003, "r-")
plt.plot(rul_y_test, "b-")
plt.show()