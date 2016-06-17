
# coding: utf-8

# In[13]:

# get_ipython().magic('matplotlib inline')
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lstm_predictor import generate_data, lstm_model


# ## Libraries
# 
# - numpy: package for scientific computing 
# - pandas: data structures and data analysis tools
# - tensorflow: open source software library for machine intelligence
# - matplotlib: 2D plotting library
# 
# 
# - **learn**: Simplified interface for TensorFlow (mimicking Scikit Learn) for Deep Learning
# - mse: "mean squared error" as evaluation metric
# - **lstm_predictor**: our lstm class 
# 

# In[14]:

LOG_DIR = './ops_logs'
TIMESTEPS = 5
RNN_LAYERS = [{'steps': TIMESTEPS}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 100000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100


# ## Parameter definitions
# 
# - LOG_DIR: log file
# - TIMESTEPS: RNN time steps
# - RNN_LAYERS: RNN layer 정보
# - DENSE_LAYERS: DNN 크기 [10, 10]: Two dense layer with 10 hidden units
# - TRAINING_STEPS: 학습 스텝
# - BATCH_SIZE: 배치 학습 크기
# - PRINT_STEPS: 학습 과정 중간 출력 단계 (전체의 1% 해당하는 구간마다 출력)

# In[15]:

regressor = learn.TensorFlowEstimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS), 
                                      n_classes=0,
                                      verbose=1,  
                                      steps=TRAINING_STEPS, 
                                      optimizer='Adagrad',
                                      learning_rate=0.03, 
                                      batch_size=BATCH_SIZE)


# ## Create a regressor with TF Learn
# 
# : 예측을 위한 모델 생성. TF learn 라이브러리에 제공되는 TensorFlowEstimator를 사용.
# 
# **Parameters**: 
# 
# - model_fn: 학습 및 예측에 사용할 모델
# - n_classes: label에 해당하는 클래스 수 (0: prediction, 1이상: classification) 확인필요
# - verbose: 과정 출력
# - steps: 학습 스텝
# - optimizer: 최적화 기법 ("SGD", "Adam", "Adagrad")
# - learning_rate: learning rate
# - batch_size: batch size
# 
# 

# In[ ]:

X, y = generate_data(np.sin, np.linspace(0, 100, 10000), TIMESTEPS, seperate=False)
# create a lstm instance and validation monitor

validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                      every_n_steps=PRINT_STEPS,
                                                      early_stopping_rounds=1000)


# ## Generate a dataset
# 
# 1. generate_data: 학습에 사용될 데이터를 특정 함수를 이용하여 만듭니다.
#  - fct: 데이터를 생성할 함수
#  - x: 함수값을 관측할 위치
#  - time_steps: 관측(observation)
#  - seperate: check multimodal
# 1. ValidationMonitor: training 이후, validation 과정을 모니터링
#  - x
#  - y
#  - every_n_steps: 중간 출력
#  - early_stopping_rounds

# In[16]:

regressor.fit(X['train'], y['train'], monitors=[validation_monitor], logdir=LOG_DIR)


# ## Train and validation
# 
# - fit: training data를 이용해 학습, 모니터링과 로그를 통해 기록
# 
# 

# In[17]:

predicted = regressor.predict(X['test'])
mse = mean_squared_error(y['test'], predicted)
print ("Error: %f" % mse)


# ## Evaluate using test set
# 
# Evaluate our hypothesis using test set. The mean squared error (MSE) is used for the evaluation metric.
# 
# 

# In[18]:

plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(y['test'], label='test')
plt.legend(handles=[plot_predicted, plot_test])



# ## Plotting
# 
# Then, plot both predicted values and original values from test set.
