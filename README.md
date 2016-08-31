# TensorFlow Tutorial for Time Series Prediction

This tutorial is designed to easily learn TensorFlow for time series prediction. 
Each tutorial subject includes both code and notebook with descriptions.

## Tutorial Index

#### Time series prediction using Recurrent Neural Networks (RNN)

- Prediction for sine wave function using Gaussian process (code / notebook)
- Prediction for sine wave function using RNN (code / notebook)
- Prediction for electricity price (code / notebook)



#### Dependencies

```
Python (3.4.4)
TensorFlow (r0.9)
numpy (1.11.1)
pandas (0.16.2)
cuda (to run examples on GPU)
```

#### Dataset

- Energy Price Forecast 2016: http://complatt.smartwatt.net
- Or use the uploaded csv file for price history for 2015.

#### Current issues

- ```tf:split_squeeze``` is deprecated and will be removed after 2016-08-01. Use ```tf.unpack``` instead.
- ```tf:dnn``` is deprecated and will be removed after 2016-08-01. Use ```tf.contrib.layers.stack``` instead.

Now I am working on modifying previous source code for tensorflow ver. 0.10.0rc0.
