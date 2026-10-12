
import numpy as np
import joblib


def test_model():
  model = joblib.load("model.joblib")
  X = np.array([ 0.04534098, -0.04464164, -0.00620595, -0.01599898,  0.1250187 ,
         0.1251981 ,  0.019187  ,  0.03430886,  0.03243232, -0.0052198 ]).reshape(1, -1)

  assert isinstance(model.predict(X)[0], (np.floating, float))


def test_output_shape():
  model = joblib.load("model.joblib")
  X = np.array([ 0.04534098, -0.04464164, -0.00620595, -0.01599898,  0.1250187 ,
         0.1251981 ,  0.019187  ,  0.03430886,  0.03243232, -0.0052198 ]).reshape(1, -1)

  assert model.predict(X).shape == (1, )
