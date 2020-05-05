# Example of using my Recognizer with wich I reached a 0.99528 accuracy

from digit_recognizer import Recognizer

a = Recognizer(levels=(2, 2))
result = a.fit(a.upgrade(8), save_weights=True, save_path='model/exapmle_weights_f.h5') # A directory .../model must already exists
a.verify()
a.output(a.predict(a.X_test), 'example_prediction_f')
