# Example of using my Recognizer with which I reached a 0.99528 accuracy

from digit_recognizer import Recognizer

a = Recognizer()
result = a.fit(a.upgrade(10), save_weights=True, save_path='model/exapmle_weights.h5') # A directory .../model must already exists
a.verify()
a.output(a.predict(a.X_test), 'example_prediction')
