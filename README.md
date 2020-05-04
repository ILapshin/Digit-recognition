# Digit-recognition
This is my machine learning project for participating in the Kaggle competition for MNIST dataset handwritten digits recognition.

The main class Recognizer has a nueral network model made as TensorFlow.Keras.Sequential which consists of sequent levels of two convolutional leyers followed by one maximum pooloing level and two fully connected leyers in the end.

To increas prediction accuracy an upgrade of dataset was implemented. Training dataset was expanded by adding a number of element-wisely modified oroginal datasets. Modification was made by rotating and shifting each digit by a moderate value. Additionally, this technic allowed me to validate a model fitted this way to the original full unmodified training dataset for much visibility.

Optimal architecture of the network which I stopped on contained two sequences of double convolutional leyers followed by poolong leyer. The accuracy score I managed to reach using such a model was 0.99528.
