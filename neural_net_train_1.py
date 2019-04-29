from keras.models import Sequential
from keras.layers import Dense
import numpy

# Random seed for reproducinilty:
numpy.random.seed(7)

# LOAD datafile:
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# Split the dataset into input(x) and output (y)
# X = cols 0 to 7 (use to find patterns related to classification in column 8)
X = dataset[:, 0:8]
# Y= column 8 (determines if the user has diabetes: Supervised Learning)
Y = dataset[:, 8]

"""
The first layer has 12 neurons and expects 8 input variables.
The second hidden layer has 8 neurons
the output layer has 1 neuron to predict the class (onset of diabetes or not).
"""
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
