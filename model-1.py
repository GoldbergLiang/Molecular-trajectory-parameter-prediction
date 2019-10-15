import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

# load dataset
data_train = pandas.read_csv("train_data.csv", header=None)
data_test = pandas.read_csv("test_data.csv", header=None)
dataset_train = data_train.values
dataset_test = data_test.values


# split into input (X) and output (Y) variables
X_train = dataset_train[:, 0:2]
Y_train=dataset_train[:, 2:]
X_test= dataset_test[:, 0:2]
Y_test=dataset_test[:, 2:]

filepath="weights1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model = Sequential()
model.add(Dense(30, input_dim=2, kernel_initializer='normal', activation='relu'))
model.add(Dense(198, kernel_initializer='normal'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=5, epochs=50, callbacks=[checkpoint])
scores = model.evaluate(X_test, Y_test, batch_size=5, verbose=1)
print('Test loss:', scores)
pred=model.predict(X_test, batch_size=5)
save = pandas.DataFrame(pred)
save.to_csv('C:/Users/111/Desktop/keras/predict.csv',index=False,header=False)



