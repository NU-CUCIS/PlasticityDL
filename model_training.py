import tensorflow as tf 
from keras.regularizers import l2
from keras.models import Model 
from keras.layers.core import Activation, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import layers
from tensorflow.python.framework import ops 
import h5py
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
import numpy as np 
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, BatchNormalization
import sys
from scipy.misc import imread, imresize
import pickle
import warnings
warnings.filterwarnings('ignore')

def residual_pool(num_filter, num_layers, prev_layer, L2, BN, activation, pool):
	"""
	build residual block
	then have a pooling layer
	"""
	identity = prev_layer
	z = prev_layer
	for i in range(num_layers-1):
		z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
		if BN:
			z = BatchNormalization(axis=-1)(z)
		z = Activation(activation)(z)
	z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
	if BN:
		z = BatchNormalization(axis=-1)(z)
	z = layers.add([z, identity])
	z = Activation(activation)(z)
	if pool:
		a = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(z)
		b = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(z)
		z = layers.add([a, b])
	else:
		z = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(z)
	return z

def residual_nopool_changeChannelnum(num_filter, num_layers, prev_layer, L2, BN, activation, pool):
	"""
	first residual block, the input dimension of which is changed
	use 1*1 conv to match the dimension of channels of previous block.
	then have residual block
	"""
	identity = Conv2D(num_filter, (1, 1), padding='same', W_regularizer=l2(L2))(prev_layer)
	if BN:	
		identity = BatchNormalization(axis=-1)(identity)
	z = prev_layer
	for i in range(num_layers-1):
		z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
		if BN:	
			z = BatchNormalization(axis=-1)(z)
		z = Activation(activation)(z)
	z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
	if BN:	
		z = BatchNormalization(axis=-1)(z)	
	z = layers.add([z, identity])
	z = Activation(activation)(z)
	return z

def image_preprocessing(file, dataset, labelset, label, process_index):
	first_index = int(process_index[0])
	second_index = int(process_index[1])
	third_index = int(process_index[2])
	short_dim = [256, 288, 320, 352]
	data = imread(file)
	dim = short_dim[first_index]
	data_temp = imresize(data, (dim,dim*3))
	crop_range = [0,1,2]
	i = crop_range[second_index]	
	square = data_temp[:, (dim*i):(dim*(i+1))]
	if third_index == 0:
		crop5 = square[int(dim/2-112):int(dim/2+112),int(dim/2-112):int(dim/2+112)]
		dataset.append(crop5)
		labelset.append(label)
	else:
		crop6 = imresize(square, (224,224))
		dataset.append(crop6)
		labelset.append(label)
	 

def generator(files, labels, shuffle, batch):
	path = './image/'
	while 1:
		index = np.arange(len(files))
		if shuffle:
			np.random.shuffle(index)
		for i in range(int(len(files)/batch)):
			x = []
			y = []
			file_list = files[index[i*batch:(i+1)*batch]]
			label_list = labels[index[i*batch:(i+1)*batch]]
			for j in range(len(file_list)):
				file_temp = str(file_list[j]).split('&')[0]
				file_temp = file_temp.split('/')[-1]
				process_temp = str(file_list[j]).split('&')[1]
				temp = path + file_temp[:-3] + 'png'
				image_preprocessing(temp, x, y, label_list[j], process_temp)
			x = np.array(x)
			x = np.expand_dims(x, axis=-1)
			yield x, np.array(y)


# load data
with open('./data_file.pkl', 'rb') as f:
	data = pickle.load(f)
train_file = np.array(data['file'][:192])
train_label = np.array(data['label'][:192])
validation_file = np.array(data['file'][192:])
validation_label = np.array(data['label'][192:])

train_file, train_label = shuffle(train_file, train_label, random_state=0)
validation_file, validation_label = shuffle(validation_file, validation_label, random_state=0)
labelencoder = LabelBinarizer()
labelencoder.fit(range(int(max(train_label))+1))
train_label = labelencoder.transform(train_label)
labelencoder.fit(range(int(max(validation_label))+1))
validation_label = labelencoder.transform(validation_label)

## parameter sets
crop_size = 24 # number of crop for each image
batchsize = crop_size # batch size
n_epoch = 2 # number of training epoch
patience = 20 # patience for early stopping
L2 = 0.001 # penalty for L2 regularization
bn = True # True - use batch normalization
activation = 'relu' # activation function
pool = 0 # 0 for maxpooling, 1 for sum of max and average pooling
inp_size = (224, 224, 1) # input shape
# create 2D CNN model
print ('create model')
def build_model():
	inp = Input(shape=inp_size)
	x = Conv2D(16, (3, 3), padding='same', W_regularizer=l2(L2))(inp)
	if bn:
		x = BatchNormalization(axis=-1)(x)
	x = Activation(activation)(x)
	if pool:
		a = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
		b = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
		x = layers.add([a, b])
	else:
		x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

	x = residual_nopool_changeChannelnum(32, 2, x, L2, bn, activation, pool)
	x = residual_pool(32, 2, x, L2, bn, activation, pool)
	x = residual_nopool_changeChannelnum(64, 2, x, L2, bn, activation, pool)
	x = residual_pool(64, 2, x, L2, bn, activation, pool)
	x = residual_nopool_changeChannelnum(128, 2, x, L2, bn, activation, pool)
	x = residual_pool(128, 2, x, L2, bn, activation, pool)
	x = GlobalAveragePooling2D()(x)
	prediction = Dense(3, init='glorot_normal', activation='softmax', W_regularizer=l2(L2))(x)

	# compile the model 
	model = Model(input=inp, output= prediction)
	model.compile(loss = "categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
	return model
print ('-------------------------')
print ('fit model')
train_step = len(train_file) / batchsize
validation_batch = crop_size
validation_step = len(validation_file) / validation_batch
training_generator = generator(train_file, train_label, True, batchsize)
validation_generator = generator(validation_file, validation_label, False, validation_batch)
model = build_model()
filepath = './my_model.hdf5'
early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True)
history = model.fit_generator(generator=training_generator, steps_per_epoch=train_step, validation_data=validation_generator, validation_steps=validation_step, nb_epoch=n_epoch, callbacks=[early_stopping, checkpoint])




