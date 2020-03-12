print('Welcome to Auto Image Classification')
print('Initializing..')
print()

import tensorflow as tf
tf.test.gpu_device_name()
import seaborn as sns
from keras.models import Sequential
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dense
import sys

from tensorflow.python.client import device_lib
print('Checking Gpu Availability')
print(device_lib.list_local_devices())



from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50
from keras.applications.resnet import ResNet101
from keras.applications.resnet import ResNet152
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.resnet_v2 import ResNet101V2
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import NASNetMobile





from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
	

from sklearn.preprocessing import LabelEncoder
from keras.optimizers import SGD
from keras.layers import GlobalAveragePooling2D, Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation

from sklearn.metrics import log_loss
import os
import glob
from keras.models import Model

import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import load_model



base_model = None
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
model_name = ''
model = None
EPOCHS = 10
BATCH_SIZE = 32
STEPS_PER_EPOCH = 100
VALIDATION_STEPS = 64
WIDTH = 299
HEIGHT = 299
BATCH_SIZE = 32
train_generator = None
validation_generator = None



def directories(): 
	print('Train directory = data/train')
	print('Test directory = data/test')
	# match the number of folders in test and train
	num_train = len(os.listdir(TRAIN_DIR))
	num_test = len(os.listdir(TEST_DIR))
	if num_train != num_test:
		print()
		print('WARNING: The number of training classes is not equal to number of test classes')
	




def plot_training(history):
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(len(acc))
	plt.plot(epochs, acc, 'r.')
	plt.plot(epochs, val_acc, 'r')
	plt.title('Training and validation accuracy')
	
	plt.figure()
	plt.plot(epochs, loss, 'r.')
	plt.plot(epochs, val_loss, 'r-')
	plt.title('Training and validation loss')
	plt.show()
  




def img_aug():
	### Image Augmentation
	global train_generator
	global validation_generator
	print('Augmenting images')

	# data prep
	train_datagen = ImageDataGenerator(
		preprocessing_function=preprocess_input,
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest')
	
	validation_datagen = ImageDataGenerator(
		preprocessing_function=preprocess_input,
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest')
	
	train_generator = train_datagen.flow_from_directory(
		TRAIN_DIR,
		target_size=(HEIGHT, WIDTH),
			batch_size=BATCH_SIZE,
			class_mode='categorical')
		
	validation_generator = validation_datagen.flow_from_directory(
		TEST_DIR,
		target_size=(HEIGHT, WIDTH),
		batch_size=BATCH_SIZE,
		class_mode='categorical')
	print('Data Augmentation complete')
	print('Plotting some of the augmented images')
	# plotting some augmentated images
	x_batch, y_batch = next(train_generator)
	plt.figure(figsize=(12, 9))
	for k, (img, lbl) in enumerate(zip(x_batch, y_batch)):
		plt.subplot(4, 8, k+1)
		plt.imshow((img + 1) / 2)
		plt.axis('off')
		



def training(model):
	directories()
	img_aug()
	global EPOCHS
	global BATCH_SIZE
	global STEPS_PER_EPOCH
	global VALIDATION_STEPS
	
	print('Enter the number of epochs: ')
	EPOCHS = int(input())
	if EPOCHS < 0:
		print("ERROR: Number of epochs should be greater than 0")
		return
	print('Enter the number of steps per epoch: ')
	STEPS_PER_EPOCH = int(input())
	if STEPS_PER_EPOCH < 0:
		print("ERROR: Number of iterations should be greater than 0")
		return
	print('Enter the Batch Size(default: 32): ')
	BATCH_SIZE = int(input())
	if BATCH_SIZE < 0:
		print('ERROR: Batch Size should be greater than 0')
		return
	
	print('Starting Training')
	history = model.fit_generator(
		train_generator,
		epochs=EPOCHS,
		steps_per_epoch=STEPS_PER_EPOCH,
		validation_data=validation_generator,
		validation_steps=VALIDATION_STEPS)
	
	print('Training Complete')
	print('Saving Model')
	# saving the model
	model.save('classifier.h5')
	print('Model saved as classifier.h5')
	
	print('Plotting the learning curve of the model')
	plot_training(history)

	




def model_exist():
	global model_name
	global model
	print('Enter the name of the model with extension(.h5): ')
	model_name = input()
	if model_name.split('.')[-1] != 'h5':
		print("ERROR: Model can not be re-trained as it is not a .h5 model")
	else:
		model = load_model(model_name)
		training(model)
	




	
	
def chosen_model(choice):
	global base_model
	if choice == 19:
		model_exist()
	else:
		while(1):
			print()
			print('Transfer Learning? - Will use pre-trained model with imagenet weights')
			print('y')
			print('n')
			weights_wanted = input()
			if weights_wanted.upper() != 'Y' and weights_wanted.upper() != 'N':
				print('ERROR: Please enter a valid choice')
			else:
				break
		if choice == 1:
			print('Selected Model = Xception')
			if weights_wanted.upper() == 'Y': 
				base_model = Xception(weights = 'imagenet', include_top=False)
			else:
				base_model = Xception(weights = None, include_top=False)
		if choice == 2:
			print('Selected Model = VGG16')
			if weights_wanted.upper() == 'Y': 
				base_model = VGG16(weights = 'imagenet', include_top=False)
			else:
				base_model = VGG16(weights = None, include_top=False)
		if choice == 3:
			print('Selected Model = VGG19')
			if weights_wanted.upper() == 'Y': 
				base_model = VGG19(weights = 'imagenet', include_top=False)
			else:
				base_model = VGG19(weights = None, include_top=False)
		if choice == 4:
			print('Selected Model = ResNet50')
			if weights_wanted.upper() == 'Y': 
				base_model = ResNet50(weights = 'imagenet', include_top=False)
			else:
				base_model = ResNet50(weights = None, include_top=False)
		if choice == 5:
			print('Selected Model = ResNet101')
			if weights_wanted.upper() == 'Y': 
				base_model = ResNet101(weights = 'imagenet', include_top=False)
			else:
				base_model = ResNet101(weights = None, include_top=False)
		if choice == 6:
			print('Selected Model = ResNet152')
			if weights_wanted.upper() == 'Y': 
				base_model = ResNet152(weights = 'imagenet', include_top=False)
			else:
				base_model = ResNet152(weights = None, include_top=False)
		if choice == 7:
			print('Selected Model = ResNet50V2')
			if weights_wanted.upper() == 'Y': 
				base_model = ResNet50V2(weights = 'imagenet', include_top=False)
			else:
				base_model = ResNet50V2(weights = None, include_top=False)
		if choice == 8:
			print('Selected Model = ResNet101V2')
			if weights_wanted.upper() == 'Y': 
				base_model = ResNet101V2(weights = 'imagenet', include_top=False)
			else:
				base_model = ResNet101V2(weights = None, include_top=False)
		if choice == 9:
			print('Selected Model = ResNet152V2')
			if weights_wanted.upper() == 'Y': 
				base_model = ResNet152V2(weights = 'imagenet', include_top=False)
			else:
				base_model = ResNet152V2(weights = None, include_top=False)
		if choice == 10:
			print('Selected Model = InceptionV3')
			if weights_wanted.upper() == 'Y': 
				base_model = InceptionV3(weights = 'imagenet', include_top=False)
			else:
				base_model = InceptionV3(weights = None, include_top=False)
		if choice == 11:
			print('Selected Model = InceptionResNetV2')
			if weights_wanted.upper() == 'Y': 
				base_model = InceptionResNetV2(weights = 'imagenet', include_top=False)
			else:
				base_model = InceptionResNetV2(weights = None, include_top=False)
		if choice == 12:
			print('Selected Model = MobileNet')
			if weights_wanted.upper() == 'Y': 
				base_model = MobileNet(weights = 'imagenet', include_top=False)
			else:
				base_model = MobileNet(weights = None, include_top=False)
		if choice == 13:
			print('Selected Model = MobileNetV2')
			if weights_wanted.upper() == 'Y': 
				base_model = MobileNetV2(weights = 'imagenet', include_top=False)
			else:
				base_model = MobileNetV2(weights = None, include_top=False)
		if choice == 14:
			print('Selected Model = DenseNet121')
			if weights_wanted.upper() == 'Y': 
				base_model = DenseNet121(weights = 'imagenet', include_top=False)
			else:
				base_model = DenseNet121(weights = None, include_top=False)
		if choice == 15:
			print('Selected Model = DenseNet169')
			if weights_wanted.upper() == 'Y': 
				base_model = DenseNet169(weights = 'imagenet', include_top=False)
			else:
				base_model = DenseNet169(weights = None, include_top=False)
		if choice == 16:
			print('Selected Model = DenseNet201')
			if weights_wanted.upper() == 'Y': 
				base_model = DenseNet201(weights = 'imagenet', include_top=False)
			else:
				base_model = DenseNet201(weights = None, include_top=False)
		if choice == 17:
			print('Selected Model = NASNetLarge')
			if weights_wanted.upper() == 'Y': 
				base_model = NASNetLarge(weights = 'imagenet', include_top=False)
			else:
				base_model = NASNetLarge(weights = None, include_top=False)
		if choice == 18:
			print('Selected Model = NASNetMobile')
			if weights_wanted.upper() == 'Y': 
				base_model = NASNetMobile(weights = 'imagenet', include_top=False)
			else:
				base_model = NASNetMobile(weights = None, include_top=False)
			 
		CLASSES = len(os.listdir('data/train'))
		print('Number of Classes = {}'.format(CLASSES))
		
		x = base_model.output
		x = GlobalAveragePooling2D(name='avg_pool')(x)
		x = Dropout(0.4)(x)
		predictions = Dense(CLASSES, activation='softmax')(x)
		model = Model(inputs=base_model.input, outputs=predictions)
		
		for layer in base_model.layers:
			layer.trainable = False
			
		model.compile(optimizer='rmsprop',
					  loss='categorical_crossentropy',
					  metrics=['accuracy'])
		training(model)
			


	


while(1):
	#choosing model
	print('Choose the image classification model by enterring its corresponding number:')
	print('1. Xception')
	print('2. VGG16')
	print('3. VGG19')
	print('4. ResNet50')
	print('5. ResNet101')
	print('6. ResNet152')
	print('7. ResNet50V2')
	print('8. ResNet101V2')
	print('9. ResNet152V2')
	print('10. InceptionV3')
	print('11. InceptionResNetV2')
	print('12. MobileNet')
	print('13. MobileNetV2')
	print('14. DenseNet121')
	print('15. DenseNet169')
	print('16. DenseNet201')
	print('17. NASNetLarge')
	print('18. NASNetMobile')
	print('19. Existing model')
	print('20. Exit')
	choice = int(input())
	if choice > 20 or choice < 0:
		print('ERROR: Please enter a valid number')
	else:
		break

if choice < 20 and choice > 0:
	chosen_model(choice)



def predict(model, img):
	"""Run model prediction on image
	Args:
		model: keras model
		img: PIL format image
	Returns:
		list of predicted labels and their probabilities 
	"""
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	preds = model.predict(x)
	return preds[0]



def plot_result(preds):
	classes = os.listdir(TRAIN_DIR)
	top_5 = np.argsort(preds)[:-6:-1]
	x = [classes[top_5[i]] for i in range(0, len(top_5))]
	y = [preds[top_5[i]]*100 for i in range(0, len(top_5))]
	
	plt.figure()
	plt.subplot(1, 2,1)
	plt.imshow(img)
	plt.grid(None)
	plt.axis('off')
	plt.subplot(1,2,2)
	sns.barplot(x=y, y=x, orient = "h")
	for i in range(5):
		print("{}".format(classes[top_5[i]])+" ({0:.2f}%)".format(preds[top_5[i]]*100))
#    for p in splot.patches:
#        splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 
#                              p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), 
#                              textcoords = 'offset points')
		


# img = image.load_img(TEST_DIR + '/cat/1042.jpeg', target_size=(HEIGHT, WIDTH))
# preds = predict(model, img)
# plot_result(preds)


# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
# preds = model.predict(x)
# preds = preds[0]



# classes = os.listdir(TRAIN_DIR)
# top_5 = np.argsort(preds)#[:-6:-1]
# # top_5 = list(top_5[0])

# x = [classes[i] for i in top_5]
# y = [preds[i]*100 for i in top_5]

# plt.figure()
# plt.subplot(1, 2,1)
# plt.imshow(img)
# plt.grid(None)
# plt.axis('off')
# plt.subplot(1,2,2)
# sns.barplot(x=y, y=x, orient = "h")
# for i in range(0, len(top_5)):
#     print("{}".format(classes[top_5[i]])+" ({0:.2f}%)".format(preds[0][top_5[i]]*100))
# #    for p in splot.patches:
	

# result = x[1].upper()
# pred_proba = y[1]
