import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer,Input
from tensorflow.python.keras.layers import Reshape,MaxPooling2D,Conv2D,Dense,Flatten


print(tf.__version__)
#print(tf.keras.__version__)

from tensorflow.examples.tutorials.mnist  import input_data
data=input_data.read_data_sets("./data/MNIST",one_hot=True)

data.test.cls=np.argmax(data.test.labels,axis=1)

img_size=28
img_size_flat=img_size*img_size
img_shape=(img_size,img_size)
img_shape_full=(img_size,img_size,1)
num_channels=1
num_classes=10

model=Sequential()
model.add(InputLayer(input_shape=(img_size_flat,)))
model.add(Reshape(img_shape_full))

model.add(Conv2D(kernel_size=5,strides=1,filters=16,padding='same',activation='relu',name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Conv2D(kernel_size=5,strides=1,filters=36,padding='same',activation='relu',name="layer_conv2"))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))

from tensorflow.python.keras.optimizers import Adam
optimizer=Adam(lr=1e-3)

model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=data.train.images,
          y=data.train.labels,
          epochs=1,
          batch_size=128)
result=model.evaluate(x=data.test.images,
                      y=data.test.labels)


for name,value in zip(model.metrics_names,result):
    print(name,value)


def plot_images(images,cls_true,cls_pred):
    assert len(images)==len(cls_true)==9
    fig,axes=plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    for i,ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape),cmap='binary')
        if cls_pred is None:
            xlabel="True:{0}".format(cls_true[i])
        else:
            xlabel="True:{0},Pred:{1}".format(cls_true[i],cls_pred[i])

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

images=data.test.images[0:9]
cls_true=data.test.cls[0:9]
y_pred=model.predict(x=images)
cls_pred=np.argmax(y_pred,axis=1)
plot_images(images=images,
            cls_true=cls_true,
            cls_pred=cls_pred)

