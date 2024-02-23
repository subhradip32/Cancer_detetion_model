# Introduction
This is an internship project which was given to me on my first intership on AIML. I this project I had used the cancer detection dataset for binary classification from the kaggel database. 
Instead of using premade CNN models I had used my own build model architecture using the tensorflow module. 

## Here is an example of the dataset visualised using the matplotlib
![image](https://github.com/subhradip32/Cancer_detetion_model/assets/83198378/2aa987e2-c387-49e7-afed-334a8edd2b22)


# Model Creation
**Model Description**
```
num_classes = 2
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16, (3,3),1, activation='relu',input_shape = (128,128,3)),
  tf.keras.layers.Dropout(.2),
  tf.keras.layers.Conv2D(32, (3,3), 1,activation='relu',padding = "valid"),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, (3,3), 1,activation='relu',padding = "valid"),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Dropout(.2),
  tf.keras.layers.Conv2D(32, (3,3), 1,activation='relu',padding = "valid"),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Dropout(.2),
  tf.keras.layers.Conv2D(32, (3,3), 1,activation='relu',padding = "valid"),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])
```
**Model Summary**
```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_5 (Conv2D)           (None, 126, 126, 16)      448       
                                                                 
 dropout_2 (Dropout)         (None, 126, 126, 16)      0         
                                                                 
 conv2d_6 (Conv2D)           (None, 124, 124, 32)      4640      
                                                                 
 max_pooling2d_4 (MaxPoolin  (None, 62, 62, 32)        0         
 g2D)                                                            
                                                                 
 conv2d_7 (Conv2D)           (None, 60, 60, 32)        9248      
                                                                 
 max_pooling2d_5 (MaxPoolin  (None, 30, 30, 32)        0         
 g2D)                                                            
                                                                 
 dropout_3 (Dropout)         (None, 30, 30, 32)        0         
                                                                 
 conv2d_8 (Conv2D)           (None, 28, 28, 32)        9248      
                                                                 
 max_pooling2d_6 (MaxPoolin  (None, 14, 14, 32)        0         
 g2D)                                                            
                                                                 
 dropout_4 (Dropout)         (None, 14, 14, 32)        0         
                                                                 
 conv2d_9 (Conv2D)           (None, 12, 12, 32)        9248      
                                                                 
 max_pooling2d_7 (MaxPoolin  (None, 6, 6, 32)          0         
 g2D)                                                            
                                                                 
 flatten_1 (Flatten)         (None, 1152)              0         
                                                                 
 dense_2 (Dense)             (None, 128)               147584    
                                                                 
 dense_3 (Dense)             (None, 2)                 258       
                                                                 
=================================================================
Total params: 180674 (705.76 KB)
Trainable params: 180674 (705.76 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```
# Accracy Graph
![image](https://github.com/subhradip32/Cancer_detetion_model/assets/83198378/a809810e-2e11-4012-ae93-daf09a2879a9)

# Summary
I had tried to use multiple model optimization methods to get maximum accuracy like- batch normalisation , dropout,regularization and many more. 
```
Epoch 32/32
9/9 [==============================] - 1s 134ms/step - loss: 4.4477 - accuracy: 0.7083 - val_loss: 4.5480 - val_accuracy: 0.7018
```
As you can see this is the best I can get in this dataset without the data-augmentation, which is around **71%**.
