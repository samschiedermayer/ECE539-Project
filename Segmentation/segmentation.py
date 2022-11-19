#!/usr/bin/python3
import argparse
import os
import re
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def LoadData(imgPath:str=None, maskPath:str=None, shape:int=256, train_ratio:float=0.90, evaluate:bool=False):
    images = []
    masks  = []

    training_file_path="data/split_list.csv"

    if not os.path.exists(training_file_path):
        imgNames = os.listdir(imgPath)
        random.shuffle(imgNames)
        df = pd.DataFrame(imgNames)
        df.to_csv(training_file_path)

    df = pd.read_csv(training_file_path).iloc[:,-1]
    imgNames = list(df.to_dict().values())

    class_dict = {}
    with open('data/class_dict.csv','r') as cdf:
      next(cdf)
      for i,line in enumerate(cdf):
        class_dict[i] = [int(x) for x in line.replace(" ","").replace("\n","").split(",")[1:]]

    maskNames = []

    ## generating mask names
    for mem in imgNames:
        maskNames.append(re.sub('\.jpg', '.png', mem))

    imgAddr  = imgPath + '/'
    maskAddr = maskPath + '/'

    len_training = int(len(imgNames)*train_ratio)
    len_test = len(imgNames) - len_training

    if not evaluate:
        images_file  = "data/train_images.npy"
        masks_file   = "data/train_masks.npy"
    else:
        images_file  = "data/test_images.npy"
        masks_file   = "data/test_masks.npy"

    if not (os.path.exists(images_file) and os.path.exists(masks_file)):
        for i in range(len_training):
            try:
                if not evaluate:
                    img = plt.imread(imgAddr + imgNames[i])
                    mask = plt.imread(maskAddr + maskNames[i])
                else:
                    img = plt.imread(imgAddr + imgNames[i+len_training])
                    mask = plt.imread(maskAddr + maskNames[i+len_training])
            except:
                continue

            img = cv2.resize(img, (shape, shape))
            mask = (cv2.resize(mask, (shape, shape)) * 256).astype(np.uint8)

            output_mask = np.zeros((256,256),dtype=np.float32)
            for k,v in class_dict.items():
                indices = np.all(mask==v, axis=-1)
                output_mask[indices] = k

            images.append(img)
            masks.append(output_mask)

        images = np.array(images)
        masks  = np.array(masks)

        print("Caching data...")
        np.save(images_file,images)
        np.save(masks_file, masks)
    else:
        print("Loading cached data...")
        images = np.load(images_file)
        masks  = np.load(masks_file)

    return images, masks

def GetModel(lr:float) -> tf.keras.models.Sequential:

    # create a reference to the output layer of the encoder for rnn connection in decoder
    encoder_out = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same')

    # defining our CNN for encoding and decoding
    model = tf.keras.models.Sequential([
        #encoder
        tf.keras.layers.Input(shape= (256, 256, 3)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu', padding='valid'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'),
        encoder_out,

        #decoder
        tf.keras.layers.UpSampling2D(size=(2,2)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'),

        tf.keras.layers.UpSampling2D(size=(2,2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D(size=(2,2)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D(size=(2,2)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same'),
    ])
    #model.summary()
    
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr), loss='mean_absolute_error', metrics=['acc'])
    #model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss='mean_squared_error', metrics=['acc'])

    return model

def CountColors(img: np.ndarray) -> dict:
    flattened = np.reshape(img,(256*256,3))
    print(np.shape(flattened))
#for pix in img:


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate",action='store_true')
    parser.add_argument("--load", action='store_true')
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)

    args = parser.parse_args()

    if args.evaluate:
        print("Running in evaluation mode...")
    else:
        print("Running in training mode...")
        print(f"Hyperparameters: {args.epoch} epochs, {args.lr} learning rate")

    print("Loading data...")
    images, masks = LoadData(imgPath='data/images', maskPath='data/labels/pixel_level_labels_colored', shape=256, evaluate=args.evaluate)

    #plt.subplot(1,2,1)
    #plt.imshow(train['img'][1])
    #plt.subplot(1,2,2)
    #plt.imshow(train['mask'][1])
    #plt.show()

    print("Compiling model...")
    model = GetModel(args.lr)

    if args.load or args.evaluate:
        print("Loading parameters for model...")
        model.load_weights('weights/image_segmentation.weights')

    if args.evaluate:
        print("Evaluating model...")
        predicted_masks = model.predict(images)
        results = model.evaluate(images,masks)
        print(f"Results: {results}")

        print(f"output size: {np.shape(predicted_masks[0])}, desired size: {np.shape(masks[0])}")
        print(f"output dtype: {predicted_masks[0].dtype}, desired dtype: {masks[0].dtype}")

        #CountColors(predicted_masks[0])

        #i = 0
        #for j in range(0,256):
        #    for k in range(0,256):
        #        print("y: {}, y_hat: {}".format(masks[i][j,k], predicted_masks[i][j,k][0]))
        for i in range(10):
            plt.subplot(1,3,1)
            plt.imshow(images[i])
            plt.subplot(1,3,2)
            plt.imshow(masks[i])
            plt.subplot(1,3,3)
            plt.imshow(predicted_masks[i])
            plt.show()

    else:
        print("Beginning training...")
        cb = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=20)

        #retVal = model.fit(images, masks, epochs=args.epoch, verbose=1, callbacks=[cb])
        retVal = model.fit(images, masks, epochs=args.epoch, verbose=1, callbacks=[])

        plt.plot(retVal.history['loss'], label = 'training_loss')
        plt.plot(retVal.history['acc'], label = 'training_accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

        model.save_weights('weights/image_segmentation.weights')


