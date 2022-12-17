#!/usr/bin/python3
import argparse
import os
import re
import cv2
import random
import webcolors
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
import tensorboard
from keras.utils.vis_utils import plot_model

def get_classes() -> dict[str,int]:
    compression_dict = {
        'bag':'accessories',
        'belt':'accessories',
        'boots':'shoes',
        'bracelet':'accessories',
        'clogs':'shoes',
        'earrings':'accessories',
        'flats':'shoes',
        'glasses':'accessories',
        'gloves':'accessories',
        'heels':'shoes',
        'loafers':'shoes',
        'necklace':'accessories',
        'pumps':'shoes',
        'purse':'accessories',
        'ring':'accessories',
        'sandals':'shoes',
        'sneakers':'shoes',
        'stockings':'socks',
        'sunglasses':'accessories',
        'tie':'accessories',
        'tights':'socks',
        'wallet':'accessories',
        'watch':'accessories',
        'wedges':'shoes',
        'sweatshirt':'hoodie',
    }

    pixel_values = []

    class_dict = {}
    with open('Segmentation/data/class_dict.csv','r') as cdf:
        next(cdf)
        for idx, line in enumerate(cdf):
            spl = line.replace(" ","").replace("\n","").split(",")
            name = spl[0]
            pixel_values.append([[int(x) for x in spl[1:]]])
            class_dict[name] = [idx,idx]



    for k,v in compression_dict.items():
        orig_idx = class_dict[k][0]
        new_idx  = class_dict[v][1]
        class_dict[k] = [orig_idx,new_idx]


    classes = {}
    i = 0
    for c_name, (orig_idx, new_idx) in class_dict.items():
        if orig_idx != new_idx:
            pixel_values[new_idx].append(pixel_values[orig_idx][0])
            pixel_values[orig_idx] = []
        else:
            classes[c_name] = i
            i += 1

    return classes

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

    compression_dict = {
        'bag':'accessories',
        'belt':'accessories',
        'boots':'shoes',
        'bracelet':'accessories',
        'clogs':'shoes',
        'earrings':'accessories',
        'flats':'shoes',
        'glasses':'accessories',
        'gloves':'accessories',
        'heels':'shoes',
        'loafers':'shoes',
        'necklace':'accessories',
        'pumps':'shoes',
        'purse':'accessories',
        'ring':'accessories',
        'sandals':'shoes',
        'sneakers':'shoes',
        'stockings':'socks',
        'sunglasses':'accessories',
        'tie':'accessories',
        'tights':'socks',
        'wallet':'accessories',
        'watch':'accessories',
        'wedges':'shoes',
        'sweatshirt':'hoodie',
    }
    
    pixel_values = []
    class_dict = {}
    with open('data/class_dict.csv','r') as cdf:
        next(cdf)
        for idx, line in enumerate(cdf):
            spl = line.replace(" ","").replace("\n","").split(",")
            name = spl[0]
            pixel_values.append([[int(x) for x in spl[1:]]])
            class_dict[name] = [idx,idx]
    
    for k,v in compression_dict.items():
        orig_idx = class_dict[k][0]
        new_idx  = class_dict[v][1]
        class_dict[k] = [orig_idx,new_idx]
        
    for orig_idx, new_idx in class_dict.values():
        if orig_idx != new_idx:
            pixel_values[new_idx].append(pixel_values[orig_idx][0])
            pixel_values[orig_idx] = []
        
    
    pixel_values = [x for x in pixel_values if x]
    
    num_dims = len(pixel_values)

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

            output_mask = np.zeros((256,256,num_dims),dtype=np.float32)
        
            for dim, px in enumerate(pixel_values):
                for individual_px in px:
                    indices = np.all(mask==individual_px, axis=-1)
                    output_mask[indices,dim] = 1.0

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

    print(f"Images taking up {(images.size * images.itemsize)/1e6}MB")
    print(f"Masks taking up {(masks.size  * masks.itemsize)/1e6}MB")
        
#     return tf.data.Dataset.from_tensor_slices(images), tf.data.Dataset.from_tensor_slices(masks)
#     return tf.data.Dataset.from_tensor_slices((images, masks))
    return images, masks

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS21_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def get_top_colors_and_categories(masks:np.ndarray, images:np.ndarray, mask_idx:int) -> Tuple[List[str],List[str]]:
    classes = get_classes()
    inv_classes = {v: k for k, v in classes.items()}
    exclusion_list = ['null','skin','hair','accessories']
    indices = [v  for k,v in classes.items() if not k in exclusion_list]

    top_classes = []
    top_values  = []
    masks = (masks == masks.max(axis=-1)[:,None]).astype('uint8')
    filtered_masks = masks[mask_idx].T[indices]
    for idx, mask in enumerate(filtered_masks):
        print(mask.shape)
        unique, counts = np.unique(mask, return_counts=True)
        print(f"{unique}: {counts}")
        if len(unique) > 1:
            i = indices[idx]
            if unique[1] == 1:
                top_classes.append(inv_classes[i])
                top_values.append(counts[1])
            else:
                top_classes.append(inv_classes[i])
                top_values.append(counts[0])

    np.amax(top_values)

    top_values = np.array(top_values)
    top_classes = np.array(top_classes)

    ind = np.argpartition(top_values, -2)[-2:]
    ind = ind[np.argsort(top_values[ind])]

    top_classes = top_classes[ind]

    colors = []
    for tc in top_classes:
        mask = (masks[mask_idx].T[classes[tc]].T).astype('uint8')

        pixels = images[mask_idx][mask==1]

        pixels = tuple(np.median(pixels,axis=0).astype('uint8'))
        _, color = get_colour_name(pixels)
        colors.append(color)

    return top_classes, colors

def GetModel(lr:float,visualize:bool=False) -> tf.keras.models.Sequential:

    # defining our CNN for encoding and decoding
    model = tf.keras.models.Sequential([
        #encoder
        tf.keras.layers.Input(shape= (256, 256, 3)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu', padding='valid'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same'),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), activation='relu', padding='same'),

        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'),
        
        #decoder
        tf.keras.layers.UpSampling2D(size=(2,2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D(size=(2,2)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D(size=(2,2)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=34, kernel_size=(3,3), padding='same'),
        tf.keras.layers.Reshape((256*256,34)),
        tf.keras.layers.Activation('softmax'),
        tf.keras.layers.Reshape((256,256,34)),
    ])
    if visualize:
        model.summary()
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss='categorical_crossentropy', metrics=['acc'])

    return model

def run_model(evaluate:bool=False, load:bool=False, epoch:int=10, lr:float=5e-5, visualize:bool=False, inference:str=None):
    if inference:
        print("Running in inference mode...")
    elif evaluate:
        print("Running in evaluation mode...")
    else:
        print("Running in training mode...")
        print(f"Hyperparameters: {epoch} epochs, {lr} learning rate")

    if not inference:
        print("Loading data...")
        images, masks = LoadData(imgPath='data/images', maskPath='data/labels/pixel_level_labels_colored', shape=256, evaluate=evaluate)

    print("Compiling model...")
    model = GetModel(lr,visualize=visualize)

    if load or evaluate or inference:
        print("Loading parameters for model...")
        model.load_weights('Segmentation/weights/image_segmentation.weights')

    if True:
        if inference:
            print("loading image to inference...")
            try:
                img = plt.imread(inference)
                img = cv2.resize(img, (256, 256))
            except:
                print("Error, could not load image for inference!")
                return

            images = np.array([img])

            print("Inferencing with model...")
            predicted_masks = model.predict(images)
            
            classes, colors = get_top_colors_and_categories(predicted_masks,images,0)

            print(classes)
            print(colors)

        elif evaluate:
            print("Evaluating model...")
            predicted_masks = model.predict(images)
            results = model.evaluate(images,masks)
            print(f"Results: {results}")

            print(f"output size: {np.shape(predicted_masks[0])}, desired size: {np.shape(masks[0])}")
            print(f"output dtype: {predicted_masks[0].dtype}, desired dtype: {masks[0].dtype}")

            for i in range(5):
                plt.subplot(1,3,1)
                plt.imshow(images[i])
                plt.subplot(1,3,2)
                plt.imshow(np.argmax(masks[i],axis=-1))
                plt.subplot(1,3,3)
                plt.imshow(np.argmax(predicted_masks[i],axis=-1))
                plt.show()

        else:
            print("Beginning training...")
            loss_cb = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=25)

            logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir)

            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(save_freq='epoch',save_weights_only=True,filepath='weights/model_checkpoint')

            retVal = model.fit(images, masks, epochs=epoch, verbose=1, callbacks=[tensorboard_cb,loss_cb,checkpoint_cb])

            plt.plot(retVal.history['loss'], label = 'training_loss')
            plt.plot(retVal.history['acc'], label = 'training_accuracy')
            plt.legend()
            plt.grid(True)
            plt.show()

            model.save_weights('weights/image_segmentation.weights')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", type=str, default=None)
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--load", action='store_true')
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)

    args = parser.parse_args()

    run_model(args.evaluate, args.load, args.epoch, args.lr, args.visualize, args.inference)
