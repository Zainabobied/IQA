import tensorflow as tf
import tensorflow_addons as tfa
from skimage import io, color
import sys
from datasets.tf_dataset import denormalize
import numpy as np
#import piq
import os
from losses.tf_dists import DISTS
from keras.applications.vgg19 import preprocess_input

IMG_WIDTH    = 256
IMG_HEIGHT   = 256
IMG_CHANNELS = 3
size = (IMG_WIDTH,IMG_HEIGHT)

def directional_difference(image1, image2):
    if image1.shape[2] == 3:
        image1 = color.rgb2gray(denormalize(image1))
    if image2.shape[2] == 3:
        image2 = color.rgb2gray(denormalize(image2))
    diff = np.maximum(image2 - image1, 0)
    return diff

# modify source of prewitt
def pad(input, ksize, mode, constant_values):
    input = tf.convert_to_tensor(input)
    ksize = tf.convert_to_tensor(ksize)
    mode = "CONSTANT" if mode is None else upper(mode)
    constant_values = (
        tf.zeros([], dtype=input.dtype)
        if constant_values is None
        else tf.convert_to_tensor(constant_values, dtype=input.dtype)
    )

    assert mode in ("CONSTANT", "REFLECT", "SYMMETRIC")

    height, width = ksize[0], ksize[1]
    top = (height - 1) // 2
    bottom = height - 1 - top
    left = (width - 1) // 2
    right = width - 1 - left
    paddings = [[0, 0], [top, bottom], [left, right], [0, 0]]
    return tf.pad(input, paddings, mode=mode, constant_values=constant_values)


def prewitt(input, mode=None, constant_values=None, name=None):

    input = tf.convert_to_tensor(input)

    gx = tf.cast([[1, 0, -1], [1, 0, -1], [1, 0, -1]], input.dtype)
    gy = tf.cast([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], input.dtype)

    ksize = tf.constant([3, 3])

    input = pad(input, ksize, mode, constant_values)

    channel = tf.shape(input)[-1]
    shape = tf.concat([ksize, tf.constant([1, 1], ksize.dtype)], axis=0)
    gx, gy = tf.reshape(gx, shape), tf.reshape(gy, shape)
    shape = tf.concat([ksize, [channel], tf.constant([1], ksize.dtype)], axis=0)
    gx, gy = tf.broadcast_to(gx, shape), tf.broadcast_to(gy, shape)

    x = tf.nn.depthwise_conv2d(
        input, tf.cast(gx, input.dtype), [1, 1, 1, 1], padding="VALID"
    )
    y = tf.nn.depthwise_conv2d(
        input, tf.cast(gy, input.dtype), [1, 1, 1, 1], padding="VALID"
    )
    epsilon = 1e-08
    return tf.math.sqrt(x * x + y * y + sys.float_info.epsilon) 

# get right loss function
def which_loss(loss_fn):
    switcher = {
        'mse':      "mse",
        'L2':       L2_Loss,
        'SSIM':     SSIM_Loss,
        'MSGMS':    MSGMS_Loss,
        'COMBINED': COMBINED_Loss,
    }
    return switcher.get(loss_fn, "Invalid loss function")


""" Gradient Magnitude Map """
def Grad_Mag_Map(I, show = False):
    I = tf.reduce_mean(I, axis=-1, keepdims=True)
    I = tfa.image.median_filter2d(I, filter_shape=(3, 3), padding='REFLECT')
    x = prewitt(I)
    if show:
        x = tf.squeeze(x, axis=0).numpy()
    return x


""" Gradient Magnitude Similarity Map"""
def GMS(I, I_r, show=False, c=0.0026):
    g_I   = Grad_Mag_Map(I)
    g_Ir  = Grad_Mag_Map(I_r)
    similarity_map = (2 * g_I * g_Ir + c) / (g_I**2 + g_Ir**2 + c)
    if show:
        similarity_map = tf.squeeze(similarity_map, axis=0).numpy()
    return similarity_map


""" Gradient Magnitude Distance Map"""
def GMS_Loss(I, I_r):
    x = tf.reduce_mean(1 - GMS(I, I_r)) 
    return x


#### LOSS FUNCTIONS ####
""" Define MSGMS """
def MSGMS_Loss(I, I_r):
    # normal scale loss
    tot_loss = GMS_Loss(I, I_r)
    # pool 3 times and compute GMS
    for _ in range(3):
        I   = tf.nn.avg_pool2d(I,   ksize=2, strides=2, padding= 'VALID')
        I_r = tf.nn.avg_pool2d(I_r, ksize=2, strides=2, padding= 'VALID')
        # sum loss
        tot_loss += GMS_Loss(I, I_r)

    return tot_loss/4


""" Define SSIM loss"""
def SSIM_Loss(I, I_r):
    I   = tf.cast(I,   dtype=tf.double)
    I_r = tf.cast(I_r, dtype=tf.double)
    img_range = 1+1#tf.reduce_max(1)-tf.reduce_min(X_train)
    ssim = tf.image.ssim(I, I_r, max_val=img_range, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    return tf.reduce_mean(1 - ssim)

def ssim_Loss(I, I_r):
    # Compute the SSIM between the two images
    ssim = tf.image.ssim(I, I_r, max_val=1.0)
    
    # Calculate the mean SSIM value over all pixels
    loss = 1 - tf.reduce_mean(ssim)
    
    return loss

""" Define l2 loss"""
def L2_Loss(I, I_r):
    l2_loss = tf.keras.losses.MeanSquaredError()
    return l2_loss(I, I_r)

def l2_loss(I, I_r):
    return tf.math.reduce_mean(tf.math.squared_difference(I, I_r))

""" Define total loss"""  
def COMBINED_Loss(I, I_r, lambda_s=1.0, lambda_m=1.0):
    l2_loss = L2_Loss(I, I_r)
    S_loss  = SSIM_Loss(I, I_r)
    M_loss  = MSGMS_Loss(I, I_r)

    x = l2_loss + lambda_s * S_loss + lambda_m * M_loss 
    return [tf.reduce_mean(x), tf.reduce_mean(M_loss), tf.reduce_mean(S_loss), tf.reduce_mean(l2_loss)]

def COMBINED_W(lambda_s):
    def COMBINED_W_Loss(I, I_r):
        l2_loss = l2_loss(I, I_r)
        S_loss  = ssim_Loss(I, I_r)
        M_loss  = MSGMS_Loss(I, I_r)
        
        lambda_m = 1 - lambda_s

        x = lambda_s*S_loss + lambda_m*M_loss + l2_loss
        return tf.reduce_mean(x)
    return COMBINED_Loss

""" Define MSGMS Map """

def MSGMS_Map(I, I_r):
    I   = tf.cast(I, dtype=tf.float32)
    I_r = tf.cast(I_r, dtype=tf.float32)
    # normal scale similarity map
    gms_tot = GMS(I, I_r)

    # pool 3 times and compute GMS
    for _ in range(3):
        I   = tf.nn.avg_pool2d(I, ksize=2, strides=2, padding= 'VALID')
        I_r = tf.nn.avg_pool2d(I_r, ksize=2, strides=2, padding= 'VALID')
        # compute GMS 
        gms_scale = GMS(I, I_r)
        # upsample
        gms_scale = tf.image.resize(gms_scale, size=size)
        gms_tot  += gms_scale

    gms_map = gms_tot/4
    gms_map = tfa.image.mean_filter2d(gms_map) 
    return gms_map



""" Define MSGMS Anomaly Map """
def Anomaly_Map(I, I_r):
    return (1 - MSGMS_Map(I, I_r)).numpy()

def ms_ssim_similarity(data, outputs):
    # Normalize images to [0, 1]
    data = data / 255.0
    outputs = outputs / 255.0

    # Calculate MS-SSIM
      # Calculate SSIM at a single scale (full size)
    target_size = [data.shape[1], data.shape[2]]  # Assuming data and outputs have shape (batch_size, height, width, channels)
    data_scaled = tf.image.resize_with_crop_or_pad(data, target_size[0], target_size[1])
    outputs_scaled = tf.image.resize_with_crop_or_pad(outputs, target_size[0], target_size[1])
    ms_ssim = tf.image.ssim_multiscale(data_scaled, outputs_scaled, max_val=1.0)

    # Expand the dimensions to have shape (batch_size, height, width, 1)
    similarity_map = tf.expand_dims(ms_ssim, axis=-1)

    return similarity_map


# Define MDSI Anomaly Map
def mdsi_similarity(data, outputs):
    # Normalize images to [0, 1]
    data = data / 255.0
    outputs = outputs / 255.0

    # Compute structural similarity at multiple scales
    scales = [1, 2, 3, 4, 5]  # Adjust the scales as needed
    similarity_maps = []
    for scale in scales:
        target_size = [IMG_WIDTH // scale, IMG_HEIGHT // scale]
        data_scaled = tf.image.resize_with_crop_or_pad(data, target_size[0], target_size[1])
        outputs_scaled = tf.image.resize_with_crop_or_pad(outputs, target_size[0], target_size[1])
        diff = data_scaled - outputs_scaled
        similarity_map = tf.reduce_mean(diff * diff, axis=-1, keepdims=True)
        similarity_maps.append(similarity_map)

    # Resize all similarity maps to the same size
    target_shape = tf.shape(similarity_maps[0])[:-1] + [len(scales),]
    similarity_maps_resized = [tf.image.resize_with_crop_or_pad(sim_map, target_shape[0], target_shape[1]) for sim_map in similarity_maps]

    return tf.concat(similarity_maps_resized, axis=-1)

def ssim_similarity(data, outputs):
    # Normalize images to [0, 1]
    data = data / 255.0
    outputs = outputs / 255.0

    # Calculate SSIM at a single scale (full size)
    target_size = [data.shape[1], data.shape[2]]  # Assuming data and outputs have shape (batch_size, height, width, channels)
    data_scaled = tf.image.resize_with_crop_or_pad(data, target_size[0], target_size[1])
    outputs_scaled = tf.image.resize_with_crop_or_pad(outputs, target_size[0], target_size[1])

    ssim = tf.image.ssim(data_scaled, outputs_scaled, max_val=1.0)
    # Expand the dimensions to have shape (batch_size, height, width, 1)
    similarity_map = tf.expand_dims(ssim, axis=-1)

    return similarity_map

def psnr_similarity(data, outputs):
    # Normalize images to [0, 1]
    data = data / 255.0
    outputs = outputs / 255.0

   # Calculate PSNR at a single scale
    target_size = [data.shape[1], data.shape[2]]  # Assuming data and outputs have shape (batch_size, height, width, channels)
    data_scaled = tf.image.resize_with_crop_or_pad(data, target_size[0], target_size[1])
    outputs_scaled = tf.image.resize_with_crop_or_pad(outputs, target_size[0], target_size[1])
    similarity_map = tf.image.psnr(data_scaled, outputs_scaled, max_val=1.0)

    return similarity_map


import tensorflow as tf
import numpy as np

def resize_with_pad(image, target_height, target_width):
    # Resize the image using the 'nearest' method to preserve similarity values
    resized_image = tf.image.resize(image, [target_height, target_width], method='nearest')
    # Compute the amount of padding needed
    height_pad = target_height - tf.shape(resized_image)[1]
    width_pad = target_width - tf.shape(resized_image)[2]
    # Pad the image
    padded_image = tf.pad(resized_image, [[0, 0], [0, height_pad], [0, width_pad], [0, 0]], mode='CONSTANT')
    return padded_image

import tensorflow as tf
import numpy as np

def dists_similarity(data, outputs):
    # Set the parameters inside the function
    scale = 4
    alpha = 0.5
    beta = 0.5
    model_name = 'vgg19'

    # Assuming data and outputs are numpy arrays representing images in RGB format
    data = np.copy(data) / 255.0
    outputs = np.copy(outputs) / 255.0

    # Instantiate the DISTS model
    model = DISTS()

    # Preprocess data and outputs using VGG19 preprocessing
    data = preprocess_input(data)
    outputs = preprocess_input(outputs)

    # Resize data and outputs to the specified scale without changing aspect ratio
    # Resize data and outputs to the specified scale without changing aspect ratio
    data_scaled = tf.image.resize_with_pad(data, target_height=data.shape[0] // scale, target_width=data.shape[1] // scale)
    outputs_scaled = tf.image.resize_with_pad(outputs, target_height=outputs.shape[0] // scale, target_width=outputs.shape[1] // scale)

    # Expand dimensions to make it 4D (batch_size, height, width, channels)
    data_scaled = tf.expand_dims(data_scaled, axis=0)
    outputs_scaled = tf.expand_dims(outputs_scaled, axis=0)
    
    print("data_scaled shape:", data_scaled.shape)
    print("outputs_scaled shape:", outputs_scaled.shape)

    # Compute similarity map using the DISTS model
    similarity_map = model.get_score(data_scaled, outputs_scaled)
    print("similarity_map")
    
    # Squeeze the batch dimension (batch_size = 1) to get the similarity map
    similarity_map = tf.squeeze(similarity_map, axis=0)

    # Convert similarity_map to numpy array
    similarity_map_resized = similarity_map.numpy()

    return similarity_map_resized



import tensorflow as tf
import keras.applications as applications

import tensorflow as tf
from keras.applications import vgg19
from keras.applications import vgg16

def sobel_edges_rgb(img):
    sobel_x = tf.image.sobel_edges(img)
    sobel_x = tf.reduce_sum(sobel_x, axis=-1)  # Sum along the channel dimension
    return sobel_x

# Define the SSIM calculation function
def ssim_per_channel(img1, img2, max_val=1.0):
    # Calculate SSIM between two tensors
    ssim_val = tf.image.ssim(img1, img2, max_val=max_val)

    return ssim_val

def sobel_edges_rgb(images):
    # Convert grayscale images to RGB images
    images = tf.image.grayscale_to_rgb(images)
    
    # Calculate sobel edges
    sobel = tf.image.sobel_edges(images)
    
    # Calculate gradient magnitude
    gradient = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(sobel), axis=-1))
    
    return gradient


def vsi_similarity(data, outputs):
    print("data len:",len(data))
    print("outputs len:",len(outputs))
    # Normalize images to [0, 1]
    data = data / 255.0
    outputs = outputs / 255.0
    
    # Calculate SSIM at a single scale (full size)
    target_size = [data.shape[1], data.shape[2]]
    data_scaled = tf.image.resize_with_crop_or_pad(data, target_size[0], target_size[1])
    outputs_scaled = tf.image.resize_with_crop_or_pad(outputs, target_size[0], target_size[1])

    # Specify parameters for SSIM calculation
    max_val = 1.0  # Dynamic range of the images
    filter_size = 11  # Filter size for Gaussian blur
    filter_sigma = 1.5  # Filter sigma for Gaussian blur

    # Split the batch into individual images
    data_scaled_list = tf.split(data_scaled, num_or_size_splits=data.shape[0], axis=0)
    outputs_scaled_list = tf.split(outputs_scaled, num_or_size_splits=outputs.shape[0], axis=0)

    # Initialize a list to store the SSIM values for each image
    ssim_list = []

    # Loop over each pair of images
    for data_img, output_img in zip(data_scaled_list, outputs_scaled_list):
        # Remove the batch dimension from the images
        data_img = tf.squeeze(data_img, axis=0)
        output_img = tf.squeeze(output_img, axis=0)

        # Add a batch dimension to the images
        data_img = tf.expand_dims(data_img, axis=0)
        output_img = tf.expand_dims(output_img, axis=0)

        # Add a channel dimension to the images
        data_img = tf.expand_dims(data_img, axis=-1)
        output_img = tf.expand_dims(output_img, axis=-1)

        # Calculate SSIM between original and distorted images
        ssim = ssim_per_channel(data_img, output_img, max_val=max_val)

        # Append the SSIM value to the list
        ssim_list.append(ssim)

    # Stack the SSIM values into a tensor
    ssim_tensor = tf.stack(ssim_list, axis=0)

    # Load a VGG-19 model from Keras applications
    vgg_model = vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(target_size[0], target_size[1], 3))

    # Generate features maps for original and distorted images using VGG-19
    vgg_data = vgg_model(data_scaled)
    vgg_outputs = vgg_model(outputs_scaled)

    # Convert VGG-19 feature maps to grayscale
    vgg_data_gray = tf.image.rgb_to_grayscale(vgg_data)
    vgg_outputs_gray = tf.image.rgb_to_grayscale(vgg_outputs)

    # Convert original and distorted images to grayscale
    data_gray = tf.image.rgb_to_grayscale(data_scaled)
    outputs_gray = tf.image.rgb_to_grayscale(outputs_scaled)

    # Calculate gradient magnitude of original and distorted images
    grad_data = sobel_edges_rgb(data_gray)
    grad_outputs = sobel_edges_rgb(outputs_gray)

    # Calculate gradient magnitude similarity between original and distorted images
    grad_ssim = ssim_per_channel(grad_data, grad_outputs, max_val=max_val)

    # Reshape or transpose VGG-19 feature maps to match SSIM tensor shape
    vgg_data_gray_reshaped = tf.reshape(vgg_data_gray, [vgg_data_gray.shape[0], -1])
    vgg_outputs_gray_reshaped = tf.reshape(vgg_outputs_gray, [vgg_outputs_gray.shape[0], -1])
    
    # Reshape or transpose SSIM tensor to match VGG-19 feature maps shape
    ssim_tensor_reshaped = tf.reshape(ssim_tensor, [ssim_tensor.shape[0], -1])
    
    # Transpose VGG-19 feature maps to match SSIM tensor shape
    vgg_data_gray_transposed = tf.transpose(vgg_data_gray_reshaped)
    
    # Transpose SSIM tensor to match VGG-19 feature maps shape
    ssim_tensor_transposed = tf.transpose(ssim_tensor_reshaped)
    
    # Weight SSIM and gradient magnitude similarity by VGG-19 feature maps
    vsi = tf.linalg.matmul(ssim_tensor_transposed, (vgg_data_gray_transposed + vgg_outputs_gray_reshaped)) + tf.linalg.matmul(grad_ssim, (vgg_data_gray_transposed + vgg_outputs_gray_reshaped))

    # Expand the dimensions to have shape (batch_size, height, width, 1)
    similarity_map = tf.expand_dims(vsi, axis=-1)

    return similarity_map


# Generic implementation of the dissimilarity and contrast measures
def compute_dissimilarity(img1, img2):
    diff = tf.abs(img1 - img2)
    dissimilarity = tf.reduce_mean(diff)
    return dissimilarity

def compute_chromatic_dissimilarity(img1, img2):
    diff = tf.abs(img1 - img2)
    chromatic_dissimilarity = tf.reduce_mean(diff)
    return chromatic_dissimilarity

def compute_structural_contrast(img):
    mean_value = tf.reduce_mean(img)
    diff = tf.abs(img - mean_value)
    structural_contrast = tf.reduce_mean(diff)
    return structural_contrast

def compute_chromatic_contrast(img):
    mean_value = tf.reduce_mean(img)
    diff = tf.abs(img - mean_value)
    chromatic_contrast = tf.reduce_mean(diff)
    return chromatic_contrast
# ... (previous code) ...

def compute_structural_dissimilarity(data, outputs):
    # Compute the squared difference between data and outputs
    squared_diff = tf.square(data - outputs)

    # Compute the mean squared difference over the color channels
    msd = tf.reduce_mean(squared_diff, axis=-1)

    # Compute the structural dissimilarity as the average over all pixels
    structural_dissimilarity = tf.reduce_mean(msd)

    return structural_dissimilarity


# ... (previous code) ...




import tensorflow as tf
from keras.applications import vgg19

# Rest of the functions compute_dissimilarity, compute_chromatic_dissimilarity, compute_structural_contrast, and compute_chromatic_contrast remain the same as before

import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2

def mdsi_similarity_with_mobilenetv2(data, outputs):
    # Normalize images to [0, 1]
    data = data / 255.0
    outputs = outputs / 255.0

    # Compute structural similarity at multiple scales
    scales = [1, 2, 3, 4, 5]  # Adjust the scales as needed
    similarity_maps = []
    for scale in scales:
        target_size = [data.shape[1] // scale,
                       data.shape[2] // scale]
        data_scaled = tf.image.resize(data,
                                      target_size,
                                      method=tf.image.ResizeMethod.BILINEAR)
        outputs_scaled = tf.image.resize(outputs,
                                         target_size,
                                         method=tf.image.ResizeMethod.BILINEAR)
        diff = data_scaled - outputs_scaled
        similarity_map = tf.reduce_mean(diff * diff,
                                        axis=-1,
                                        keepdims=True)
        similarity_maps.append(similarity_map)

    # Compute structural and chromatic dissimilarity measures for data and outputs
    Ds_list = []
    Dc_list = []
    for i in range(len(scales)):
        target_size = [data.shape[1] // scales[i],
                       data.shape[2] // scales[i]]
        data_scaled = tf.image.resize(data,
                                      target_size,
                                      method=tf.image.ResizeMethod.BILINEAR)
        outputs_scaled = tf.image.resize(outputs,
                                         target_size,
                                         method=tf.image.ResizeMethod.BILINEAR)
        Ds = compute_structural_dissimilarity(data_scaled,
                                              outputs_scaled)
        Dc = compute_chromatic_dissimilarity(data_scaled,
                                             outputs_scaled)
        Ds_list.append(Ds)
        Dc_list.append(Dc)

    # Compute structural and chromatic contrast measures for data and outputs
    Cs_list = []
    Cc_list = []
    for i in range(len(scales)):
        target_size = [data.shape[1] // scales[i],
                       data.shape[2] // scales[i]]
        data_scaled = tf.image.resize(data,
                                      target_size,
                                      method=tf.image.ResizeMethod.BILINEAR)
        Cs = compute_structural_contrast(data_scaled)
        Cc = compute_chromatic_contrast(data_scaled)
        Cs_list.append(Cs)
        Cc_list.append(Cc)

    # Combine dissimilarity and contrast measures according to the MDSI formula
    mdsi_values = []
    for i in range(len(scales)):
        s = scales[i]
        N = data.shape[1] * data.shape[2]
        eps = s * N * 0.01
        Ds = Ds_list[i]
        Dc = Dc_list[i]
        Cs = Cs_list[i]
        Cc = Cc_list[i]
        mdsi_value = 2 / (1 + Ds + Cs / (Ds + eps) + Dc + Cc / (Dc + eps))
        # Expand the dimensions of mdsi_value to match the rank of the other tensors
        mdsi_value = tf.expand_dims(tf.expand_dims(mdsi_value, axis=-1), axis=-1)
        mdsi_values.append(mdsi_value)

    # Resize data and outputs to match the size of MobileNetV2 input
    mobilenet_target_size = [32, 32]  # Adjust this to the desired size for MobileNetV2
    data_resized = tf.image.resize(data,
                                   mobilenet_target_size,
                                   method=tf.image.ResizeMethod.BILINEAR)
    outputs_resized = tf.image.resize(outputs,
                                      mobilenet_target_size,
                                      method=tf.image.ResizeMethod.BILINEAR)

    # Create MobileNetV2 model
    mobilenet_model = MobileNetV2(weights='imagenet',
                                  include_top=False,
                                  input_shape=(mobilenet_target_size[0],
                                               mobilenet_target_size[1], 3))
    mobilenet_model.trainable = False  # Freeze the MobileNetV2 weights

    # Preprocess the data and outputs to fit MobileNetV2 input requirements
    data_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
        data_resized)
    outputs_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
        outputs_resized)

    # Extract MobileNetV2 features for data and outputs
    data_features = mobilenet_model(data_preprocessed)
    outputs_features = mobilenet_model(outputs_preprocessed)

    # Resize similarity maps to match the size of MobileNetV2 features
    similarity_maps_resized = [
        tf.image.resize(sim_map,
                        mobilenet_target_size,
                        method=tf.image.ResizeMethod.BILINEAR)
        for sim_map in similarity_maps
    ]

    # Repeat the similarity maps along the channel dimension to match the number of channels of the other tensors
    similarity_maps_resized = [
        tf.tile(sim_map, [1, 1, 1, 1280])
        for sim_map in similarity_maps_resized
    ]

    # Concatenate the MobileNetV2 features, MDSI values, and resized similarity maps along the channel dimension
    combined_features = tf.concat(
        [data_features, outputs_features] + mdsi_values +
        similarity_maps_resized,
        axis=-1)

    return combined_features
