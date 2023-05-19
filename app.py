import gradio as gr
from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import  img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import tensorflow as tf
from colorizers import *
from colorizers import model_building as mb

def my_func_predict(img):
	colorizer_model = mb.colorize_model(pretrained=True).eval()
	(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
	out_img = postprocess_tens(tens_l_orig, colorizer_model(tens_l_rs).cpu())
	return out_img

Title="""
	<div style="text-align: center; max-width: 650px; margin: 0 auto;">
        <div
        style="
            display: inline-flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.75rem;
        "
        >
        <h1 style="font-weight: 900; margin-bottom: 7px;">
            Team -14 SDM
        </h1>
        </div>
        <p style="margin-bottom: 10px; font-size: 94%">
        Image Colorization Project using CNN
        </p>
    </div>
	"""
Article="This is our presentation demosite, Check out our other works [here](https://hemanth-thaluru.github.io/portfolio/)."

iface = gr.Interface(
						fn=my_func_predict, 
						inputs=gr.Image(shape=(256,256)), 
						outputs='image',
						title=Title, 
						article=Article, 
						examples=['Flower.jpeg','Kid.jpeg','Place.jpeg']
					)
iface.launch()