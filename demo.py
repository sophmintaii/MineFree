import cv2
import torch
import numpy as np
import streamlit as st
import rasterio as rio
from torch import tensor

from classification_task import ClassificationTask
from metrics import TPR_at_FPR

from torchvision import transforms


def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def stretch_to_min_max(image):
    min_percent = 2
    max_percent = 98
    lo, hi = np.nanpercentile(image, (min_percent, max_percent))

    res_img = (image.astype(float) - lo) / (hi - lo)

    return np.maximum(np.minimum(res_img * 255, 255), 0).astype(np.uint8)


def minmax_stretch(image):
    new_image = []
    for channel in range(image.shape[2]):
        channel_values = image[:, :, channel]
        new_image.append((stretch_to_min_max(channel_values)))
    new_image = np.dstack(new_image)
    return new_image


def classify_image(image, model_path="../model/resnet50-demo.pt"):
    overlay = np.zeros(image.shape)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(64),
        transforms.Normalize([0.1930, 0.1665, 0.1089], [0.1352, 0.1172, 0.0753])
    ])

    model = torch.load(model_path)
    model.eval()

    for i in range(0, image.shape[0] // 256):
        for j in range(0, image.shape[1] // 256):
            x = i * 256
            y = j * 256
            end_x = min(x + 256, image.shape[0])
            end_y = min(y + 256, image.shape[1])
            end_x = image.shape[0] if image.shape[0] - end_x < 256 else end_x
            end_y = image.shape[1] if image.shape[1] - end_y < 256 else end_y

            curr_part = image[x:end_x, y:end_y, :]
            if np.any(curr_part):
                curr_part = transform(curr_part).unsqueeze(0)
                print(curr_part.max())
                output = torch.argmax(model(curr_part), dim=1).tolist()[0]
                overlay[x:end_x, y:end_y, :] = [1. * (1 - output), 1. * output, 0.]
    overlay = cv2.convertScaleAbs(overlay, alpha=255.)

    combined = cv2.addWeighted(image, 0.4, overlay.astype('uint8'), 0.1, 0)
    return combined


uploaded = st.file_uploader("Upload a TIFF file", type=["tif"])

if uploaded:
    image = rio.open(uploaded).read().transpose((1, 2, 0))
    image = np.delete(image, [3], axis=2)[:, :, :]
    st.write("Input image:")
    image = ((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype('uint8')
    image = enhance_image(image)
    st.image(image)
    # image = minmax_stretch(image)
    # st.image(image)
    st.write("Classification in progress...")
    st.image(classify_image(image))
