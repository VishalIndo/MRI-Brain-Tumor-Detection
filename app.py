import os
import shutil
import cv2
import gradio as gr
import numpy as np
import torch
from torchvision import transforms as T

from config import DEVICE, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, CLASS_MAPPING, BEST_MODEL_PATH
from backbone import build_model
from medical_utils import is_dicom_folder, is_nifti_file, is_image_file, convert_dicom_to_images, convert_nifti_to_images

REV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}


class ModelWrapper:
    def __init__(self, weights_path=BEST_MODEL_PATH):
        self.model = build_model(weights=None)
        self.model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()

    def predict_image(self, image_path):
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = T.ToTensor()(rgb).to(DEVICE)
        with torch.no_grad():
            output = self.model([tensor])[0]

        for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
            if score.item() < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            text = f"{REV_CLASS_MAPPING.get(label.item(), 'unknown')}: {score.item():.2f}"
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rgb, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return rgb


wrapper = None


def predict_on_folder(folder_path):
    global wrapper
    if wrapper is None:
        wrapper = ModelWrapper()

    outputs = []
    for fname in sorted(os.listdir(folder_path)):
        if is_image_file(fname):
            full_path = os.path.join(folder_path, fname)
            outputs.append(wrapper.predict_image(full_path))
    return outputs[0] if outputs else None


def process_upload(uploaded_path):
    temp_dir = 'converted_slices'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    if os.path.isdir(uploaded_path) and is_dicom_folder(uploaded_path):
        converted = convert_dicom_to_images(uploaded_path, temp_dir)
        return predict_on_folder(converted)
    elif os.path.isfile(uploaded_path) and is_nifti_file(uploaded_path):
        converted = convert_nifti_to_images(uploaded_path, temp_dir)
        return predict_on_folder(converted)
    elif os.path.isfile(uploaded_path) and is_image_file(uploaded_path):
        global wrapper
        if wrapper is None:
            wrapper = ModelWrapper()
        return wrapper.predict_image(uploaded_path)
    else:
        raise ValueError('Unsupported file or folder format.')


iface = gr.Interface(
    fn=process_upload,
    inputs=gr.File(type='filepath', label='Upload DICOM folder, NIfTI file, or image'),
    outputs=gr.Image(type='numpy', label='Prediction'),
    title='MRI Brain Tumor Detection',
    description='Upload a DICOM folder, NIfTI file, or standard image for tumor detection.',
)


if __name__ == '__main__':
    iface.launch(debug=True)
