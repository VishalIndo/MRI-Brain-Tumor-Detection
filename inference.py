import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T

from config import DEVICE, CLASS_MAPPING, BEST_MODEL_PATH, CONFIDENCE_THRESHOLD
from backbone import build_model

IDX_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}


def load_trained_model(weights_path=BEST_MODEL_PATH):
    model = build_model(weights=None)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict_single_image(image_path, model=None, confidence_threshold=CONFIDENCE_THRESHOLD):
    if model is None:
        model = load_trained_model()

    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = T.ToTensor()(rgb).to(DEVICE)

    with torch.no_grad():
        output = model([tensor])[0]

    pred_image = rgb.copy()
    for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
        if score.item() < confidence_threshold:
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        cls = IDX_TO_CLASS.get(label.item(), 'unknown')
        cv2.rectangle(pred_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            pred_image,
            f'{cls}: {score.item():.2f}',
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    plt.figure(figsize=(10, 8))
    plt.imshow(pred_image)
    plt.axis('off')
    plt.title('Prediction Result')
    plt.show()
    return pred_image
