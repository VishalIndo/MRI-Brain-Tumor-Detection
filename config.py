import torch

# ------------------ Paths ------------------
TRAIN_DIR = '/content/gdrive/MyDrive/archive (3)/train'
VALID_DIR = '/content/gdrive/MyDrive/archive (3)/valid'
TEST_DIR = '/content/gdrive/MyDrive/archive (3)/test'
PREPROCESSED_DIR = '/content/gdrive/MyDrive/archive (3)/preprocessed_images'
BEST_MODEL_PATH = '/content/best_fasterrcnn.pth'

# ------------------ Device ------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------ Classes ------------------
CLASS_MAPPING = {
    'glioma_tumor': 1,
    'meningioma_tumor': 2,
    'pituitary_tumor': 3,
}
REV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
NUM_CLASSES = 4  # 3 classes + background

# ------------------ Thresholds ------------------
CONFIDENCE_THRESHOLD = 0.7
IOU_THRESHOLD = 0.75

# ------------------ Training ------------------
BATCH_SIZE = 4
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20
PATIENCE = 10
