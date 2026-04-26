# MRI Brain Tumor Detection Project



## Project Purpose



---

## What This Project Does

This project is built for **brain tumor detection from MRI data**.  

- loading MRI images or medical image files
- preprocessing the input data
- preparing the dataset and labels
- training a detection or classification model
- evaluating the trained model
- running inference on new scans
- showing results through a Gradio web app

---

## Project Structure

```bash
mri_project_split/
│
├── main.py
├── config.py
├── backbone.py
├── dataset.py
├── train_utils.py
├── train_main.py
├── evaluate.py
├── inference.py
├── medical_utils.py
├── eda.py
├── app.py
├── requirements.txt
└── README.md
```

### File Explanation

#### `main.py`
This is the main entry point of the project.  
You can use this file to connect the major parts of the pipeline and run the workflow.

#### `config.py`
This file stores configuration values such as:
- dataset paths
- batch size
- learning rate
- number of epochs
- image size
- model paths
- class names

Keeping these values in one file makes the project easier to edit.

#### `backbone.py`
This file contains model-related code.  
If the original notebook used a custom model, a pretrained backbone, or a detection network, that logic belongs here.

#### `dataset.py`
This file handles dataset loading and preprocessing.  
It usually includes:
- reading image paths
- loading labels
- applying transforms
- preparing dataloaders

#### `train_utils.py`
This file contains training helper functions such as:
- training loop
- loss calculation
- optimizer step
- validation step
- metric tracking
- checkpoint saving

#### `train_main.py`
This file is focused on running the training process.  
Instead of keeping the whole training loop inside a notebook cell, this file makes training reusable and easier to run.

#### `evaluate.py`
This file is used for model evaluation.  
It may include:
- accuracy calculation
- precision / recall / F1 score
- confusion matrix
- validation metrics
- performance summary

#### `inference.py`
This file is for prediction on new MRI samples.  
It helps run the trained model on unseen data and generate outputs.

#### `medical_utils.py`
This file may contain helper functions related to medical image handling, such as:
- slice processing
- DICOM/NIfTI handling
- normalization
- resizing
- scan conversion

#### `eda.py`
This file is for exploratory data analysis.  
It can include:
- class distribution plots
- sample image display
- label inspection
- dataset quality checks

#### `app.py`
This file runs the **Gradio app**.  
It is used to build a simple user interface where a user can upload an MRI image or scan and get a prediction result from the model.

---


## Installation

It is recommended to create a virtual environment before installing dependencies.

### Step 1: Create virtual environment

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / Mac
```bash
python -m venv venv
source venv/bin/activate
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

---

## Required Libraries

The exact libraries depend on the notebook, but common ones may include:

- Python 3.10+
- torch
- torchvision
- numpy
- pandas
- matplotlib
- scikit-learn
- opencv-python
- pillow
- gradio
- nibabel
- pydicom

If your notebook used other libraries, add them into `requirements.txt`.

---

## How to Run the Project

### 1. Train the model
If the training pipeline is ready, run:

```bash
python train_main.py
```

or sometimes:

```bash
python main.py
```

depending on how the workflow is connected.

### 2. Evaluate the model
To evaluate the trained model:

```bash
python evaluate.py
```

### 3. Run inference on test samples
To test prediction on new MRI images:

```bash
python inference.py
```

### 4. Launch the Gradio app
To start the web application:

```bash
python app.py
```

After that, Gradio will usually provide a local link in the terminal such as:

```bash
Running on local URL:  http://127.0.0.1:7860
```

Open that link in your browser to use the application.

---

## Expected Workflow

A normal project workflow may look like this:

1. Prepare the dataset  
2. Load MRI images and labels  
3. Apply preprocessing and transforms  
4. Build or load the model  
5. Train the model  
6. Save the best checkpoint  
7. Evaluate the model  
8. Run inference on new scans  
9. Deploy the model using Gradio  

This structure supports that full pipeline.

---

## Dataset Notes

This project expects MRI data prepared in a consistent format.  
Depending on your original notebook, data may come from:

- PNG or JPG image slices
- DICOM files
- NIfTI files (`.nii` / `.nii.gz`)
- folders containing patient scans

Before running the project, make sure:

- dataset paths are correct
- labels are available
- file names match the code logic
- class names are properly defined
- train/validation/test splits are correct

You should update these values inside `config.py` or wherever the path settings are stored.



## Saving and Loading Models

During training, you should save checkpoints such as:

```bash
best_model.pth
last_model.pth
```

These checkpoints can later be loaded inside:
- `evaluate.py`
- `inference.py`
- `app.py`

This avoids retraining every time you want to test the model.

---

## Gradio App

The `app.py` file is used to create a simple interactive interface.

The Gradio app can help in the following way:

- upload MRI image or scan
- run model prediction
- display class label or output
- show confidence or detection result
- make the project easy to demonstrate to others


---


### Gradio App 


```markdown
## Gradio App Interface

(https://raw.githubusercontent.com/VishalIndo/repo/main/images/gradio_app1.png)
(https://raw.githubusercontent.com/VishalIndo/repo/main/images/gradio_app1.png)
```


---

## Recommended Folder for Images

Create one folder inside the project:

```bash
images/
```

Then save your screenshots inside it, for example:

```bash
images/training_output.png
images/evaluation_output.png
images/inference_output.png
images/gradio_app.png
```

That way the README image links will work correctly.

---





## Author

**Radha Mungara** 
**Venil Kukadiya**
**Prasiddh Dhameliya**
**Vishal Mangukiya**  

MRI Brain Tumor Detection Project

