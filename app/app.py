### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict
from calories_protein import nutrition_db

# Setup class names
with open("class_names.txt", "r") as f: # reading them in from class_names.txt
    class_names = [food_name.strip() for food_name in  f.readlines()]
    
### 2. Model and transforms preparation ###    

# Create model
effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes=101, # could also use len(class_names)
)

# Load saved weights
effnetb2.load_state_dict(
    torch.load(
        f="09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)

### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float, str]:
    # Start the timer
    start_time = timer()
    
    # Transform and add batch dimension
    img = effnetb2_transforms(img).unsqueeze(0)
    
    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Get top prediction
    top_pred = max(pred_labels_and_probs, key=pred_labels_and_probs.get).lower()
    
    # Lookup nutrition info
    nutrients = nutrition_db.get(
        top_pred,
        {"calories": "N/A", "protein": "N/A"}
    )
    nutrition_text = (
        f"Calories: {nutrients['calories']}\n"
        f"Protein: {nutrients['protein']}\n"
    )
    
    # Prediction time
    pred_time = round(timer() - start_time, 5)
    
    return pred_labels_and_probs, pred_time, nutrition_text


### 4. Gradio app ###

# Create title, description and article strings
title = "NutriNet🍲"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food into [101 different classes](https://github.com/omkar00004/NutriNet-Vision-Transformer-ViT-/blob/main/extras/food101_class_names.txt)."

# Create examples list from "examples/" directory
example_list = [
    ["examples/" + example]
    for example in os.listdir("examples")
    if not example.startswith(".")  # ignore hidden files
]

# Create Gradio interface 
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
        gr.Textbox(label="Nutrients & Calories")   # 👈 new output box
    ],
    examples=example_list,
    title=title,
    description=description,
)


# Launch the app!
demo.launch()
