import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Class mapping with food items and their calories
class_mapping = {
    0: {'label': 'aloo-gobi', 'calories': 108},
    1: {'label': 'aloo-fry', 'calories': 125},
    2: {'label': 'dum-aloo', 'calories': 164},
    3: {'label': 'fish-curry', 'calories': 241},
    4: {'label': 'ghevar', 'calories': 61},
    5: {'label': 'green-chutney', 'calories': 21},
    6: {'label': 'gulab-jamun', 'calories': 145},
    7: {'label': 'idli', 'calories': 40},
    8: {'label': 'jalebi', 'calories': 150},
    9: {'label': 'chicken-seekh-kebab', 'calories': 158},
    10: {'label': 'kheer', 'calories': 266},
    11: {'label': 'kulfi', 'calories': 136},
    12: {'label': 'bhature', 'calories': 230}, 
    13: {'label': 'lassi', 'calories': 183},
    14: {'label': 'mutton-curry', 'calories': 298},
    15: {'label': 'onion-pakoda', 'calories': 80},
    16: {'label': 'palak-paneer', 'calories': 338},
    17: {'label': 'poha', 'calories': 270},
    18: {'label': 'rajma-curry', 'calories': 235},
    19: {'label': 'rasmalai', 'calories': 188},
    20: {'label': 'samosa', 'calories': 308},
    21: {'label': 'shahi-paneer', 'calories': 261},
    22: {'label': 'white-rice', 'calories': 135},
    23: {'label': 'bhindi-masala', 'calories': 225},
    24: {'label': 'chicken-biryani', 'calories': 348},
    25: {'label': 'chai', 'calories': 54},
    26: {'label': 'chole', 'calories': 311},
    27: {'label': 'coconut-chutney', 'calories': 105},
    28: {'label': 'dal-tadka', 'calories': 260},
    29: {'label': 'dosa', 'calories': 106}
}

def calculate_total_calories(class_label, count):
    class_info = class_mapping.get(class_label, {'label': 'unknown', 'calories': 0})
    calories_per_item = class_info['calories']
    total_calories = count * calories_per_item
    return total_calories

def detect_food(image, model_path="best.pt", confidence_threshold=0.25):
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Convert Gradio image to numpy array
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Make prediction
    results = model.predict(source=img, conf=confidence_threshold)
    
    # Process results
    detected_items = [0] * 30
    float_detections = results[0].boxes.xyxy.tolist()
    detections = [[int(value) for value in detection] for detection in float_detections]
    confidences = results[0].boxes.conf.tolist()
    float_classes = results[0].boxes.cls.tolist()
    classes = [int(value) for value in float_classes]
    
    total_calories = 0
    result_img = img.copy()
    
    # Draw detections on the image
    for i in range(len(detections)):
        box = detections[i]
        class_index = classes[i]
        class_info = class_mapping.get(class_index, {'label': 'unknown', 'calories': 0})
        conf = confidences[i]
        
        if conf > 0.4:
            detected_items[class_index] += 1
            label = f"{class_info['label']} ({class_info['calories']} kcal) {conf:.2f}"
            
            # Draw rectangle and label
            cv2.rectangle(result_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(result_img, label, (box[0], box[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Calculate total calories
    items_with_calories = []
    for i in range(30):
        if detected_items[i] > 0:
            item_cal = class_mapping[i]['calories'] * detected_items[i]
            items_with_calories.append({
                'Food': class_mapping[i]['label'],
                'Count': detected_items[i],
                'Calories per item': class_mapping[i]['calories'],
                'Total calories': item_cal
            })
            total_calories += item_cal
    
    # Convert result image back to RGB for Gradio
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    # Create a summary string
    summary = f"Total estimated calories: {total_calories}"
    
    # Convert items to DataFrame for better display
    if items_with_calories:
        import pandas as pd
        df = pd.DataFrame(items_with_calories)
        return result_img, summary, df
    else:
        return result_img, "No food items detected.", None

# Create Gradio interface
demo = gr.Interface(
    fn=detect_food,
    inputs=gr.Image(type="pil", label="Upload Food Image"),
    outputs=[
        gr.Image(label="Detected Food"),
        gr.Textbox(label="Summary"),
        gr.Dataframe(label="Detected Items", headers=["Food", "Count", "Calories per item", "Total calories"])
    ],
    title="🍽️ Food Recognition & Calorie Estimation",
    description="Upload an image of food to detect items and estimate calories.",
    examples=[["example_images/example1.jpg"], ["example_images/example2.jpg"]],
    allow_flagging="never"
)

if __name__ == "__main__":
    # Create example_images directory if it doesn't exist
    os.makedirs("example_images", exist_ok=True)
    demo.launch(share=True)
