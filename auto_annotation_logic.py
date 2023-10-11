import cv2
import torch
from ultralytics import YOLO
import os


def detect_objects_on_image(image, model, conf_thres=0.25):
    results = model.predict(image)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        if prob >= conf_thres:  # Apply confidence threshold
            output.append([x1, y1, x2, y2, result.names[class_id], prob])
    return output


def save_image_and_annotation(image, detection, save_path, frame_number):
    # Constructing paths
    img_name = os.path.join(save_path, f"{frame_number}.jpg")
    txt_name = os.path.join(save_path, f"{frame_number}.txt")
    
    # Save the image
    cv2.imwrite(img_name, image)
    
     # Save annotation
    height, width, _ = image.shape
    with open(txt_name, 'w') as f:
        for x1, y1, x2, y2, class_id, prob in detection:
            x_center, y_center, w, h = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
            f.write(f"{class_id} {x_center/width} {y_center/height} {w/width} {h/height}\n")

            # Crop and save the image if needed
            cropped_img = image[y1:y2, x1:x2]
            cropped_img_name = os.path.join(save_path, f"{frame_number}_{class_id}_crop.jpg")
            cv2.imwrite(cropped_img_name, cropped_img)

def main():
    VIDEO_NAME = 'C:/Users/UKF/Desktop/Ta.mp4' 
    SAVE_PATH = './frames'  # Update path as needed
    
    START_SECOND = 180  # Specify the start second
    
    cap = cv2.VideoCapture(VIDEO_NAME)
    cap.set(cv2.CAP_PROP_POS_MSEC, START_SECOND * 1000)  # Set the start position in ms

    # Load model once outside the loop
    model = YOLO("best.pt")

    
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detection = detect_objects_on_image(frame, model, 0.70)
        if detection:
            save_image_and_annotation(frame, detection, SAVE_PATH, frame_number)
        
        frame_number += 1
    
    cap.release()

# Running the main function
if __name__ == "__main__":
    main()
