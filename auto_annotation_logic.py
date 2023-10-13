import cv2
from ultralytics import YOLO
import os
import time

class_dict = {"Suspected_Region": 0, "Another_Class": 1}  # Add all your classes here


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
        for x1, y1, x2, y2, class_label, prob in detection:
            x_center, y_center, w, h = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
            class_id = class_dict[class_label]
            f.write(f"{class_id} {x_center/width} {y_center/height} {w/width} {h/height}\n")

    # Testing Phase
    # Read bounding box information from file
    with open(txt_name, 'r') as f:
        line = f.readline().strip()
        class_id, x_center, y_center, w, h = map(float, line.split())

        # Denormalize coordinates
        x_center = x_center * width
        y_center = y_center * height
        w = w * width
        h = h * height

        # Compute bounding box coordinates
        x1 = int(x_center - w/2)
        x2 = int(x_center + w/2)
        y1 = int(y_center - h/2)
        y2 = int(y_center + h/2)

        # Crop and save the image if needed
        cropped_img = image[y1:y2, x1:x2]
        cropped_img_name = os.path.join(save_path, f"{frame_number}_{class_id}_crop.jpg")
        cv2.imwrite(cropped_img_name, cropped_img)

def main():
    VIDEO_NAME = 'C:/Users/UKF/Desktop/Ta.mp4' 
    SAVE_PATH = './frames'  # Update path as needed
    os.mkdir(SAVE_PATH)
    
    START_SECOND = 180  # Specify the start second
    
    cap = cv2.VideoCapture(VIDEO_NAME)
    # cap.set(cv2.CAP_PROP_POS_MSEC, START_SECOND * 1000)  # Set the start position in ms

    # Load model once outside the loop
    model = YOLO("best.pt")

    
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detection = detect_objects_on_image(frame, model, 0.9)
        if detection:
            save_image_and_annotation(frame, detection, SAVE_PATH, frame_number)
        
        frame_number += 1
    
    cap.release()

# Running the main function
if __name__ == "__main__":
    start_time = time.time()
    # The main function took 45 minutes and 34 seconds to run. 80%
    # The main function took 46 minutes and 39 seconds to run. 90%
    main()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"The main function took {minutes} minutes and {seconds} seconds to run.")