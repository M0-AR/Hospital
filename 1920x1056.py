import cv2
from ultralytics import YOLO
import os
import time
import torch
import csv
import logging
import matplotlib.pyplot as plt

class_dict = {"Suspected_Region": 0, "Another_Class": 1}  # Add all your classes here

def detect_objects_on_image(image, model, conf_thres=0.25):
    # Ensure the image dimensions are divisible by 32
    h, w, _ = image.shape
    h = h - h % 32
    w = w - w % 32
    image = cv2.resize(image, (w, h))

    # Convert to tensor and move to GPU
    image_cuda = torch.from_numpy(image).to('cuda').float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    results = model.predict(image_cuda)


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
    img_name = os.path.join(save_path, f"{frame_number}.png")
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
        cropped_img_name = os.path.join(save_path, f"{frame_number}_{class_id}_crop_{prob}.png")
        cv2.imwrite(cropped_img_name, cropped_img)

# Set up logging
logging.basicConfig(filename='auto_anotation.log', level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')

def main(VIDEO_NAME, detection_statistics):
    video_basename = os.path.basename(VIDEO_NAME)
    video_name_without_extension = os.path.splitext(video_basename)[0]
    
    SAVE_PATH = 'E:/Mohamad_Temp/auto_annotation/1280x720_annotation/'
    SAVE_PATH = 'F:/Mohamad_Temp/1920x1056_annotation/'
    SAVE_PATH = os.path.join(SAVE_PATH, video_name_without_extension) 
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    
    try:
        cap = cv2.VideoCapture(VIDEO_NAME)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened():
            logging.error(f"Error opening video: {VIDEO_NAME}")
            return

        model = YOLO("best.pt")

        local_detection_count = 0
        start_time = time.time()

        for frame_number in range(total_frames):
            ret, frame = cap.read()

            if not ret:
                # logging.warning(f"Failed to read frame {frame_number} of video {VIDEO_NAME}. Skipping...")
                continue

            detection = detect_objects_on_image(frame, model, 0.3)
            if detection:
                local_detection_count += len(detection)
                save_image_and_annotation(frame, detection, SAVE_PATH, frame_number)

    except Exception as e:
        # logging.error(f"Error processing video {VIDEO_NAME}: {e}")
        print(f"Error processing video {VIDEO_NAME}: {e}")
    finally:
        # Ensure that the video capture is always released
        cap.release()
    
    video_duration = time.time() - start_time

    detection_statistics['total_detections'] += local_detection_count
    detection_statistics['total_videos'] += 1
    detection_statistics['total_time'] += video_duration
    detection_statistics['detections_per_video'].append(local_detection_count)
    detection_statistics['time_per_video'].append(video_duration)

if __name__ == "__main__":
    start_time = time.time()

    DIRECTORY_PATH = 'C:/Users/UKF/Desktop/new_110_videos'
    DIRECTORY_PATH = 'E:/Mohamad_Temp/auto_annotation/video_with_suspected_region'
    DIRECTORY_PATH = 'F:/Mohamad_Temp/videos_with_suspected_region/'

    detection_statistics = {
        'total_detections': 0,
        'total_videos': 0,
        'total_time': 0,
        'detections_per_video': [],
        'time_per_video': []
    }

    for entry in os.scandir(DIRECTORY_PATH):
        if entry.is_file() and entry.name.endswith(('.mp4', '.avi')):
            main(entry.path, detection_statistics)
    
    # Compute the statistics
    average_detections = detection_statistics['total_detections'] / detection_statistics['total_videos']
    average_time = detection_statistics['total_time'] / detection_statistics['total_videos']

    # Printing statistics
    print(f"Total videos processed: {detection_statistics['total_videos']}")
    print(f"Total detections: {detection_statistics['total_detections']}")
    print(f"Average detections per video: {average_detections:.2f}")
    print(f"Average time per video: {average_time:.2f} seconds")
    
    # For storage space, you'd need to actually measure the size of the saved files.
    # This is just a placeholder:
    print("Total storage space needed: [compute the space based on file sizes]")

    # Plotting a bar graph for detections per video
    plt.bar(range(detection_statistics['total_videos']), detection_statistics['detections_per_video'])
    plt.xlabel('Video Index')
    plt.ylabel('Detections')
    plt.title('Detections per Video')
    plt.show()

    # Plotting a bar graph for processing time per video
    plt.bar(range(detection_statistics['total_videos']), detection_statistics['time_per_video'])
    plt.xlabel('Video Index')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Processing Time per Video')
    plt.show()

    # Compute the statistics
    average_detections = detection_statistics['total_detections'] / detection_statistics['total_videos']
    average_time = detection_statistics['total_time'] / detection_statistics['total_videos']
    
    # Save results to CSV
    with open('1920x1056_auto_annotation.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total videos processed", detection_statistics['total_videos']])
        writer.writerow(["Total detections", detection_statistics['total_detections']])
        writer.writerow(["Average detections per video", f"{average_detections:.2f}"])
        writer.writerow(["Average time per video", f"{average_time:.2f} seconds"])
        # Placeholder for storage space, compute the space based on file sizes
        writer.writerow(["Total storage space needed", "[compute the space based on file sizes]"])
        
        writer.writerow([])
        writer.writerow(["Video Index", "Detections", "Processing Time (seconds)"])
        for i in range(detection_statistics['total_videos']):
            writer.writerow([i, detection_statistics['detections_per_video'][i], detection_statistics['time_per_video'][i]])
