import cv2
from ultralytics import YOLO

def detect_objects_on_image(buf, model, conf_thres=0.25):
    results = model.predict(buf)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        if prob >= conf_thres:  # Apply confidence threshold
            output.append([x1, y1, x2, y2, result.names[class_id], prob])
    return output


def main():
    VIDEO_NAME = 'C:/Users/UKF/Desktop/Ta.mp4' 
    video = cv2.VideoCapture(VIDEO_NAME)

    START_SECOND = 180  # Specify the start second
    
    video.set(cv2.CAP_PROP_POS_MSEC, START_SECOND * 1000)  # Set the start position in ms

    # Load model once outside the loop
    model = YOLO("best.pt")

    # Reduce frame size for faster processing
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    new_width = int(frame_width * 0.5)
    new_height = int(frame_height * 0.5)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print('Reached the end of the video!')
            break
        
        # Reduce frame size
        frame = cv2.resize(frame, (new_width, new_height))

        detected_objects = detect_objects_on_image(frame, model, conf_thres=0.70)
        
        for det_obj in detected_objects:
            x1, y1, x2, y2, obj_type, prob = det_obj
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{obj_type}: {prob*100:.2f}%"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()