import os
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter


def load_label_map(label_path):
    """
    Load labels from the specified path.

    Args:
    - label_path (str): Path to the label map file.

    Returns:
    - List of labels.
    """
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def process_frame(frame, width, height, floating_model):
    """
    Preprocess the video frame for TensorFlow Lite model.

    Args:
    - frame (numpy.ndarray): Video frame.
    - width (int): Desired width for frame preprocessing.
    - height (int): Desired height for frame preprocessing.
    - floating_model (bool): Indicates if the model is floating point.

    Returns:
    - Processed input data for the TFLite model.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_mean = 127.5
        input_std = 127.5
        input_data = (np.float32(input_data) - input_mean) / input_std

    return input_data


def draw_detections(frame, boxes, classes, scores, labels, imW, imH, threshold):
    """
    Draw bounding boxes and labels on the frame.

    Args:
    - frame (numpy.ndarray): Original video frame.
    - boxes, classes, scores: Outputs from the TFLite model.
    - labels (list): List of class names.
    - imW, imH (int): Width and height of the frame.
    - threshold (float): Minimum confidence threshold for displaying detected objects.

    Returns:
    - Annotated frame.
    """
    for i in range(len(scores)):
        if threshold < scores[i] <= 1.0:
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)

            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return frame


def main():
    # Constants
    MODEL_NAME = 'Sample_TFLite_model'
    GRAPH_NAME = 'first_detect.tflite'
    LABELMAP_NAME = 'labelmap.txt'
    VIDEO_NAME = 'C:/Users/UKF/Desktop/Ta.mp4'
    MIN_CONF_THRESHOLD = 0.8 # TODO:

    # Paths
    CWD_PATH = os.getcwd()
    VIDEO_PATH = os.path.join(CWD_PATH, VIDEO_NAME)
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
    PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

    # Load labels
    labels = load_label_map(PATH_TO_LABELS)

    # Load the TFLite model and allocate tensors
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
    interpreter.allocate_tensors()

    # Model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]
    floating_model = input_details[0]['dtype'] == np.float32

    # Model version check
    outname = output_details[0]['name']
    if 'StatefulPartitionedCall' in outname:  # TF2 model
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else:  # TF1 model
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    MAX_FRAMES_TO_KEEP = 0  # Keep detections alive for 5 frames # TODO:
    recent_detections = []  # This will hold our recent detections

    video = cv2.VideoCapture(VIDEO_PATH)
    imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Jump to the desired second in the video
    desired_second = 150  # Replace with the second you want to jump to
    video.set(cv2.CAP_PROP_POS_MSEC, desired_second*1000)  # Convert seconds to milliseconds

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print('Reached the end of the video!')
            break

        # Process frame and get detections
        input_data = process_frame(frame, width, height, floating_model)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        # Add the current detections to the list of recent detections
        recent_detections.append((boxes, classes, scores))

        # Draw all recent detections
        for detection in recent_detections:
            frame = draw_detections(frame, *detection, labels, imW, imH, MIN_CONF_THRESHOLD)

        # If the list of recent detections gets too long, pop the oldest detections off
        if len(recent_detections) > MAX_FRAMES_TO_KEEP:
            recent_detections.pop(0)

        cv2.imshow('Object detector', frame)
        if cv2.waitKey(70) == ord('q'): # TODO:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
