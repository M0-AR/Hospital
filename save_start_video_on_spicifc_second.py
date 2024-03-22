import cv2

def convert_video(start_time_seconds, source_path, output_path):
    # Open the source video
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print("Error: Could not open source video.")
        return

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Calculate the starting frame number
    start_frame = int(fps * start_time_seconds)
    # Set the current frame position to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # cap.set(cv2.CAP_PROP_POS_MSEC, START_SECOND * 1000)  # Set the start position in ms
    
    # Get the width and height of the video frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create a VideoWriter object to write our new video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can change *'mp4v' with other codecs if needed
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Read from the source video and write to the output until the end of the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of source video
        out.write(frame)  # Write frame to output video
    
    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()
    print("Video conversion completed.")

# Configuration
start_time_seconds = 115  # Time in seconds to start the new video from
source_path = 'C:/src/video_detection/Ta.mp4'  # Path to the source video
output_path = 'C:/src/video_detection/Ta115.mp4'  # Path for the new video

# Convert the video
convert_video(start_time_seconds, source_path, output_path)
