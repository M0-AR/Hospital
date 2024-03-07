from moviepy.editor import VideoFileClip

# Replace 'input_video.mp4' with the path to your video file
input_video_path = './AI_3D_Bladder.mp4'
output_video_path = './AI_3D_Bladder_Demo.mp4'

# Set this to False if you want to remove audio
include_audio = False # or False to exclude audio

# Load the video file
clip = VideoFileClip(input_video_path)

# Target size in bytes (25 MB)
target_size = 23 * 1024 * 1024

# Calculate the bitrate needed to achieve the target size
# Subtract audio bitrate if you plan to keep the audio
audio_bitrate = 128000  # A typical audio bitrate in bits per second
if include_audio:
    target_video_bitrate = int((target_size * 8 - audio_bitrate * clip.duration) / clip.duration)
else:
    target_video_bitrate = int((target_size * 8) / clip.duration)

# Write the video file with the new bitrate and optionally without audio
clip.write_videofile(
    output_video_path,
    bitrate=f'{target_video_bitrate}',
    audio=include_audio
)

# Release the clip to free up system resources
clip.close()
