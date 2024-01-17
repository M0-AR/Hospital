# from moviepy.editor import VideoFileClip
# from tkinter.filedialog import *

# video_file = askopenfilename()
# clip = VideoFileClip(video_file)
# clip.write_gif("ouput.gif", fps=10)

from moviepy.editor import VideoFileClip

# video_filename = './test.mp4'
video_filename = 'C:\\Users\\UKF\\Desktop\\SR.mp4'
output_gif_filename = 'sr.gif'

# Load the video file
clip = VideoFileClip(video_filename)

# Optional: Resize the clip to reduce the dimensions, here we resize to half of the original dimensions
clip_resized = clip.resize(0.5)

# Write the GIF file with reduced fps and resized clip
clip_resized.write_gif(output_gif_filename, fps=3)  # Reduced fps to 5
