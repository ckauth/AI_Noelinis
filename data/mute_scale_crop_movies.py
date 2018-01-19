import os
import subprocess as sp

def mute_scale_crop_movies(movies_original_path, movies_processed_path, image_size):

    if (not os.path.isdir(movies_original_path)):
        print('To preprocess the movies yourself, you need the original movie files.')
        return

    if (not os.path.isdir(movies_processed_path)):
        os.makedirs(movies_processed_path)

    for filename in os.listdir(movies_original_path):
        if filename.endswith(".mov"):
            ffmpeg_exe = 'ffmpeg.exe'
            command = [ffmpeg_exe,
                       '-y',
                       '-i', os.path.join(movies_original_path, filename),
                       '-vf', 'crop=240:240:40:0, scale='+str(image_size)+':'+str(image_size),
                       '-an', os.path.join(movies_processed_path, filename)]
            sp.call(command)

script_directory = os.path.dirname(os.path.abspath(__file__))
image_size = 64

for i in range(6):
    mute_scale_crop_movies(
        os.path.join(script_directory, 'movies', '320x240', 'shooting-' + str(i+1)),
        os.path.join(script_directory, 'movies', str(image_size)+'x'+str(image_size), 'shooting-' + str(i+1)),
        image_size)
