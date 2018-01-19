import numpy as np
import os
from PIL import Image
import subprocess as sp
import shutil
import xml.etree.cElementTree as et
import xml.dom.minidom

from pathlib import Path

def extract_frames(script_path, movies_path, images_path, image_size, isTraining):
    
    if (not os.path.isdir(os.path.join(script_path, movies_path))):
        print('To extract the frames, you need the movies.')
        return
        
    os.makedirs(os.path.join(script_path, images_path))

    if (isTraining):
        image_map = open(os.path.join(Path(os.path.join(script_path, images_path)).parent, 'train_map.txt'), 'a')
    else:
        image_map = open(os.path.join(Path(os.path.join(script_path, images_path)).parent, 'test_map.txt'), 'a')

    global mean_train_image
    global train_image_count

    for filename in os.listdir(os.path.join(script_path, movies_path)):
        if filename.endswith(".mov"):
            noelini_name = filename[:-4]

            ffmpeg_exe = 'ffmpeg.exe'
            command = [ ffmpeg_exe,
                '-i', os.path.join(script_path, movies_path, filename),
                '-f', 'image2pipe',
                '-r', '10',
                '-pix_fmt', 'rgb24',
                '-vcodec', 'rawvideo', '-']

            pipe = sp.Popen(
                command,
                stdout = sp.PIPE,
                bufsize = 10**8)

            bytes_per_frame = image_size*image_size*3
            index = 0
            while (True):
                image_buffer = pipe.stdout.read(bytes_per_frame)
                image_buffer =  np.fromstring(image_buffer, dtype='uint8')
                if (image_buffer.size < bytes_per_frame):
                    break

                image_buffer = image_buffer.reshape((image_size, image_size, 3))
                       
                #plt.figure()
                #plt.imshow(image_buffer)
                #plt.show()

                image = Image.fromarray(image_buffer)
                image_filename = os.path.join(images_path, noelini_name + '_' + str(index).zfill(4) + '.jpg')
                image.save(image_filename)
                image_map.write(image_filename + '\t' + name_label_dict[noelini_name] + '\n')

                if (isTraining):
                    image_buffer = np.transpose(image_buffer, (2, 0, 1))
                    mean_train_image += image_buffer
                    train_image_count += 1
 
                index += 1

            pipe.stdout.flush()
           
    image_map.close()


def save_mean_train_image(mean_filename, mean_image):
    root = et.Element('opencv_storage')
    et.SubElement(root, 'Channel').text = '3'
    et.SubElement(root, 'Row').text = str(image_size)
    et.SubElement(root, 'Col').text = str(image_size)
    meanImg = et.SubElement(root, 'MeanImg', type_id='opencv-matrix')
    et.SubElement(meanImg, 'rows').text = '1'
    et.SubElement(meanImg, 'cols').text = str(image_size * image_size * 3)
    et.SubElement(meanImg, 'dt').text = 'f'
    et.SubElement(meanImg, 'data').text = ' '.join(['%e' % n for n in np.reshape(mean_image, (image_size * image_size * 3))])

    tree = et.ElementTree(root)
    tree.write(mean_filename)
    x = xml.dom.minidom.parse(mean_filename)
    with open(mean_filename, 'w') as f:
        f.write(x.toprettyxml(indent = '  '))
        
name_label_dict = {
    'bella' : '0',
    'benny' : '1',
    'emilie' : '2',
    'flurina' : '3',
    'julie' : '4',
    'kira' : '5',
    'klaus' : '6',
    'lino' : '7',
    'louis' : '8',
    'ole' : '9',
    'pat' : '10',
    'remy' : '11',
    'rosa' : '12',
    'stella' : '13',
    'void' : '14' }

script_path = os.path.dirname(os.path.abspath(__file__))
image_size = 64

images_path = os.path.join('..', 'data', 'images', str(image_size)+'x'+str(image_size))
if (os.path.isdir(os.path.join(script_path, images_path))):
    os.rename(os.path.join(script_path, images_path), os.path.join(Path(os.path.join(script_path, images_path)).parent, 'delete.me'))
    shutil.rmtree(os.path.join(Path(os.path.join(script_path, images_path)).parent, 'delete.me'), ignore_errors=True)

mean_train_image = np.zeros((3, image_size, image_size))
train_image_count = 0

for i in range(5):
    extract_frames(
        script_path,
        os.path.join('..', 'data', 'movies', str(image_size)+'x'+str(image_size), 'shooting-' + str(i+1)),
        os.path.join('..', 'data', 'images', str(image_size)+'x'+str(image_size), 'shooting-' + str(i+1)),
        image_size,
        True)

extract_frames(
    script_path,
    os.path.join('..', 'data', 'movies', str(image_size)+'x'+str(image_size), 'shooting-6'),
    os.path.join('..', 'data', 'images', str(image_size)+'x'+str(image_size), 'shooting-6'),
    image_size,
    False)

save_mean_train_image(
    os.path.join(script_path, images_path, 'mean_image.xml'),
    mean_train_image / train_image_count)


