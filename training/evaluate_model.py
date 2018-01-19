# Based on code from https://github.com/Microsoft/CNTK

from cntk.ops.functions import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import xml.etree.cElementTree as et
from xml.dom.minidom import parse

tree = et.parse(os.path.join('..', 'data', 'images', '64x64', 'mean_image.xml'))
root = tree.getroot()
Ch = int(root.findall(".//Channel")[0].text)
H = int(root.findall(".//Row")[0].text)
W = int(root.findall(".//Col")[0].text)
mean_image = np.array(root.findall(".//*/data")[0].text.split())
mean_imape = mean_image.astype(np.float)
mean_image = np.asarray(mean_image, dtype=np.float32)
mean_image = np.reshape(mean_image, (Ch, H, W))
mean_image = np.transpose(mean_image, (1, 2, 0))

mystery = np.asarray(Image.open(os.path.join('..', 'data', 'images', '64x64', 'shooting-6', 'benny_0010.jpg')), dtype=np.float32) - mean_image
bgr_image = mystery[..., [2, 1, 0]]
pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

plt.imshow((mystery + mean_image)/ 255.0)
plt.axis('off')
plt.show()

pred = load_model('NoeliniModel.dnn')
predicted_label_prob = np.squeeze(pred.eval({pred.arguments[0]:[pic]}))
print(predicted_label_prob)

label_lookup = ["bella", "benny", "emilie", "flurina", "julie", "kira", "klaus", "lino", "louis", "ole", "pat", "remy", "rosa", "stella", "void"]
gtlabel = np.argmax(predicted_label_prob)
print(label_lookup[gtlabel])
