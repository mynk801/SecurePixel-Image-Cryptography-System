from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import numpy as np
import random
import pandas as pd 
import PIL
import os
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from math import log
from tqdm import tqdm
import warnings
import threading
from werkzeug.utils import secure_filename
warnings.filterwarnings("ignore")

app = Flask(__name__, template_folder='templates')

BASE_IMAGE_FOLDER = 'X:/Minor-2/Frontend/templates/images/1'
TARGET_IMAGE_FOLDER = 'X:/Minor-2/Frontend/templates/images/2'
ENCRYPTED_IMAGE_FOLDER = 'X:/Minor-2/Frontend/templates/images/encrypted'
def henonMapEnc(image,x0,y0):
    im = Image.open(image)
    px = im.load()
    w, h = im.size

    a = 1.4
    b = 0.3
    ncoords = []

    for i in range(0, h * w):
        x = 1 - 1.4 * pow(x0, 2) + y0 
        y = 0.3 * x0 

        xr = int(('%.12f' % (x))[5:9]) % w

        yr = int(('%.12f' % (y))[5:9]) % h

        ncoords.append((xr, yr))
        x0 = float('%.14f' % (x))
        y0 = float('%.14f' % (y))

    ncoords.reverse()

    for i in range(0, h * w):
        (xr, yr) = ncoords[i]
        j = h * w - i - 1
        (xr, yr) = ncoords[i]

        j = h * w - i - 1

        p = px[j % w, int(j / w)]

        pr = px[xr, yr]
        px[j % w, int(j / w)] = pr
        px[xr, yr] = p

    im.save('encrypted_target.png')

ENCRYPTEDIMAGE = 'encrypted_target.png'

def hide(image1, image2):

  img1 = cv2.imread(image1) 
  img2 = cv2.imread(image2) 
  for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
      for l in range(3):
        v1 = format(img1[i][j][l], '08b')
        v2 = format(img2[i][j][l], '08b')

        v3 = v1[:4] + v2[:4]

        img1[i][j][l]= int(v3, 2)
  cv2.imwrite('steg.png', img1)

HIDDENIMAGE = 'steg.png'

def reveal(image):
  img = cv2.imread(image)
  width = img.shape[0]
  height = img.shape[1]
  img1 = np.zeros((width, height, 3), np.uint8)
  img2 = np.zeros((width, height, 3), np.uint8)
  for i in range(width):
    for j in range(height):
      for l in range(3):
        v1 = format(img[i][j][l], '08b')
        v2 = v1[:4] + chr(random.randint(0, 1)+48) * 4
        v3 = v1[4:] + chr(random.randint(0, 1)+48) * 4

        img1[i][j][l]= int(v2, 2)
        img2[i][j][l]= int(v3, 2)

  cv2.imwrite('Base_re.png', img1)
  cv2.imwrite('Target_re.png', img2)

FINALBASE= 'Base_re.png'
FINALTARGET= 'Target_re.png'

def henonMapDec(image,x0,y0):
    im = Image.open(image)
    px = im.load()
    w, h = im.size
    
    """
    Flattening to linear using one loop
    """
    for i in range(0, h * w):
        x = 1 - 1.4 * pow(x0, 2) + y0
        y = 0.3 * x0
        x0 = float('%.14f' % (x))
        y0 = float('%.14f' % (y))
        xr = int(('%.11f' % (x))[5:9]) % w
        yr = int(('%.11f' % (y))[5:9]) % h

        p = px[i % w, int(i / w)]
        pr = px[xr, yr]
        px[i % w, int(i / w)] = pr
        px[xr, yr] = p

    im.save('decrypted_target.png', progressive=False, quality=100)

DECRYPTEDIMAGE = 'decrypted_target'
EXTENSIONDECRYPTED = '.png'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encrypt')
def encrypt():
    return render_template('encrypt.html')

@app.route('/decrypt')
def decrypt():
    return render_template('decrypt.html')


@app.route('/upload', methods=['POST'])
def upload():

    global base_image_path, target_image_path, key1, key2

    if request.method == 'POST':
        base_image = request.files['base_image']
        target_image = request.files['target_image']
        key1 = request.form['key1']
        key2 = request.form['key2']
        
        base_image_filename = secure_filename(base_image.filename)
        target_image_filename = secure_filename(target_image.filename)
        base_image_path = os.path.join(BASE_IMAGE_FOLDER, base_image_filename)
        target_image_path = os.path.join(TARGET_IMAGE_FOLDER, target_image_filename)
        base_directory = 'X:/Minor-2/Frontend/templates'
        relative_base_image_path = os.path.relpath(base_image_path, base_directory)
        relative_target_image_path = os.path.relpath(target_image_path, base_directory)
        base_image.save(base_image_path)
        target_image.save(target_image_path)
        
        return render_template('result.html', 
                               base_image_path=relative_base_image_path.replace("\\", "/"), 
                               target_image_path=relative_target_image_path.replace("\\", "/"), 
                               key1=key1, 
                               key2=key2,
                                base_image_path_for_encryption=base_image_path,
                                target_image_path_for_encryption=target_image_path)
    else:
        return render_template('result.html')

@app.route('/encrypt_image', methods=['POST'])
def encrypt_image():
    data = request.json
    key1 = float(data['key1'])
    key2 = float(data['key2'])
    
    henonMapEnc(target_image_path, key1, key2)
    hide(base_image_path, ENCRYPTEDIMAGE)

    return jsonify({'message': 'Encryption completed successfully'})


@app.route('/decrypt_image', methods=['POST'])
def decrypt_image():
    global encrypted_image_path
    if request.method == 'POST':
        encrypted_image = request.files['encrypted_image']
        key1 = request.form['key1']
        key2 = request.form['key2']
        
        encrypted_image_filename = secure_filename(encrypted_image.filename)
        encrypted_image_path = os.path.join(ENCRYPTED_IMAGE_FOLDER, encrypted_image_filename)
        encrypted_image.save(encrypted_image_path)

        base_directory = 'X:/Minor-2/Frontend/templates'
        relative_encrypted_image_path = os.path.relpath(encrypted_image_path, base_directory)

        return render_template('decrypt_result.html', 
                               encrypted_image_path=relative_encrypted_image_path.replace("\\", "/"), 
                               key1=key1, 
                               key2=key2)

@app.route('/image_decrypt', methods=['POST'])
def image_decrypt():
    data = request.json
    key1 = float(data['key1']) 
    key2 = float(data['key2']) 
    
    reveal(encrypted_image_path)
    henonMapDec(FINALTARGET, key1, key2)

    return jsonify({'message': 'Encryption completed successfully'})


if __name__ == '__main__':
    app.run(debug=True)
