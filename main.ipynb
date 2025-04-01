import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns; sns.set_theme()
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow import keras
from google.colab import drive
from google.colab import files
from google.colab.patches import cv2_imshow
# from keras import backend as K
from tensorflow.keras import backend as K
import cv2

%tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name)) # remember to activate GPU

from google.colab import drive
drive.mount('/content/drive')
google_drive_path = "/content/drive/My Drive/Colab Notebooks/"

# creating functions to measure frame quality

# 1) PSNR (Peak Signal-to-Noise Ratio)
# The higher the PSNR value, the closer the compressed or predicted image is to
# the original one. PSNR = 10⋅log(Imax^2 / MSE) where
# MSE: The mean squared error between the original and predicted image.
# Imax: The maximum pixel value, typically 255 for 8-bit images or 1.0 for normalized images for 0-1 pixel values.
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1))))

# 2) SSIM (Structural Similarity Index)
# calculates the SSIM between two images, which is a better measure of image
# similarity than MSE, as it accounts for structural, contrast, and luminance
# differences that align more closely with human perception.
# If score is close to 1, images are very similar.
def SSIM(y_true, y_pred):
    return tf.image.ssim(y_true,y_pred,max_val=1)
    
# This function trains a given model with specified data and parameters,
# returning historical metrics like loss, PSNR, and SSIM.
def run_model_iteration( model, x_train_np, y_train_np, epochs, batch_size) :
  x_train_np = tf.convert_to_tensor(x_train_np, dtype=tf.float32)
  y_train_np = tf.convert_to_tensor(y_train_np, dtype=tf.float32)
  
  hist = model.fit(x_train_np, y_train_np,
                  epochs=epochs,
                  shuffle=True,
                  batch_size=batch_size,
                  validation_split=0.2)
                  
  print(hist.history.keys())

  toReturn = dict();
  toReturn['loss'] = hist.history['loss']
  toReturn['PSNR'] = hist.history['psnr']
  toReturn['SSIM'] = hist.history['ssim']

  return toReturn
    
# Extract frames per second (FPS) from the video file
vidcap = cv2.VideoCapture(google_drive_path + 'traffic.mp4')
success,image = vidcap.read()
print("google_drive_path:", google_drive_path)
print("success:", success)
count = 0
height = 0
width = 0
channels = 0
fps = 0

fps = vidcap.get(cv2.CAP_PROP_FPS)

while success:

  # Resize the frame for simplicity purposes
  image = cv2.resize(image, (640, 360))

  # save frame as JPEG file
  cv2.imwrite(google_drive_path + "out_frame/frame%d.jpg" % count, image)
  success,image = vidcap.read()
  count += 1

# check image size visually
print(count, fps)
img0 = cv2.imread(google_drive_path + "out_frame/frame%d.jpg" % 3)
print(img0.shape)
img0 = cv2.imread(google_drive_path + "out_frame/frame%d.jpg" % 0)
print(img0.shape)
plt.imshow(img0)
plt.show()

# inspect difference between images visually
img0 = cv2.imread(google_drive_path + "out_frame/frame%d.jpg" % 3)
img1 = cv2.imread(google_drive_path + "out_frame/frame%d.jpg" % (3+1))
img2 = cv2.imread(google_drive_path + "out_frame/frame%d.jpg" % (3+2))

# difference across frames
into = (img2/255) - (img0/255)
outof = (img1/255) - (img0/255)

into = (img2.astype(np.float32) / 255.0) - (img0.astype(np.float32) / 255.0)
into = (into - into.min()) / (into.max() - into.min()) # it is between [0,1]

plt.imshow(img0)
plt.show()
plt.imshow(into)
plt.show()
plt.imshow(outof)
plt.show()


# Create Auto Encoder Model
# tanh is used since  all the values are differences between two frames their values can range from [-1,1]. This
class VideoModel(Model):
  def __init__(self):
    super(VideoModel, self).__init__()
    self.encoder = tf.keras.Sequential([
      # layers.Input(shape=(height, width, channels)),
      layers.Input(shape=(360, 640, 3)),
      layers.Conv2D(240, (3,3), activation='tanh', padding='same', strides=2),
      layers.Conv2D(64, (3,3), activation='tanh', padding='same', strides=2),
      layers.Conv2D(16, (3,3), activation='tanh', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='tanh', padding='same'),
      layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='tanh', padding='same'),
      layers.Conv2DTranspose(240, kernel_size=3, strides=2, activation='tanh', padding='same'),
      layers.Conv2D(3, kernel_size=(3,3), activation='tanh', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = VideoModel()

# initiate Adam optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
autoencoder.compile( optimizer=opt, metrics=[PSNR, SSIM], loss=losses.MeanSquaredError())

# This script processes video frames in structured blocks to train an autoencoder model.
# It loads frames, calculates pixel differences, and iterates over multiple training blocks, storing loss, PSNR, and SSIM metrics for further analysis.
block_amount = 10 # How many blocks of data to process, open to experimentation
block_size = 4 # How many frames per block of data, open to experimentation

hist_loss = []
hist_psnr = []
hist_ssim = []

for block in range(0,block_amount) :
  # Create X and Y Train Arrays
  x_train = []
  y_train = []

  print("Loading Block: " , block)
  for i in range(0,block_size) :

    # print("Frame: " , ((block_size * block) + i) ) # Frame 0,1,2
    img0 = cv2.imread(google_drive_path + "out_frame/frame%d.jpg" % ((block_size * block) + i))
    img1 = cv2.imread(google_drive_path + "out_frame/frame%d.jpg" % ((block_size * block) + (i+1)))
    img2 = cv2.imread(google_drive_path + "out_frame/frame%d.jpg" % ((block_size * block) + (i+2)))

    # We want to train the model to predict differance to the next frame from the differance of the 0 and 2 frames
    x_train.append((img2/255) - (img0/255))
    y_train.append((img1/255) - (img0/255))

  print("Loaded Block: " , block)
  x_train_np = np.array(x_train)
  y_train_np = np.array(y_train)

  # Print the shapes of the input data and model input shape
  print("x_train_np shape:", x_train_np.shape)
  print("y_train_np shape:", y_train_np.shape)
  print("Model input shape:", autoencoder.encoder.input_shape)

  # Fit model over block of data and store history
  historyObj = run_model_iteration(
      autoencoder,
      x_train_np,
      y_train_np,
      1, # epoch 512
      4
      )
  print("Trained Block: " , block)

  hist_loss = hist_loss + historyObj['loss']
  hist_psnr = hist_psnr + historyObj['PSNR']
  hist_ssim = hist_ssim + historyObj['SSIM']
   

np.save(google_drive_path+"hist_loss" , hist_loss)
np.save(google_drive_path+"hist_psnr" , hist_psnr)
np.save(google_drive_path+"hist_ssim" , hist_ssim)

plt.plot(hist_loss, label='test')
plt.savefig(google_drive_path+ "one_input_model_loss.png")
plt.legend()
plt.show()
# plot PSNR
plt.plot(hist_psnr, label='PSNR')
plt.savefig(google_drive_path+ "one_input_model_PSNR.png")
plt.legend()
plt.show()
# plot SSIM
plt.plot(hist_ssim, label='SSIM')
plt.savefig(google_drive_path+ "one_input_model_SSIM.png")
plt.legend()
plt.show()

# Visualizing the actual vs. predicted frame differences using heatmaps
# The first set of heatmaps shows the red, green, and blue channels of the ground truth frame difference
# The second set represents the predicted difference from the autoencoder model
# The third set highlights the error between actual and predicted frame differences

b_act,g_act,r_act = cv2.split(y_train_np[3])
fig, axs = plt.subplots(ncols=3,figsize=(100,20))
sns.heatmap(r_act, ax=axs[0], vmin=-1, vmax=1)
sns.heatmap(g_act, ax=axs[1], vmin=-1, vmax=1)
sns.heatmap(b_act, ax=axs[2], vmin=-1, vmax=1)

diff = autoencoder.predict(np.array([x_train_np[3]]))
diff[0]
b_pred,g_pred,r_pred = cv2.split(diff[0])
fig, axs = plt.subplots(ncols=3,figsize=(100,20))
sns.heatmap(r_pred, ax=axs[0], vmin=-1, vmax=1)
sns.heatmap(g_pred, ax=axs[1], vmin=-1, vmax=1)
sns.heatmap(b_pred, ax=axs[2], vmin=-1, vmax=1)

r_amp = r_act - r_pred
g_amp = g_act - g_pred
b_amp = b_act - b_pred
fig, axs = plt.subplots(ncols=3,figsize=(100,20))
sns.heatmap(r_amp, ax=axs[0], vmin=-1, vmax=1)
sns.heatmap(g_amp, ax=axs[1], vmin=-1, vmax=1)
sns.heatmap(b_amp, ax=axs[2], vmin=-1, vmax=1)
