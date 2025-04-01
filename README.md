# Video-Frame-Autoencoder

Frame difference techniques are essential for video compression, as they help reduce redundancy by encoding only the changes between consecutive frames rather than storing each frame independently. This approach is widely used in modern compression algorithms, such as H.264 and H.265, where motion estimation and compensation allow efficient encoding. Autoencoders, like the one used in this script, can learn these frame differences and predict upcoming frames, enabling even more optimized compression. By leveraging deep learning for difference-based encoding, video storage and transmission become more efficient while maintaining visual quality. This method is particularly valuable for applications like video streaming, surveillance, and real-time processing, where bandwidth and storage optimization are critical.

This script focuses on video frame processing, feature extraction, and autoencoder training to predict frame differences for video compression tasks. It uses TensorFlow, OpenCV, and seaborn to process and analyze video frames while evaluating performance with PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).

**`Workflow`**

**`Frame Extraction and Resizing`**

Video frames are extracted from a .mp4 file and resized to 640x360 resolution and saved for further processing. Differences between consecutive frames are calculated and normalized. Visual inspection is performed to verify changes across frames.

**`Autoencoder Model`**

A convolutional autoencoder is trained to predict differences between frames. The encoder extracts compressed features, while the decoder reconstructs frame differences.

**`Model Training and Evaluation`**

The autoencoder is trained over structured blocks of frames.Performance is measured using loss values, PSNR, and SSIM metrics. Training history is saved and plotted for analysis.

