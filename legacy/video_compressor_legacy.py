import numpy as np
import cv2
import os
import time
from network import Autoencoder, Dense, Activation, relu, relu_prime, sigmoid, sigmoid_prime
from image_compressor import build_autoencoder, image_to_patches, patches_to_image

def compress_video(input_video_path, output_video_path, patch_size=16, encoding_dim=64,
                   epochs_per_frame=2, learning_rate=0.01, callback=None):
    """
    Compress and reconstruct each frame of a video using the autoencoder.
    callback(msg): optional function to receive log messages.
    """
    def log(msg):
        print(msg)
        if callback:
            callback(msg)

    log(f"Opening Video: {input_video_path}")
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video at {input_video_path}.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    log(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    h_adj = height - (height % patch_size)
    w_adj = width - (width % patch_size)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w_adj, h_adj))

    input_dim = patch_size * patch_size * 3
    ae = build_autoencoder(input_dim, encoding_dim)

    frame_count = 0
    start = time.time()

    log("Processing frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb.astype('float32') / 255.0
        frame_adj = frame_normalized[:h_adj, :w_adj, :]

        patches = image_to_patches(frame_adj, patch_size)
        n_patches = patches.shape[0]
        x_train = patches.reshape((n_patches, input_dim))

        # Train longer on the first frame
        if frame_count == 1:
            ae.fit(x_train, epochs=max(epochs_per_frame, 10), learning_rate=learning_rate,
                   batch_size=32, verbose=False)
        else:
            ae.fit(x_train, epochs=epochs_per_frame, learning_rate=learning_rate,
                   batch_size=32, verbose=False)

        reconstructed_flat = ae.forward(x_train)
        reconstructed_patches = reconstructed_flat.reshape((n_patches, patch_size, patch_size, 3))
        reconstructed_frame = patches_to_image(reconstructed_patches, frame_adj.shape, patch_size)

        reconstructed_bgr = np.clip(reconstructed_frame * 255, 0, 255).astype('uint8')
        reconstructed_bgr = cv2.cvtColor(reconstructed_bgr, cv2.COLOR_RGB2BGR)
        out.write(reconstructed_bgr)

        if frame_count % 5 == 0 or frame_count == 1:
            log(f"Frame {frame_count}/{total_frames}")

    cap.release()
    out.release()

    elapsed = time.time() - start
    log(f"Video processing complete in {elapsed:.1f}s")
    log(f"Output saved: {output_video_path}")
    return output_video_path

if __name__ == "__main__":
    pass
