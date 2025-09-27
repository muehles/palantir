import cv2
import numpy as np
import time

# ==============================================================
# CONFIG
# ==============================================================
USE_MULTIBAND = True  # Toggle multi-band blending on/off

# ==============================================================
# HELPERS
# ==============================================================

def cylindrical_warp(img, K):
    """
    Warp image into cylindrical projection.
    K = camera intrinsic matrix
    """
    h, w = img.shape[:2]
    f = K[0,0]  # focal length from calibration

    # Meshgrid of pixel coordinates
    y_i, x_i = np.indices((h, w))
    X = (x_i - K[0,2]) / f
    Y = (y_i - K[1,2]) / f
    Z = np.sqrt(X**2 + 1)

    # Project to cylindrical coords
    map_x = f * np.arctan(X) + K[0,2]
    map_y = f * (Y / Z) + K[1,2]

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

def match_histogram_color(src, ref):
    """
    Adjust color and brightness of src image to match ref image using histogram matching.
    Both must be same size.
    """
    src_yuv = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)
    ref_yuv = cv2.cvtColor(ref, cv2.COLOR_BGR2YUV)

    # Match Y (luminance)
    src_hist, _ = np.histogram(src_yuv[:,:,0].flatten(), 256, [0,256])
    ref_hist, _ = np.histogram(ref_yuv[:,:,0].flatten(), 256, [0,256])

    cdf_src = src_hist.cumsum()
    cdf_ref = ref_hist.cumsum()
    cdf_src = 255 * cdf_src / cdf_src[-1]
    cdf_ref = 255 * cdf_ref / cdf_ref[-1]

    M = np.zeros(256, dtype=np.uint8)
    j = 0
    for i in range(256):
        while j < 255 and cdf_ref[j] < cdf_src[i]:
            j += 1
        M[i] = j

    src_yuv[:,:,0] = cv2.LUT(src_yuv[:,:,0], M)

    return cv2.cvtColor(src_yuv, cv2.COLOR_YUV2BGR)

def multiband_blend(img1, img2):
    """
    Multi-band blending between two warped images of the same height.
    """
    h = min(img1.shape[0], img2.shape[0])
    img1 = img1[:h]
    img2 = img2[:h]

    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(5)
    blender.prepare((0, 0, img1.shape[1] + img2.shape[1], h))

    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)

    mask1 = 255 * np.ones(img1.shape[:2], np.uint8)
    mask2 = 255 * np.ones(img2.shape[:2], np.uint8)

    blender.feed(img1_float, mask1, (0,0))
    blender.feed(img2_float, mask2, (img1.shape[1]//2, 0))

    result, _ = blender.blend(None, None)
    return cv2.convertScaleAbs(result)

# ==============================================================
# MAIN PIPELINE
# ==============================================================

# Load calibration (must be created beforehand)
calib1 = np.load("calib_cam0.npz")
calib2 = np.load("calib_cam1.npz")
mtx1, dist1 = calib1["mtx"], calib1["dist"]
mtx2, dist2 = calib2["mtx"], calib2["dist"]

# Camera setup
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

width, height = 640, 480
for cap in (cap1, cap2):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Video writer
output_filename = f"/home/pi/stitched_{int(time.time())}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_filename, fourcc, 20.0, (width*2, height))

# Stitcher
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

print("Recording cylindrical-stitched video... press Ctrl+C to stop.")

try:
    while True:
        ret1, frame1 = cap1.read()
        ret2 = cap2.read()[1]
        if not ret1 or frame2 is None:
            break

        # Undistort
        frame1 = cv2.undistort(frame1, mtx1, dist1, None, mtx1)
        frame2 = cv2.undistort(frame2, mtx2, dist2, None, mtx2)

        # Cylindrical warp
        frame1 = cylindrical_warp(frame1, mtx1)
        frame2 = cylindrical_warp(frame2, mtx2)

        # Exposure & color match frame2 -> frame1
        if frame1.shape == frame2.shape:
            frame2 = match_histogram_color(frame2, frame1)

        # Stitch attempt
        status, stitched = stitcher.stitch([frame1, frame2])
        if status == cv2.Stitcher_OK:
            stitched_resized = cv2.resize(stitched, (width*2, height))
            writer.write(stitched_resized)
            cv2.imshow("Cylindrical Stitched Video", stitched_resized)

        elif USE_MULTIBAND and frame1.shape[0] == frame2.shape[0]:
            # Fallback with multi-band blending
            blended = multiband_blend(frame1, frame2)
            blended_resized = cv2.resize(blended, (width*2, height))
            writer.write(blended_resized)
            cv2.imshow("Cylindrical Stitched Video", blended_resized)

        else:
            # Simple side-by-side fallback
            combined = cv2.hconcat([frame1, frame2])
            writer.write(combined)
            cv2.imshow("Cylindrical Stitched Video", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    cap1.release()
    cap2.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved stitched video to {output_filename}")
