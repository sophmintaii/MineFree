import argparse
import os

import cv2
import imutils
import numpy as np
import tifffile


def rotate(image):
    """
    Rotates the satellite image so there is no blank space.
    :param image: input image to be rotated.
    :return: transformed image, the transformation matrix, output_shape.
    """
    # extend the image borders so the morphological closing does not interfere with corners
    extended_image = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # resize image to reduce noisy contours
    scale_percent = 100
    width = int(extended_image.shape[1] * scale_percent / 100)
    height = int(extended_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(extended_image, dim, interpolation=cv2.INTER_AREA)

    # thresholding
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) > 2 else resized
    th, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    # find the largest contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    # minimal bounding rectangle
    rot_rect = cv2.minAreaRect(cnt)
    angle = rot_rect[2]  # angle
    cx, cy = (rot_rect[0][0], rot_rect[0][1])  # center
    sx, sy = (rot_rect[1][0], rot_rect[1][1])  # size

    # construct affine transformation
    dst_pts = np.array([[0, sy], [0, 0], [sx, 0], [sx, sy]]).astype('int')
    current_pts = cv2.boxPoints(rot_rect).astype('int')

    # sort the points
    idx_dst = np.lexsort((dst_pts[:, 1], dst_pts[:, 0]))
    idx_current = np.lexsort((current_pts[:, 1], current_pts[:, 0]))
    dst_pts = np.array([dst_pts[i] for i in idx_dst])
    current_pts = np.array([current_pts[i] for i in idx_current])

    # estimate the transformation matrix
    M, _ = cv2.estimateAffine2D(current_pts, dst_pts)

    warped = cv2.warpAffine(extended_image, M, (int(sx),int(sy)))

    return warped, M, (int(sx),int(sy))


def minmax_stretch_one_channel(image, method="percentile"):
    """Min-max stretch for one-channel image."""
    if method == "percentile":
        min_percent = 2
        max_percent = 98
        lo, hi = np.nanpercentile(image, (min_percent, max_percent))
    elif method == "mean_std":
        image_mean = np.mean(image)
        image_std = np.std(image)
        lo, hi = image_mean - 2 * image_std, image_mean + 2 * image_std
    res_img = (image.astype(float) - lo) / (hi - lo)
    return np.maximum(np.minimum(res_img * 255, 255), 0).astype(np.uint8)


def minmax_stretch(image, method="percentile"):
    """Min-max stretch for each channel of the image."""
    if method is not None:
        new_image = []
        i = 0
        for channel in range(image.shape[2]):
            channel_values = image[:, :, channel]
            new_image.append((minmax_stretch_one_channel(channel_values, method)))
            i += 1
        new_image = np.dstack(new_image)
        return new_image
    return image


def cut_tiff(input_path, output_path, size=1024, preprocessing_method="percentile"):
    """
    Cuts TIFF image into parts of the passed size.
    :param input_path: path to the input image.
    :param output_path: path to the output directory.
    :param size: size of the resulting tiles.
    :preprocess: "percentile"/"mean_std"/None.
    :return: None.
    """
    filename = os.path.basename(input_path).split(".")[0]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(f"{output_path}/{filename}_{size}"):
        os.mkdir(f"{output_path}/{filename}_{size}")

    image = tifffile.imread(input_path)
    if image.shape[0] <= 4:  # channels last
        image = np.moveaxis(image, 0, -1)
    image = np.delete(image, [3], axis=2)[:, :, :]
    image = ((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype("uint8")
    # image, M, _ = rotate(image)
    image = minmax_stretch(image, method=preprocessing_method)
    # tifffile.imwrite(os.path.join(input_path[:-4] + "-rotated.tif"), image)
    
    range_x = range(0, image.shape[0] // size) if (image.shape[0] // size) else [0]
    range_y = range(0, image.shape[1] // size) if (image.shape[0] // size) else [0]

    for i in range_x:
        for j in range_y:
            x = i * size
            y = j * size
            end_x = min(x + size, image.shape[0])
            end_y = min(y + size, image.shape[1])
            end_x = image.shape[0] if image.shape[0] - end_x < size else end_x
            end_y = image.shape[1] if image.shape[1] - end_y < size else end_y
            curr_part = image[x:end_x, y:end_y, :]
            if np.any(curr_part):
                cv2.imwrite(f"{output_path}/{filename}_{size}/{filename}_{x}_{y}.png", curr_part)
                # np.save(f"{output_path}/{filename}_{size}/{filename}_transformation-matrix.npy", np.asarray(M[:, :]))



def cut_classify_tiff(input_path, mask_path, output_path, size=128, preprocessing_method="percentile", mask_thresh=0.033):
    """
    Cuts TIFF image and separated it into 'bombed' and 'not-bombed' directories.
    depending whether there are craters labeled on the mask.
    :param input_path: path to the input image.
    :param mask_path: path to the segmentation mask image.
    :param output_path: path to the output directory.
    :param size: size of the resulting tiles.
    :return: None.
    """
    filename = os.path.basename(input_path).split(".")[0]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(f"{output_path}/{filename}_{size}"):
        os.mkdir(f"{output_path}/{filename}_{size}")
    if not os.path.exists(f"{output_path}/{filename}_{size}/not-bombed"):
        os.mkdir(f"{output_path}/{filename}_{size}/not-bombed")
    if not os.path.exists(f"{output_path}/{filename}_{size}/bombed"):
        os.mkdir(f"{output_path}/{filename}_{size}/bombed")

    image = tifffile.imread(input_path)

    if image.shape[-1] > 3:
        image = np.delete(image, [3], axis=2)[:, :, :]
    image = ((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype("uint8")
    image = minmax_stretch(image, method=preprocessing_method)
    # image, M, output_shape = rotate(image)
    image = minmax_stretch(image, method=preprocessing_method)

    mask = tifffile.imread(mask_path)

    for i in range(0, image.shape[0] // size):
        for j in range(0, image.shape[1] // size):
            x = i * size
            y = j * size
            end_x = min(x + size, image.shape[0])
            end_y = min(y + size, image.shape[1])
            end_x = image.shape[0] if image.shape[0] - end_x < size else end_x
            end_y = image.shape[1] if image.shape[1] - end_y < size else end_y
            curr_part = image[x:end_x, y:end_y, :]
            mask_curr_part = mask[x:end_x, y:end_y]
            if np.any(curr_part) and np.any(mask_curr_part):
                label = "bombed" if (np.count_nonzero(mask_curr_part) / mask_curr_part.size >= mask_thresh) else "not-bombed"
                cv2.imwrite(f"{output_path}/{filename}_{size}/{label}/{filename}_{x}_{y}.png", curr_part)
                # cv2.imwrite(f"{output_path}/{filename}_{size}/{label}/{filename}_{x}_{y}_mask.png", mask_curr_part)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str,
                        help="Path to the input tiff file.")
    parser.add_argument("--mask", "-m", type=str, nargs="?",
                        help="Path to the mask tiff file.")
    parser.add_argument("--output", "-o", type=str, default="./output",
                        help="Path to the folder with output images.")
    parser.add_argument("--size", "-s", type=int, default=128,
                        help="Size of the output images.")
    args = parser.parse_args()
    cut_classify_tiff(args.input, args.mask, args.output, args.size, "percentile")
