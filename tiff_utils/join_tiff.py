import argparse
import os
from collections import defaultdict

import cv2
import numpy as np
import tifffile
import imagesize


def collect_filenames_sizes(directory_path, filenames):
    """
    Collects information about the large image from its piles.
    :param directory_path: directory to the input images.
        (assuming file names are like the following: xxxxxx_locationX_locationY.png, where xxxxxx can not be
        separated by underscores)
    :param filenames:
    :return: (image sizes: dict(image filename: [h, w]), piles filenames: dict(image filename: [list of piles of the
        image]))
    """
    image_sizes = defaultdict(lambda: (0, 0))
    piles = defaultdict(lambda: [])
    for filename in filenames:
        img_name, x, y = tuple(filename.split(".")[0].split("_"))
        piles[img_name].append(os.path.join(directory_path, filename))
        dimensions = imagesize.get(os.path.join(directory_path, filename))[::-1]
        image_sizes[img_name] = (max(image_sizes[img_name][0], int(x) + dimensions[0]),
                                 max(image_sizes[img_name][1], int(y) + dimensions[1]))
    return image_sizes, piles


def replace_image_part(input_image, pile_path, pile_content=None):
    """
    Replaces part of the image corresponding to the passed pile with the passed content.
    :param input_image: array with the full image.
    :param pile_path: path and filename of the pile (part of the full image).
        (assuming file names are like the following: xxxxxx_locationX_locationY.png, where xxxxxx can not be
        separated by underscores)
    :param pile_content: one of True, False, None.
        If True, replaces the part of image with ones;
        If False, replaces the part of image with zeros;
        If None, replaces the part of the image with the pile contents.
    :return: image with replaced part.
    """
    _, x, y = tuple(pile_path.split(".")[0].split("_"))
    x = int(x)
    y = int(y)
    pile = cv2.imread(pile_path)
    dimensions = pile.shape[:-1]
    # assumes 2-dimensionality and 1 channel
    if pile_content is None:
        input_image[x:(x + dimensions[0]), y:(y + dimensions[1])] = \
        cv2.threshold(cv2.cvtColor(pile, cv2.COLOR_BGR2GRAY),
                      0, 255, cv2.THRESH_BINARY)[1]
    elif pile_content:
        input_image[x:(x + dimensions[0]), y:(y + dimensions[1])] = np.ones(dimensions)
    else:
        input_image[x:(x + dimensions[0]), y:(y + dimensions[1])] = np.zeros(dimensions)
    return input_image


def join_classification_dataset(input_path, output_path):
    """
    Joins classification dataset so that the bombed parts are displayed as black and not bombed parts - as white.
    :param input_path: path to the directory with the input images. Contains subdirectories 'bombed' and 'not-bombed'.
        (assuming filenames are like the following: input_path/label/xxxxx_locationX_locationY.tif, where xxxxxx
        can not be separated by underscores)
    :param output_path: path to the directory for the output images to be saved.
        resulting output filenames: output_path/filename_labels.tif
    :return: None
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    input_contents = [(out[0], out[2]) for out in os.walk(input_path)][1:]

    # collect all the full filenames and corresponding image sizes and filenames of their parts
    image_sizes = defaultdict(lambda: (0, 0))
    piles = defaultdict(lambda: [])
    for subdir, filenames in input_contents:
        subdir_image_sizes, subdir_piles = collect_filenames_sizes(subdir, filenames)
        image_sizes = dict(image_sizes, **subdir_image_sizes)
        piles = dict(piles, **subdir_piles)

    # create an empty canvas corresponding to the image size and fill it in with mask piles
    for filename in image_sizes:
        image = np.zeros(image_sizes[filename])
        for pile_path in piles[filename]:
            if "not-bombed" not in pile_path:
                image = replace_image_part(image, pile_path, None)
        image = cv2.convertScaleAbs(image, alpha=255.0)
        tifffile.imwrite(os.path.join(output_path, filename + "_labels.tif"), image)


def join_masks(masks_path, output_path):
    """
    Joins segmentation masks for each picture into one TIFF file for each picture and saves them.
    :param masks_path: path to the directory for input images
        (assuming file names are like the following: xxxxxx_locationX_locationY.png, where xxxxxx
        can not be separated by underscores)
    :param output_path: path to the directory for the output images to be saved.
        resulting output filenames: output_path/filename_mask.tif
    :return:None
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    input_filenames = [out for out in os.walk(masks_path)][0][-1]

    # collect all the full filenames and corresponding image sizes and filenames of their parts
    image_sizes, piles = collect_filenames_sizes(masks_path, input_filenames)

    # create an empty canvas corresponding to the image size and fill it in with mask piles
    for filename in image_sizes:
        image = np.zeros(image_sizes[filename])
        for pile_path in piles[filename]:
            image = replace_image_part(image, pile_path, None)
        image = cv2.convertScaleAbs(image, alpha=255.0)
        print(os.path.join(output_path, filename + "_mask.tif"))
        tifffile.imwrite(os.path.join(output_path, filename + "_mask.tif"), image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str,
                        help='Path to the input directory.')
    parser.add_argument('--output', '-o', type=str, default='./output',
                        help='Path to the output directory.')
    args = parser.parse_args()
    join_masks(args.input, args.output)
