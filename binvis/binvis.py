#!/usr/bin/env python3

import argparse
import math
import os
import sys

from PIL import Image
import numpy as np
from progress.bar import IncrementalBar

# Note: We don't use transparency too much here, since it actually produces
#       rather strange brightness curves during direct-to-garment printing:
#       https://artistshopshelp.threadless.com/article/198-dtg-faq
HIGH_FREQ_COLOR = np.array([255, 255, 255])
LOW_FREQ_COLOR = np.array([0, 0, 0])

BINS = 256
# BINS = 128

def curve(frequency_map):
    # look up all non-zero values, and sort so that we can get a sense for the distribution
    values = np.sort(frequency_map[np.where(frequency_map != 0)])

    # pull out the box plot
    # q1, q3 = np.quantile(values, [0.25, 0.75])
    q1, q3 = np.quantile(values, [0.1, 0.9])
    iqr = q3 - q1
    upper_whisker = min(q3 + (1.5 * iqr), 1.0)

    # squish the high end of the frequency map, to let the lower-valued frequencies show
    return np.interp(frequency_map, [0, upper_whisker], [0, 1])

def to_pixel(x_norm):
    return (
        x_norm * (HIGH_FREQ_COLOR[0] - LOW_FREQ_COLOR[0]) + LOW_FREQ_COLOR[0],
        x_norm * (HIGH_FREQ_COLOR[1] - LOW_FREQ_COLOR[1]) + LOW_FREQ_COLOR[1],
        x_norm * (HIGH_FREQ_COLOR[2] - LOW_FREQ_COLOR[2]) + LOW_FREQ_COLOR[2],
    )

def main(bin_file, show, max_bytes):
    # make a simple 2D array that is 256x256
    frequency_map = np.zeros(shape=(BINS, BINS), dtype=np.float32)

    # open our file, and iterate over byte pairs
    raw_bytes = np.fromfile(bin_file, dtype=np.uint8, count=max_bytes)

    # re-map the byte values into the appropriate number of bins we have
    raw_bytes = np.around(np.interp(raw_bytes, [0, 255], [0, BINS-1])).astype(int)

    with IncrementalBar('Building bigrams', max=len(raw_bytes) / 128) as bar:
        i = 0;
        for a, b in zip(raw_bytes, raw_bytes[1:]):
            frequency_map[a][b] += 1
            i += 1
            if i % 128 == 0:
                bar.next()

    # normalize
    print("Normalizing")
    frequency_map = frequency_map / np.amax(frequency_map)

    # apply custom curve, to help smooth out outliers, and amplify texture
    print("Curving")

    frequency_map = curve(frequency_map)

    # convert into an RGB image (extra dimension of size 3 for RGB channels)
    print("Converting to image")
    image_array = np.zeros(shape=frequency_map.shape + (3,), dtype=np.uint8)
    for i, freq in np.ndenumerate(frequency_map):
        image_array[i] = to_pixel(freq)

    # create the image
    image = Image.fromarray(image_array)
    image = image.resize((256 * 3, 256 * 3), resample=Image.NEAREST)

    if show:
        image.show()
    else:
        image_file = os.path.basename(bin_file) + ".png"
        print("Saving {}".format(image_file))
        image.save(image_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create visualizations of binary files.')
    parser.add_argument('binary', nargs='*', help='binary file to process')
    parser.add_argument('--show', action='store_true', help='image file name to generate')
    parser.add_argument('--max-bytes', type=int, default=1000 * 1000, help='maximum number of bytes to read from a file. -1 indicates whole file.')
    args = parser.parse_args()

    for i, binary in enumerate(args.binary):
        print("Processing ({}/{}) {}".format(i+1, len(args.binary), binary))
        main(binary, args.show, args.max_bytes)

    sys.exit(0)