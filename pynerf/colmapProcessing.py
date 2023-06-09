# MIT License

# Copyright (c) 2023 Adithya Yerramsetty

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This module implements tooling
to take a video file,
and convert it from video
to a sparse + dense colmap
reconstruction.
"""
import os
import argparse

import imageio.v3 as iio
from PIL import Image

from pynerf.colmapUtils import COLMAPDirectoryReader
from disjoint import DisjointSetManager

COLMAP_INPUT_DIR = "images"

def videoFileToImages(videoFileName, frameSkip) -> None:
    """
    Takes a video file, and
    converts it to a list of
    PIL images, skipping
    every frameSkip frames

    :param videoFileName: The name of the video file
    :param frameSkip: The number of frames to skip
       in between images to use in the final
       reconstruction. Higher counts = faster but lower
       quality due to less overlap, lower counts = slower
       but higher quality due to more overlap. Howevrer,
       too small and you might get ambiguities due
       to low parralax between frames.
    """

    # Make the directory if it doesn't exist
    if not os.path.exists(COLMAP_INPUT_DIR):
        os.makedirs(COLMAP_INPUT_DIR)

    frameCount = 0
    for frame in iio.imiter(f"{videoFileName}", plugin="pyav"):
        frameCount += 1
        if frameCount % frameSkip == 0:
            # Convert to image
            pilImage = Image.fromarray(frame)
            pilImage.save(f"{COLMAP_INPUT_DIR}/{frameCount}.jpg")

def runColmap() -> None:
    """
    Runs CLI calls to run colmap on the input images.
    """
    DATASET_PATH = os.getcwd()

    # This runs all the components of the automatic reconstruction
    # that runs on sparse, and adds the constraint
    # that all images are from the same camera, implying
    # shared intrinsics(we assume pinhole camera model)
    os.system(
        f"colmap feature_extractor \
        --database_path {DATASET_PATH}/database.db \
        --image_path {DATASET_PATH}/images \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model PINHOLE "
    )

    os.system(
        f"colmap exhaustive_matcher \
        --database_path {DATASET_PATH}/database.db"
        ""
    )

    os.system(f"mkdir {DATASET_PATH}/sparse")

    os.system(
        f"colmap mapper \
        --database_path {DATASET_PATH}/database.db \
        --image_path {DATASET_PATH}/images \
        --output_path {DATASET_PATH}/sparse"
    )

    # Converts bin files to text files for easier reading
    os.system(
        f"python3 colMapBinToText.py --input_model {DATASET_PATH}/sparse/0 --input_format .bin --output_model {DATASET_PATH}/sparse/0 --output_format .txt"
    )


def convertVideoFileToMP4(videoFileName) -> str:
    fileName = videoFileName.split(".")[0]
    newFileName = f"{fileName}-asmp4.mp4"
    os.system(f"ffmpeg -i {videoFileName} -vcodec h264 -acodec mp2 {newFileName}")

    return newFileName

def clearDirectories() -> None:
    """
    Clears the directory of all files
    generated during the run.
    """
    os.system("rm -rf sparse")
    os.system("rm -rf images")
    os.system("rm -rf database.db")
    os.system(f"rm -rf {SEMANTIC_RESULT_DIR}")

def runOnVideoFile(videoFileName, cleanDir, frameSkip=10) -> None:
    """
    Takes a video file, and
    runs the entire pipeline
    on it.

    :param videoFileName: The name of the video file
    :param frameSkip: The number of frames to skip
       in between images to use in the final
       reconstruction. Higher counts = faster but lower
       quality due to less overlap, lower counts = slower
       but higher quality due to more overlap. Howevrer,
       too small and you might get ambiguities due
       to low parralax between frames.
    """
    clearDirectories()

    videoFileName = convertVideoFileToMP4(videoFileName)
    videoFileToImages(videoFileName, frameSkip)
    runColmap()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video file")
    parser.add_argument(
        "video_file",
        type=str,
        help="The name of the video file to process",
    )
    parser.add_argument(
        "--clean_dir",
        type=bool,
        help="Whether to clean the directory of all files before starting the run",
        default=False
    )
    args = parser.parse_args()
    fileName = args.video_file
    cleanDir = args.clean_dir

    runOnVideoFile(fileName, cleanDir)