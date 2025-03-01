from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np


def eval_frame_consistency(
        frame1: np.ndarray,
        frame2: np.ndarray,
):
    """
    Evaluate the normalized consistency between two frames.
    """
    diff = frame1 - frame2
    # find out the point on (h, w) where the difference is not zero
    diff = np.sum(diff, axis=-1)
    return np.mean(diff == 0)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input-folder', type=str, required=True)
    parser.add_argument('-e', '--extension', type=str, default='png')
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    image_paths = list(sorted(input_folder.glob(f'*.{args.extension}')))

    consistency = 0
    for f1, f2 in tqdm(zip(image_paths[:-1], image_paths[1:]), leave=False):
        frame1 = np.array(Image.open(f1))
        frame2 = np.array(Image.open(f2))
        consistency += eval_frame_consistency(frame1, frame2)
    consistency /= len(image_paths) - 1
    print(f'{args.input_folder}: {consistency}')