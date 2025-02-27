# from numpy.random import RandomState, MT19937, SeedSequence
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from types import MappingProxyType
from lang_sam import LangSAM
from pathlib import Path
from typing import List
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import cv2


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
torch.inference_mode().__enter__()


VIDEO_RESOLUTION_TYPE = MappingProxyType({
    '480p_16_9': (854, 480),
    '480p_4_3': (640, 480),
    '360p_16_9': (640, 360),
    '360p_4_3': (480, 360),
}) # 16:9


def merge_and_reduce(results: List[dict]) -> List[List[tuple]]:
    """
    Merge replicated labels and reject labels with spaces.
    Args:
        results: Picture list of diction: {'scores', 'labels', 'boxes', 'masks', 'mask_scores'}
    Returns:
        List of sorted list: [(label, mask, score), ...]
    """
    new_results = []
    for r in results:
        new_r, foreground = [], 0
        labels, masks, scores = r['labels'], r['masks'].copy(), r['scores']
        for candidate_label in set(labels):
            if " " in candidate_label: continue
            candidate_mask = 0
            for label, mask, score in zip(labels, masks, scores):
                if label == candidate_label:
                    candidate_mask = np.logical_or(candidate_mask, mask.astype(np.bool))
                    candidate_score = max(0, score.item())
            new_r.append((candidate_label, candidate_mask, candidate_score))
        new_r = sorted(new_r, key=lambda x: x[0], reverse=True) # Sort by label name
        new_r = [(l, fill_pole(m), s) for l, m, s in new_r] # Fill the pole before make the background
        for _, mask, _ in new_r:
            foreground = np.logical_or(foreground, mask)
        new_results.append([('background', foreground == False, 1.)] + new_r)
    return new_results


def fill_pole(mask: np.ndarray, kernel_size: int = 13) -> np.ndarray:
    """
    Try to fill the pole in the mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(np.bool)
    return mask


def draw_mask(r: List[tuple], color_dict: dict, output_path: Path):
    # r = sorted(r, key=lambda x: x[1].sum(), reverse=True) # Sort by mask area
    r = sorted(r, key=lambda x: x[0], reverse=True) # Sort by label name
    image = np.zeros((*r[0][1].shape, 3)).astype(np.uint8)
    for label, mask, _ in r:
        image[mask] = color_dict[label.lower()]
    plt.imsave(output_path, image)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=3407)
    parser.add_argument('-m', '--model', type=str, default='tiny', choices=['tiny', 'small', 'base_plus', 'large'])
    parser.add_argument('-t', '--text', '--text-prompt', type=str, nargs='+', required=True)
    parser.add_argument('-bt', '--box-threshold', type=float, required=True)
    parser.add_argument('-tt', '--text-threshold', type=float, required=True)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', '--output-folder', type=str, default='./output')
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('-e', '--extension', type=str, default='png')
    parser.add_argument('-f', '--fps', type=int, default=30)
    parser.add_argument('-r', '--resolution', type=str, default='360p_4_3', choices=VIDEO_RESOLUTION_TYPE.keys())
    args = parser.parse_args()

    np.random.seed(args.seed)
    langsam = LangSAM(sam_type=f'sam2.1_hiera_{args.model}', device=DEVICE)
    text_prompt = '.'.join(args.text)
    color_dict_ks = ['background'] + args.text
    color_dict_vs = np.random.randint(0, 256, (len(args.text) + 1, 3))
    color_dict = dict(zip([k.lower() for k in color_dict_ks], color_dict_vs))
    # input can be a folder or a single image
    input = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if input.is_dir():
        # Get all image paths and check if any exist
        image_paths = list(sorted(input.glob(f'*.{args.extension}')))
            
        # Process in batches
        batches = []
        for i in range(0, len(image_paths), args.batch_size):
            batches.append(image_paths[i:i+args.batch_size])
            
        for batch in tqdm(batches):
            torch.cuda.empty_cache()
            images_pil = [Image.open(p).convert("RGB").resize(VIDEO_RESOLUTION_TYPE[args.resolution]) for p in batch]

            results = langsam.predict(
                images_pil=images_pil,
                texts_prompt=[text_prompt] * len(images_pil),
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
            results = merge_and_reduce(results)
            for r, p in zip(results, batch):
                draw_mask(r, color_dict, output / p.name)
    else:
        image_pil = Image.open(input).convert("RGB").resize(VIDEO_RESOLUTION_TYPE[args.resolution])
        results = langsam.predict(
            images_pil=[image_pil],
            texts_prompt=[text_prompt],
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        results = merge_and_reduce(results)
        draw_mask(results[0], color_dict, output / f'{input.stem}.{args.extension}')



192.168.55.203

default via 192.168.55.203 dev wlan0 proto dhcp src 192.168.55.254 metric 600