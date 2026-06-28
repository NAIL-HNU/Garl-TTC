from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class FrameBoxes:
    image_path: Path
    bboxes: list[list[float]]
    ids: set[str]


def show_mask(mask: np.ndarray, ax, random_color: bool = False) -> None:
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 1, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box: np.ndarray, ax) -> None:
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    import matplotlib.pyplot as plt

    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def _walk_frame_records(node: Any) -> Iterable[dict[str, Any]]:
    if isinstance(node, dict):
        if 'box' in node and 'public_track_id' in node and 'meta' in node:
            yield node
        return
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk_frame_records(item)


def collect_frame_boxes_from_dataset(dataset) -> dict[Path, FrameBoxes]:
    frame_boxes: dict[Path, FrameBoxes] = {}
    datablob_dir = Path(dataset.datablob_dir)

    for frame_record in _walk_frame_records(dataset.db):
        track_id = frame_record.get('public_track_id')
        if not track_id:
            continue
        meta = frame_record.get('meta') or {}
        rel_path = meta.get('image_path')
        if not rel_path:
            continue

        asset_id = track_id.split('_')[0]
        image_path = datablob_dir / asset_id / rel_path
        entry = frame_boxes.setdefault(image_path, FrameBoxes(image_path=image_path, bboxes=[], ids=set()))
        if track_id in entry.ids:
            continue
        entry.bboxes.append([float(v) for v in frame_record['box']])
        entry.ids.add(track_id)

    return frame_boxes


def _expand_boxes(boxes: list[list[float]], expand: float, scale_factor: float) -> np.ndarray:
    input_box = np.asarray(boxes, dtype=np.float32)
    cx = (input_box[:, [0]] + input_box[:, [2]]) / 2
    cy = (input_box[:, [1]] + input_box[:, [3]]) / 2
    obj_width = input_box[:, [2]] - input_box[:, [0]]
    obj_height = input_box[:, [3]] - input_box[:, [1]]
    x1 = cx - 0.5 * obj_width * expand
    x2 = cx + 0.5 * obj_width * expand
    y1 = cy - 0.5 * obj_height * expand
    y2 = cy + 0.5 * obj_height * expand
    return np.concatenate([x1, y1, x2, y2], axis=1) * scale_factor


def generate_sam_masks_for_dataset(
    dataset,
    sam_checkpoint: str | Path,
    *,
    model_type: str = 'vit_h',
    device: str = 'cuda',
    scale_factor: float = 1.0,
    expand: float = 1.1,
    image_color: str = 'bgr',
    combine_mode: str = 'last',
    overwrite: bool = False,
    visualize: bool = False,
    vis_dir: str | Path = 'outputs/sam_mask_vis',
    limit: int | None = None,
) -> dict[str, int]:
    if image_color not in {'bgr', 'rgb'}:
        raise ValueError(f'Unsupported image_color: {image_color}')
    if combine_mode not in {'last', 'union'}:
        raise ValueError(f'Unsupported combine_mode: {combine_mode}')

    sam_checkpoint = Path(sam_checkpoint).expanduser()

    if visualize:
        import matplotlib.pyplot as plt  # noqa: F401

        Path(vis_dir).mkdir(parents=True, exist_ok=True)

    frame_boxes = collect_frame_boxes_from_dataset(dataset)
    items = list(frame_boxes.values())
    if limit is not None:
        items = items[:limit]

    stats = {'total': len(items), 'written': 0, 'skipped': 0, 'missing_images': 0, 'failed': 0}
    work_items = []
    for entry in items:
        save_path = entry.image_path.with_suffix('.npy')
        if save_path.exists() and not overwrite:
            stats['skipped'] += 1
        else:
            work_items.append(entry)

    if not work_items:
        return stats

    if not sam_checkpoint.exists():
        raise FileNotFoundError(f'SAM checkpoint not found: {sam_checkpoint}')

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            'segment-anything is required for SAM mask generation. '
            'Install it with: uv pip install git+https://github.com/facebookresearch/segment-anything.git'
        ) from exc

    sam = sam_model_registry[model_type](checkpoint=str(sam_checkpoint))
    sam = sam.to(device=device)
    predictor = SamPredictor(sam)

    for entry in tqdm(work_items, desc='Processing SAM masks', ncols=100):
        image_path = entry.image_path
        save_path = image_path.with_suffix('.npy')

        image = cv2.imread(str(image_path))
        if image is None:
            stats['missing_images'] += 1
            continue
        if image_color == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]
        if scale_factor != 1.0:
            image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))

        try:
            predictor.set_image(image)
            input_box = _expand_boxes(entry.bboxes, expand=expand, scale_factor=scale_factor)
            all_masks = None
            last_masks = None
            all_scores: list[float] = []
            for obj_id in range(len(input_box)):
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[[obj_id]],
                    multimask_output=False,
                )
                last_masks = masks
                all_masks = masks if all_masks is None else np.logical_or(all_masks, masks)
                all_scores.append(float(np.asarray(scores).reshape(-1)[0]))

            mask = last_masks if combine_mode == 'last' else all_masks
            if mask is None:
                stats['failed'] += 1
                continue

            save_record = {
                'mask': mask,
                'score': all_scores,
                'bbox': entry.bboxes,
                'expanded_bbox': input_box.tolist(),
                'ids': sorted(entry.ids),
                'combine_mode': combine_mode,
                'image_color': image_color,
                'scale_factor': scale_factor,
                'expand': expand,
            }
            np.save(str(save_path), save_record)
            stats['written'] += 1

            if visualize:
                import matplotlib.pyplot as plt

                fig = plt.figure(figsize=(20, 10))
                plt.imshow(image)
                show_mask(mask[0], plt.gca())
                for box in input_box:
                    show_box(box, plt.gca())
                plt.axis('off')
                fig.savefig(Path(vis_dir) / image_path.name)
                plt.close(fig)
        except Exception:
            stats['failed'] += 1
            continue

    return stats


def process_and_save_targets(
    dataset,
    ckpt_path: str | Path = 'checkpoints/sam_vit_h_4b8939.pth',
    model_type: str = 'vit_h',
    scale_factor: float = 1.0,
    expand: float = 1.1,
    visualize: bool = False,
    vis_dir: str | Path = './tempt_vis',
):
    return generate_sam_masks_for_dataset(
        dataset,
        ckpt_path,
        model_type=model_type,
        scale_factor=scale_factor,
        expand=expand,
        visualize=visualize,
        vis_dir=vis_dir,
        image_color='bgr',
        combine_mode='last',
        overwrite=True,
    )
