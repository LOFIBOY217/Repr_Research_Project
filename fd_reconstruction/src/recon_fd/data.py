"""Deterministic reconstruction inputs, stable sample identities, resumable order."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler, Subset, DataLoader
from .config import expand_path
from .provenance import fingerprint


def center_crop(image, size):
    # ADM/Grounded BOX downsampling followed by bicubic resize and center crop.
    while min(image.size) >= 2 * size:
        image = image.resize(tuple(n // 2 for n in image.size), Image.Resampling.BOX)
    scale = size / min(image.size)
    image = image.resize(tuple(round(n * scale) for n in image.size), Image.Resampling.BICUBIC)
    x, y = (image.width - size) // 2, (image.height - size) // 2
    return image.crop((x, y, x + size, y + size))


class ImageDataset(Dataset):
    def __init__(self, root, resolution):
        self.root, self.resolution = Path(expand_path(root)), resolution
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.paths = sorted(p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
        if len(self.paths) < 2:
            raise ValueError(f"Need >=2 images under {self.root}")
        self.ids = [p.relative_to(self.root).as_posix() for p in self.paths]
        manifest = hashlib.sha256()
        for name, path in zip(self.ids, self.paths):
            stat = path.stat()
            manifest.update(f"{name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        self.identity = {"kind": "images", "inventory_sha256": manifest.hexdigest(),
                         "count": len(self), "resolution": resolution, "transform": "adm_center_crop_rgb01_v1"}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = center_crop(image.convert("RGB"), self.resolution)
            tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
        return {"image": tensor, "id": self.ids[index]}


class SyntheticDataset(Dataset):
    def __init__(self, count, resolution, seed):
        self.count, self.resolution, self.seed = count, resolution, seed
        self.ids = [f"synthetic_{i:06d}" for i in range(count)]
        self.identity = {"kind": "synthetic", "count": count, "resolution": resolution, "seed": seed}

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rng = torch.Generator().manual_seed(self.seed + index)
        return {"image": torch.rand(3, self.resolution, self.resolution, generator=rng), "id": self.ids[index]}


def build_dataset(config, split):
    spec = config["data"]
    if spec["kind"] == "synthetic":
        dataset = SyntheticDataset(spec["synthetic_count"], spec["resolution"], spec["seed"] + (100000 if split == "val" else 0))
    elif spec["kind"] == "imagenet":
        dataset = ImageDataset(spec[f"{split}_path"], spec["resolution"])
    else:
        raise ValueError("Only imagenet or synthetic datasets are supported")
    dataset.identity["split"] = split
    return dataset


def selected_dataset(dataset, count, seed, random_subset=False):
    if count > len(dataset) or count < 2:
        raise ValueError(f"Requested {count} samples, dataset contains {len(dataset)}; no silent truncation")
    indices = (torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:count].tolist()
               if random_subset else list(range(count)))
    subset = Subset(dataset, indices)
    subset.ids = [dataset.ids[i] for i in indices]
    subset.identity = {"dataset": dataset.identity, "indices_sha256": fingerprint(indices), "count": count}
    return subset


class ResumableBatchSampler(Sampler):
    def __init__(self, count, batch_size, seed, start_step, end_step):
        self.count, self.batch_size, self.seed = count, batch_size, seed
        self.start_step, self.end_step = start_step, end_step
        self.batches_per_epoch = count // batch_size
        if not self.batches_per_epoch:
            raise ValueError("Training data smaller than one batch")

    def __iter__(self):
        last_epoch, order = None, None
        for step in range(self.start_step, self.end_step):
            epoch, offset = divmod(step, self.batches_per_epoch)
            if epoch != last_epoch:
                order = torch.randperm(self.count, generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
                last_epoch = epoch
            start = offset * self.batch_size
            yield order[start:start + self.batch_size]

    def __len__(self):
        return max(0, self.end_step - self.start_step)


def sequential_loader(dataset, batch_size, workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=workers,
                      generator=torch.Generator().manual_seed(0))
