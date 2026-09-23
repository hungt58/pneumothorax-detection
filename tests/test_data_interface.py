import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from build_dataset import PneumothoraxDataset, get_transforms, smoke_test_dataloaders


def _make_records(tmp_path, count=6):
    records = []
    for index in range(count):
        image = np.full((48, 64), 20 + index, dtype=np.uint8)
        mask = np.zeros((48, 64), dtype=np.uint8)
        has_disease = index % 2
        if has_disease:
            mask[12:30, 16:40] = 255

        image_path = tmp_path / f"sample_{index}.png"
        mask_path = tmp_path / f"sample_{index}_mask.png"
        assert cv2.imwrite(str(image_path), image)
        assert cv2.imwrite(str(mask_path), mask)
        records.append(
            {
                "ImageId": f"sample_{index}",
                "ImagePath": str(image_path),
                "MaskPath": str(mask_path),
                "HasDisease": has_disease,
            }
        )
    return records


@pytest.mark.parametrize("mode", ["train", "val", "test"])
def test_dataset_sample_contract(tmp_path, mode):
    dataset = PneumothoraxDataset(_make_records(tmp_path), get_transforms(mode))
    sample = dataset[1]

    assert sample["image"].shape == (1, 256, 256)
    assert sample["mask"].shape == (1, 256, 256)
    assert sample["label"].shape == (1,)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.float32
    assert sample["label"].dtype == torch.float32
    assert torch.isfinite(sample["image"]).all()
    assert torch.isfinite(sample["mask"]).all()
    assert torch.isfinite(sample["label"]).all()
    assert set(torch.unique(sample["mask"]).tolist()) <= {0.0, 1.0}


def test_all_dataloaders_pass_smoke_test(tmp_path):
    records = _make_records(tmp_path)

    def make_loader(mode):
        dataset = PneumothoraxDataset(records, get_transforms(mode))
        return DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    batches = smoke_test_dataloaders(
        make_loader("train"), make_loader("val"), make_loader("test")
    )

    assert set(batches) == {"train", "val", "test"}
    for batch in batches.values():
        assert batch["image"].shape == (2, 1, 256, 256)
        assert batch["mask"].shape == (2, 1, 256, 256)
        assert batch["label"].shape == (2, 1)
