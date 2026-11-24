"""Check that VAL/TEST do not leak from TRAIN.

The split in build_dataset.py is stratified by label only. If the same patient,
or the same X-ray saved twice, lands in train AND val/test, test scores are
optimistic. This script reports:

1. Patient overlap - only if patient ids are available, from either
   --metadata CSV (columns ImageId + PatientID) or --dicom-dir (needs pydicom).
   The PNG release of SIIM-ACR has no patient ids, so this part is optional.
2. Duplicate images (always) - exact copies (MD5 of pixels) and near copies
   (64-bit difference hash, Hamming distance <= --max-distance) across splits.

    python check_split_leakage.py --images ... --masks ...
Writes results/leakage/report.txt and duplicates.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def image_hashes(path: str) -> tuple[str, int]:
    """(MD5 of decoded pixels, 64-bit dHash) for one grayscale image."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    md5 = hashlib.md5(image.tobytes()).hexdigest()
    small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    dhash = int("".join("1" if b else "0" for b in bits), 2)
    return md5, dhash


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between two uint64 hash vectors -> [len(a), len(b)]."""
    xor = np.bitwise_xor(a[:, None], b[None, :])
    return _POPCOUNT[xor.view(np.uint8).reshape(*xor.shape, 8)].sum(-1)


def near_duplicates(hashes_a, hashes_b, max_distance: int, block: int = 512):
    """Indices (i, j, distance) with Hamming(a[i], b[j]) <= max_distance."""
    a = np.asarray(hashes_a, dtype=np.uint64)
    b = np.asarray(hashes_b, dtype=np.uint64)
    found = []
    for start in range(0, len(a), block):
        dist = hamming_matrix(a[start:start + block], b)
        for i, j in zip(*np.nonzero(dist <= max_distance)):
            found.append((start + int(i), int(j), int(dist[i, j])))
    return found


def load_patient_ids(metadata: Path | None, dicom_dir: Path | None) -> dict[str, str]:
    ids: dict[str, str] = {}
    if metadata is not None:
        with metadata.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                image_id = row.get("ImageId") or row.get("image_id")
                patient = row.get("PatientID") or row.get("patient_id")
                if image_id and patient:
                    ids[Path(image_id).stem] = patient
    if dicom_dir is not None:
        import pydicom  # optional dependency
        for path in dicom_dir.rglob("*.dcm"):
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            ids[str(getattr(ds, "SOPInstanceUID", path.stem))] = str(ds.PatientID)
            ids.setdefault(path.stem, str(ds.PatientID))
    return ids


def patient_overlap(splits: dict[str, list[dict]], patient_ids: dict[str, str]) -> dict:
    owner = {}
    for name, records in splits.items():
        for r in records:
            if r["ImageId"] in patient_ids:
                owner.setdefault(patient_ids[r["ImageId"]], set()).add(name)
    shared = {p: s for p, s in owner.items() if len(s) > 1}
    by_pair = defaultdict(int)
    for s in shared.values():
        for x in sorted(s):
            for y in sorted(s):
                if x < y:
                    by_pair[f"{x}-{y}"] += 1
    return {"patients_with_ids": len(owner), "shared_patients": len(shared), "by_pair": dict(by_pair)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--metadata", type=Path, help="CSV with ImageId,PatientID (optional)")
    p.add_argument("--dicom-dir", type=Path, help="original DICOM folder (optional, needs pydicom)")
    p.add_argument("--max-distance", type=int, default=4, help="dHash Hamming distance for near copies")
    p.add_argument("--output-dir", type=Path, default=Path("results/leakage"))
    a = p.parse_args()

    from build_dataset import build_file_list, split_data  # heavy imports only when run

    records = build_file_list(str(a.images), str(a.masks))
    train, val, test = split_data(records)
    splits = {"train": train, "val": val, "test": test}
    print(f"Pairs: {len(records)} | train {len(train)} | val {len(val)} | test {len(test)}")

    hashes = {}
    for name, recs in splits.items():
        hashes[name] = [image_hashes(r["ImagePath"]) for r in recs]
        print(f"  hashed {name}: {len(recs)}")

    lines, dup_rows = [], []
    for other in ("val", "test"):
        train_md5 = {h[0]: i for i, h in enumerate(hashes["train"])}
        exact = [(train_md5[h[0]], j) for j, h in enumerate(hashes[other]) if h[0] in train_md5]
        near = near_duplicates([h[1] for h in hashes["train"]], [h[1] for h in hashes[other]], a.max_distance)
        exact_set = set(exact)
        near_only = [(i, j, d) for i, j, d in near if (i, j) not in exact_set]
        lines.append(f"train-{other}: exact copies {len(exact)} | near copies (dHash<= {a.max_distance}) "
                     f"{len(near_only)} | {other} images affected "
                     f"{len({j for _, j in exact} | {j for _, j, _ in near_only})}/{len(splits[other])}")
        for i, j in exact:
            dup_rows.append(("exact", 0, other, splits["train"][i]["ImageId"], splits[other][j]["ImageId"]))
        for i, j, d in near_only:
            dup_rows.append(("near", d, other, splits["train"][i]["ImageId"], splits[other][j]["ImageId"]))

    patient_ids = load_patient_ids(a.metadata, a.dicom_dir)
    patient_leak = False
    if patient_ids:
        info = patient_overlap(splits, patient_ids)
        patient_leak = bool(info["shared_patients"])
        lines.append(f"patients: {info['patients_with_ids']} with ids | in >1 split: "
                     f"{info['shared_patients']} | {info['by_pair']}")
    else:
        lines.append("patients: no patient ids given (PNG release) - duplicate-image check only")

    leak = any(r for r in dup_rows if r[0] == "exact") or patient_leak
    lines.append("VERDICT: " + ("LEAK FOUND - fix the split (group patients/duplicates) before final test"
                                if leak else "no exact copies across splits; review near copies in duplicates.csv"))

    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (a.output_dir / "duplicates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "hamming", "split", "train_image", "other_image"])
        writer.writerows(dup_rows)
    print("\n".join(lines))
    print("Saved:", a.output_dir / "report.txt", "and duplicates.csv")


if __name__ == "__main__":
    main()
