import json
from pathlib import Path

import yaml

from vnsign.data.prepare import parse_label_file, prepare, split_groups


def test_parse_label_file_keeps_valid_and_reports_bad(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.5 0.5 0.2 0.2\n5 0.5 0.5 0.2 0.2\n1 1.5 0.5 0.2 0.2\n1 0.5 0.5\nx a b c d\n")
    boxes, errs = parse_label_file(f, num_classes=3)
    assert boxes == [(0, 0.5, 0.5, 0.2, 0.2)]
    assert len(errs) == 4


def test_parse_label_file_missing(tmp_path: Path):
    assert parse_label_file(tmp_path / "none.txt", 3) == ([], ["missing"])


def test_split_groups_keeps_groups_together_and_ratio():
    groups = [[i] for i in range(80)] + [[80 + i for i in range(20)]]  # 1 nhóm 20 ảnh
    out = split_groups(groups, {"train": 0.8, "valid": 0.1, "test": 0.1}, seed=1)
    allidx = sum(out.values(), [])
    assert sorted(allidx) == list(range(100))
    big = set(range(80, 100))
    assert any(big <= set(v) for v in out.values())  # cả nhóm nằm chung 1 tập
    assert abs(len(out["train"]) - 80) <= 20


def test_prepare_end_to_end(tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "labels").mkdir()
    (raw / "classes_en.txt").write_text("stop\nyield\n")
    for i in range(20):
        (raw / "images" / f"{i}.jpg").write_bytes(b"fake")
        (raw / "labels" / f"{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.2 0.2\n")
    cfg = {"data": {"raw_dir": str(raw), "processed_dir": str(tmp_path / "out"),
                    "split": {"train": 0.8, "valid": 0.1, "test": 0.1},
                    "seed": 0, "dedupe_hamming": -1, "kaggle_dataset": "x"}}
    y = prepare(cfg)
    data = yaml.safe_load(y.read_text())
    assert data["names"] == {0: "stop", 1: "yield"}
    rep = json.loads((tmp_path / "out" / "split_report.json").read_text())
    assert sum(v["images"] for v in rep.values()) == 20
    for s in ("train", "valid", "test"):
        imgs = {p.stem for p in (tmp_path / "out" / s / "images").iterdir()}
        lbls = {p.stem for p in (tmp_path / "out" / s / "labels").iterdir()}
        assert imgs == lbls
