#!/usr/bin/env python3
import glob
import json
import os
import struct
import sys


def check_file(path):
    size = os.path.getsize(path)
    if size < 8:
        return False, f"{path}: too small ({size} bytes)"
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        if header_len > size - 8:
            return False, f"{path}: incomplete header ({header_len} > {size - 8})"
        header = json.loads(f.read(header_len))
    max_end = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        offsets = meta.get("data_offsets")
        if not offsets or len(offsets) != 2:
            return False, f"{path}: missing data_offsets for {name}"
        max_end = max(max_end, int(offsets[1]))
    expected = 8 + header_len + max_end
    if expected != size:
        return False, f"{path}: size mismatch expected {expected}, got {size}"
    return True, f"{path}: ok ({size} bytes)"


def main():
    if len(sys.argv) != 2:
        print("usage: check_safetensors_shards.py MODEL_DIR", file=sys.stderr)
        return 2
    model_dir = sys.argv[1]
    paths = sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors")))
    if not paths:
        print(f"no shards found in {model_dir}", file=sys.stderr)
        return 2
    failed = False
    for path in paths:
        ok, message = check_file(path)
        print(message)
        failed = failed or not ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
