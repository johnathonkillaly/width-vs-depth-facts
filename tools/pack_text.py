"""Pack plain text into the token shards recipe/proxy.py reads.

    python tools/pack_text.py --tokenizer tokenizer.json --eos-id 2 --out data/text/general/train  docs*.txt
    python tools/pack_text.py --tokenizer tokenizer.json --eos-id 2 --out data/text/general/validation  held_out.txt

Each input file is read as one document per blank-line-separated paragraph block (or pass --jsonl with a "text"
field per line). Documents are tokenized, joined with the EOS id, and cut into 4,096-token uint16 blocks; a
manifest.json lists the shards. The vocabulary must be below 65,536. Run it once for `general` (train and
validation) and once for a second held-out source named `stack` (validation only; E02-E04 used held-out code).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BLOCK = 4096


def documents(paths, jsonl):
    for p in paths:
        text = Path(p).read_text(encoding="utf-8")
        if jsonl:
            for line in text.splitlines():
                if line.strip():
                    yield json.loads(line)["text"]
        else:
            for doc in text.split("\n\n"):
                if doc.strip():
                    yield doc.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--eos-id", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jsonl", action="store_true")
    ap.add_argument("--blocks-per-shard", type=int, default=4096)
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    if tok.get_vocab_size() >= 65536:
        raise SystemExit("vocabulary must fit uint16")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    buf, shards, blocks = [], [], []

    def flush():
        if not blocks:
            return
        name = f"tokens-{len(shards):05d}.bin"
        np.asarray(blocks, dtype="<u2").tofile(out / name)
        shards.append({"path": name, "blocks": len(blocks)})
        blocks.clear()

    for doc in documents(a.inputs, a.jsonl):
        buf += tok.encode(doc, add_special_tokens=False).ids + [a.eos_id]
        while len(buf) >= BLOCK:
            blocks.append(buf[:BLOCK]); buf = buf[BLOCK:]
            if len(blocks) == a.blocks_per_shard:
                flush()
    flush()
    if not shards:
        raise SystemExit("not enough text for one 4,096-token block")
    total = sum(s["blocks"] for s in shards)
    (out / "manifest.json").write_text(json.dumps({"format": "uint16-le 4096-token blocks", "sequence_length": BLOCK,
                                                    "total_blocks": total, "shards": shards}, indent=1))
    print(f"{total} blocks ({total * BLOCK:,} tokens) in {len(shards)} shard(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
