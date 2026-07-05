#!/usr/bin/env python3
"""Fix DSA indexer block-table overflow crash at concurrency >= 3.

Co-discovered independently with NVIDIA forum user ciprianveg
(thread 374125 posts 105/107, his `mods/fix-dsa-block-table-dim`).

Root cause
----------
The DeepSeek-Sparse-Attention indexer sizes its `expanded_block_table_buffer`
from `max_model_len` alone:

    max_num_blocks_per_req = cdiv(self.max_model_len, self.kv_cache_block_size)

But the scheduler's actual block table can be **one block wider** than this —
MTP speculative tokens and scheduler overhang push a request one block past the
`max_model_len`-derived count. When that happens (reliably at concurrency >= 3),
the engine crashes with:

    RuntimeError: The expanded size of the tensor (N) must match the existing
    size (N+1) at non-singleton dimension ...

The fix
-------
Add one block of headroom before allocating the buffer:

    max_num_blocks_per_req = cdiv(self.max_model_len, self.kv_cache_block_size) + 1

Idempotent: safe to run more than once.
"""

import re
import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py"

# Match the cdiv(...) assignment for max_num_blocks_per_req, tolerating
# whitespace / exact arg names, as long as it is not already `+ 1`-adjusted.
PATTERN = re.compile(
    r"(max_num_blocks_per_req\s*=\s*cdiv\([^\n]*?\))(?!\s*\+\s*1)"
)


def main() -> int:
    with open(TARGET, "r", encoding="utf-8") as fh:
        src = fh.read()

    if re.search(r"max_num_blocks_per_req\s*=\s*cdiv\([^\n]*?\)\s*\+\s*1", src):
        print("[fix-dsa-indexer-block-table] already applied; nothing to do.")
        return 0

    if not PATTERN.search(src):
        print(
            "[fix-dsa-indexer-block-table] ERROR: anchor not found in "
            f"{TARGET}. vLLM version mismatch?",
            file=sys.stderr,
        )
        return 1

    patched = PATTERN.sub(r"\1 + 1", src, count=1)
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(patched)

    print(f"[fix-dsa-indexer-block-table] applied to {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
