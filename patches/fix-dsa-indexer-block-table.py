#!/usr/bin/env python3
"""Fix DSA indexer block-table overflow crash at concurrency >= 3.

Co-discovered independently with NVIDIA forum user ciprianveg
(thread 374125 posts 105/107, his `mods/fix-dsa-block-table-dim`).
Independently validated by Zatz at 4-concurrent on his cluster
(thread 375416).

Root cause
----------
The DeepSeek-Sparse-Attention indexer sizes its `expanded_block_table_buffer`
from `max_model_len` alone:

    max_num_blocks_per_req = cdiv(
        self.vllm_config.model_config.max_model_len,
        self.kv_cache_spec.block_size * get_total_cp_world_size(),
    )

But the scheduler's actual block table can be **one block wider** than this —
MTP speculative tokens and scheduler overhang push a request one block past the
`max_model_len`-derived count. When that happens (reliably at concurrency >= 3),
the engine crashes with:

    RuntimeError: The expanded size of the tensor (N) must match the existing
    size (N+1) at non-singleton dimension ...

The fix
-------
Add one block of headroom before allocating the buffer.

Implementation note: the `cdiv(...)` call is split across multiple lines in
this vLLM tree, so this patcher anchors on the exact multi-line text. (An
earlier revision of this file used a single-line regex and exited
"anchor not found" — thanks to Zatz for the report.) Idempotent: safe to run
more than once.
"""

import py_compile
import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py"

OLD = """        max_num_blocks_per_req = cdiv(
            self.vllm_config.model_config.max_model_len,
            self.kv_cache_spec.block_size * get_total_cp_world_size(),
        )
        self.expanded_block_table_buffer = torch.zeros("""

NEW = """        max_num_blocks_per_req = cdiv(
            self.vllm_config.model_config.max_model_len,
            self.kv_cache_spec.block_size * get_total_cp_world_size(),
        ) + 1  # fix: scheduler block table can exceed cdiv(max_model_len) by one (MTP/scheduler overhang)
        self.expanded_block_table_buffer = torch.zeros("""

MARKER = ") + 1  # fix: scheduler block table can exceed"


def main() -> int:
    with open(TARGET, "r", encoding="utf-8") as fh:
        src = fh.read()

    if MARKER in src:
        print("[fix-dsa-indexer-block-table] already applied; nothing to do.")
        return 0

    if OLD not in src:
        near = [
            line
            for line in src.splitlines()
            if "max_num_blocks_per_req" in line
            or "expanded_block_table_buffer" in line
        ]
        print(
            "[fix-dsa-indexer-block-table] ERROR: multi-line anchor not found in "
            f"{TARGET}. vLLM version mismatch? Nearby lines:\n  "
            + "\n  ".join(near[:6]),
            file=sys.stderr,
        )
        return 1

    src = src.replace(OLD, NEW, 1)
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(src)
    py_compile.compile(TARGET, doraise=True)
    print(
        "[fix-dsa-indexer-block-table] applied: "
        "expanded_block_table_buffer gets +1 block headroom."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
