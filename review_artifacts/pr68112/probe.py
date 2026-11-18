# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Algorithm probe, not a vLLM integration test or a serving benchmark.

Execute selected, unchanged method bodies from PR cbc5b29ec7 with integer
marker sequences. No model, tokenizer, grammar backend, or sampler is loaded.
"""

from __future__ import annotations

import ast
import itertools
import random
import subprocess
from array import array
from collections.abc import Collection, Iterable, Iterator, Sequence
from pathlib import Path
from typing import overload

ROOT = Path(__file__).resolve().parents[2]
PR_HEAD = "cbc5b29ec71c645855551a133e0d22a96c2c35a4"
SOURCE_PATHS = {
    "token_buffer.py": "vllm/v1/token_buffer.py",
    "kimi.py": "vllm/reasoning/kimi_k3_reasoning_parser.py",
}


def source_nodes(filename, names):
    if filename == "candidate_token_buffer.py":
        source = (ROOT / SOURCE_PATHS["token_buffer.py"]).read_text()
    else:
        source = subprocess.check_output(
            ["git", "show", f"{PR_HEAD}:{SOURCE_PATHS[filename]}"],
            cwd=ROOT,
            text=True,
        )
    tree = ast.parse(source)
    return [node for node in tree.body if getattr(node, "name", None) in names]


namespace = {
    "array": array,
    "as_list": list,
    "Collection": Collection,
    "Iterable": Iterable,
    "Iterator": Iterator,
    "Sequence": Sequence,
    "overload": overload,
}
buffer_nodes = source_nodes(
    "token_buffer.py", {"TokenIdsView", "rfind_tokens", "rfind_token_sequences"}
)
kimi_nodes = source_nodes("kimi.py", {"_newest_marker"})
kimi_class = source_nodes("kimi.py", {"KimiK3ReasoningParser"})[0]
method = next(
    node
    for node in kimi_class.body
    if getattr(node, "name", None) == "is_reasoning_end_streaming"
)
module = ast.Module(body=buffer_nodes + kimi_nodes + [method], type_ignores=[])
exec(compile(module, "selected_pr_source", "exec"), namespace)
TokenIdsView = namespace["TokenIdsView"]
streaming_check = namespace["is_reasoning_end_streaming"]


class BoundaryInputs:
    def __init__(self, close, opened):
        self._thinking_enabled = True
        self._think_close_ids = close
        self._think_open_ids = opened
        self.calls = 0
        self.delta_volume = 0

    def is_reasoning_end_streaming(self, ids, delta):
        self.calls += 1
        self.delta_volume += len(delta)
        return streaming_check(self, ids, delta)


candidate_namespace = dict(namespace)
candidate_module = ast.Module(
    body=source_nodes("candidate_token_buffer.py", {"TokenIdsView"}),
    type_ignores=[],
)
exec(compile(candidate_module, "candidate_view_source", "exec"), candidate_namespace)
CandidateView = candidate_namespace["TokenIdsView"]


def boundary(history, draft, reasoner, mode="pr", committed=False, view=TokenIdsView):
    if not draft:
        return 1
    ids = view(history + draft if committed else history, () if committed else draft)
    delta = list(draft)
    if mode == "old":
        if not reasoner.is_reasoning_end_streaming(ids, delta):
            return len(draft) + 1
        for i in range(len(draft) - 1, 0, -1):
            ids.pop()
            delta.pop()
            if not reasoner.is_reasoning_end_streaming(ids, delta):
                return i + 1
        return 1
    start = len(draft) + 1
    for i in range(len(draft), 0, -1):
        query = (draft[i - 1],) if mode == "single" else delta
        if reasoner.is_reasoning_end_streaming(ids, query):
            start = i
        ids.pop()
        delta.pop()
    return start


def first_close(history, draft, close):
    """Independent finite-marker reference: first close completed by this block."""
    for end in range(1, len(draft) + 1):
        joined = history + draft[:end]
        if joined[-len(close) :] == close:
            return end
    return len(draft) + 1


class CountedHistory(Sequence):
    def __init__(self, length):
        self.length = length
        self.reads = 0

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            count = len(range(*index.indices(self.length)))
            self.reads += count
            return [9] * count
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        self.reads += 1
        return 9


def count_history_reads(length, view):
    history = CountedHistory(length)
    ids = view(history, [9] * 8)
    parser = BoundaryInputs([4, 2, 3], [1, 2, 3])
    for token in reversed([9] * 8):
        parser.is_reasoning_end_streaming(ids, (token,))
        ids.pop()
    return history.reads


def main():
    close, opened = [4, 2, 3], [1, 2, 3]
    draft = [9] + close + opened
    print("Recorded shape, normalized marker IDs:")
    for mode in ("old", "pr", "single"):
        print(mode, boundary(opened, draft, BoundaryInputs(close, opened), mode))

    # End markers split before any of their tokens, two closures, partial
    # markers, a stale historical closure followed by a new turn, and no close.
    scenarios = []
    for split in range(len(close)):
        history = close + opened + [9] + close[:split]
        for suffix in ([], opened, opened + close, opened[:1], [9] * 8):
            scenarios.append((history, close[split:] + suffix))
    scenarios += [(opened, []), (opened, [9] * 12), (opened, [4, 2])]
    for history, draft in scenarios:
        expected = first_close(history, draft, close)
        for committed in (False, True):
            for mode in ("pr", "single"):
                assert (
                    boundary(
                        history,
                        draft,
                        BoundaryInputs(close, opened),
                        mode,
                        committed,
                        CandidateView,
                    )
                    == expected
                )
    print("Targeted committed/uncommitted scenarios:", len(scenarios) * 4, "passed")

    # Enumerate suffixes around partial and historical markers. Then add
    # longer marker-dense random drafts. Compare the PR and singleton calls.
    histories = [[], opened, close + opened, opened + [4], opened + [4, 2]]
    cases = 0
    for size in range(7):
        for tokens in itertools.product((1, 2, 3, 4, 9), repeat=size):
            draft = list(tokens)
            for history in histories:
                expected = boundary(history, draft, BoundaryInputs(close, opened))
                actual = boundary(
                    history,
                    draft,
                    BoundaryInputs(close, opened),
                    "single",
                    view=CandidateView,
                )
                assert actual == expected, (history, draft, expected, actual)
                cases += 1
    rnd = random.Random(68112)
    for _ in range(10000):
        history = [rnd.choice((1, 2, 3, 4, 9)) for _ in range(rnd.randrange(15))]
        draft = [rnd.choice((1, 2, 3, 4, 9)) for _ in range(rnd.randrange(40))]
        expected = boundary(history, draft, BoundaryInputs(close, opened))
        actual = boundary(
            history, draft, BoundaryInputs(close, opened), "single", view=CandidateView
        )
        assert actual == expected, (history, draft, expected, actual)
        cases += 1
    # The PR's GPT-2 fixture expands the marker strings; the algorithm must
    # not depend on the native 3-token length. Include unequal lengths too.
    for close, opened in (
        ([4] + [2] * 10 + [3], [1] + [2] * 10 + [3]),
        ([4, 2, 3], [1, 2, 2, 3]),
    ):
        for _ in range(1000):
            chunks = [
                rnd.choice(([9], close, opened, close[:-1], opened[:-1]))
                for _ in range(rnd.randrange(1, 8))
            ]
            full = [token for chunk in chunks for token in chunk]
            split = rnd.randrange(len(full) + 1)
            history, draft = full[:split], full[split:]
            expected = boundary(history, draft, BoundaryInputs(close, opened))
            actual = boundary(
                history,
                draft,
                BoundaryInputs(close, opened),
                "single",
                view=CandidateView,
            )
            assert actual == expected, (history, draft, expected, actual)
            cases += 1
    close, opened = [4, 2, 3], [1, 2, 3]
    print("K3 PR/single-token boundary comparisons:", cases, "passed")

    # Check candidate view indexing independently against list semantics,
    # including reversed slices, negative indices, pop, and cached iteration.
    view_cases = 0
    for a in range(5):
        for b in range(5):
            for cached in (False, True):
                original = list(range(a + b))
                view = CandidateView(original[:a], original[a:])
                if cached:
                    assert list(view) == original
                for remaining in range(a + b, -1, -1):
                    expected = original[:remaining]
                    for start, stop, step in itertools.product(
                        (None, -10, -2, 0, 2, 10),
                        (None, -10, -2, 0, 2, 10),
                        (-3, -1, 1, 2),
                    ):
                        sl = slice(start, stop, step)
                        assert view[sl] == expected[sl]
                        view_cases += 1
                    for i in range(-remaining, remaining):
                        assert view[i] == expected[i]
                    if remaining:
                        view.pop()
    print("Candidate view/list slice comparisons:", view_cases, "passed")

    print("No-close draft: n, PR delta volume, single-token delta volume")
    for n in (4, 8, 16, 32, 64, 128):
        pr, single = BoundaryInputs(close, opened), BoundaryInputs(close, opened)
        boundary(opened, [9] * n, pr)
        boundary(opened, [9] * n, single, "single", view=CandidateView)
        print(n, pr.delta_volume, single.delta_volume)
    print("History length, existing-view reads, bounded-slice-view reads")
    for length in (128, 8192, 131072):
        print(
            length,
            count_history_reads(length, TokenIdsView),
            count_history_reads(length, CandidateView),
        )


if __name__ == "__main__":
    main()
