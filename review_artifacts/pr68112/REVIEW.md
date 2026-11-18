# Review of vLLM PR #68112

Reviewed head: `cbc5b29ec71c645855551a133e0d22a96c2c35a4`.
This branch applies an exploratory optimization on top of that PR. It is not
a validated replacement for the upstream PR. AI assistance was used to review
the algorithm, prepare this candidate, and run the probes.

## Correctness

For a single transition from reasoning into the output grammar, the PR computes
the earliest ended nonempty prefix. Thus a later unaccepted draft suffix cannot
erase the constraint boundary. With the reported seven-token shape, the old
boundary is 8 and the corrected boundary is 4 (zero-based first constrained row).
The first reopen token at row 4 is rejected by the JSON mask before it can be
accepted by speculative verification.

The same helper serves draft trimming, prospective masks, and committed-token
acceptance. Trimming retains the first grammar-invalid token as a rejection
position. Masks after that invalid token may be unconstrained because ordered
speculative acceptance must stop at that first invalid position. If verification
stops before the reasoning close, no JSON is committed yet; a replacement that
completes the close makes the boundary the end of that output block.

This addresses the reported boundary inconsistency, not arbitrary malformed
output, response-header parsing, or a protocol allowing repeated free-form
reasoning blocks after grammar entry. Existing explicit/implicit end handling
and prompt initialization remain outside the changed fallback.

## Complexity and simplification

Let n be draft length, H be preceding history length, and m be the maximum K3
marker length. The PR calls the streaming predicate on delta lengths n through 1.
K3 copies each delta and constructs a window, so its delta work is quadratic.
Additionally, the first short slice of TokenIdsView materializes the entire
history. For fixed markers the current path costs O(H + n^2), not O(n).

The exploratory diff preserves the descending view scan but passes only the
token newly ending each prefix as delta. This removes the delta list and pop.
K3's existing streaming method supplies the previous m-1 context tokens, so it
still recognizes split markers. A bounded implementation of TokenIdsView indexing
removes the hidden full-history materialization. The resulting K3 work is
O(n*m), hence O(n) for fixed markers, independent of H.

Reversing the loop alone does not remove repeated prefix scans. Binary search
is invalid on the non-monotonic predicate. Stopping at the first match of the
descending scan would return a later boundary, not the earliest one.

This is not a universal linear-time guarantee for arbitrary parser plugins.
Some predicates scan/decode all input_ids regardless of delta size. Also, the
singleton-delta replacement needs compatibility verification for shared and
third-party predicates; the K3 comparisons below are not proof for all parsers.
No interface extension is proposed: vllm/reasoning/AGENTS.md explicitly disallows
changing the legacy ReasoningParser interface.

## Evidence and limits

Command, from the repository root:

```sh
.venv/bin/python -u review_artifacts/pr68112/probe.py > review_artifacts/pr68112/results.txt
```

The probe loads reference K3 streaming and token-view method bodies using
`git show` at the pinned PR commit, and candidate view code from this checkout.
It uses synthetic integer markers and a modeled manager loop.
It is algorithm-level evidence: no vLLM constructors, actual tokenizer, XGrammar,
rejection sampler, model weights, or server are loaded. Counts are operation/input
volume counts, not latency measurements.

- 72 targeted checks cover split markers, stale historical markers, multiple
  closures, no closure, and committed/uncommitted representation.
- 109,655 comparisons of PR versus singleton boundary search passed, including
  dense/partial markers, three-token and expanded markers, and unequal lengths.
- 36,000 candidate-view/list slice comparisons passed, with negative/strided
  slices, shortening, and prior materialization.
- n=8: cumulative delta volume falls from 36 to 8; n=32: 528 to 32.
- With an eight-token draft, counted historical elements accessed fall from H
  to 3 for the tested three-token K3 marker, including H=131072.

The PR's four mask tests and current-head model-serving validation remain
unverified here. Before adopting the candidate, extend the existing
tests/v1/test_request.py view tests, run real shared-parser and manager suites,
exercise trim -> mask -> speculative verification -> committed acceptance, and
measure relevant engine latency/output quality. For a user-visible regression,
tests/AGENTS.md also requires coverage through LLM or RemoteOpenAIServer.

## Design and review disposition

The first-boundary fix has appropriate small scope and includes rendered docs.
There is no reason to add protocol reopening or scheduler recovery to this PR.
The singleton candidate simplifies the manager but introduces a shared view
change, so total diff scope and compatibility evidence must be considered.
The existing PR transparently discloses its evidence gaps; neither that report
nor these probes justify an unconditional merge approval.
