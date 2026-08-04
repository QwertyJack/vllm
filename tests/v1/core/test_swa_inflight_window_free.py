# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for freeing SWA blocks while GPU steps are in flight."""

import pytest
import torch

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler, mock_kv

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

NUM_PROMPT_TOKENS = 100
BLOCK_SIZE = 16
SLIDING_WINDOW = 16
NUM_OUT_OF_WINDOW_BLOCKS = 85 // BLOCK_SIZE


def _make_model_runner_output(
    scheduler_output: SchedulerOutput,
) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[0] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _create_swa_scheduler(async_scheduling: bool, use_kv_connector=False):
    return create_scheduler(
        block_size=BLOCK_SIZE,
        async_scheduling=async_scheduling,
        use_kv_connector=use_kv_connector,
        kv_cache_spec=SlidingWindowSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=SLIDING_WINDOW,
        ),
    )


def _num_null_blocks(scheduler, request_id: str) -> int:
    manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    null_block = manager._null_block
    return sum(block is null_block for block in manager.req_to_blocks[request_id])


def test_num_in_flight_tokens_accounting():
    scheduler = create_scheduler(async_scheduling=True)
    request = create_requests(num_requests=1, num_tokens=NUM_PROMPT_TOKENS)[0]
    scheduler.add_request(request)

    prefill_output = scheduler.schedule()
    assert request.num_in_flight_tokens == NUM_PROMPT_TOKENS

    decode_output = scheduler.schedule()
    assert request.num_in_flight_tokens == NUM_PROMPT_TOKENS + 1

    scheduler.update_from_output(
        prefill_output, _make_model_runner_output(prefill_output)
    )
    assert request.num_in_flight_tokens == 1

    scheduler.update_from_output(
        decode_output, _make_model_runner_output(decode_output)
    )
    assert request.num_in_flight_tokens == 0


def test_swa_free_waits_for_in_flight_step():
    scheduler = _create_swa_scheduler(async_scheduling=True)
    request = create_requests(
        num_requests=1,
        num_tokens=NUM_PROMPT_TOKENS,
        block_size=BLOCK_SIZE,
    )[0]
    scheduler.add_request(request)
    request_id = request.request_id
    block_pool = scheduler.kv_cache_manager.block_pool

    prefill_output = scheduler.schedule()
    free_after_prefill = block_pool.get_num_free_blocks()

    scheduler.schedule()
    assert _num_null_blocks(scheduler, request_id) == 0
    assert block_pool.get_num_free_blocks() == free_after_prefill

    scheduler.update_from_output(
        prefill_output, _make_model_runner_output(prefill_output)
    )
    scheduler.schedule()
    assert _num_null_blocks(scheduler, request_id) == NUM_OUT_OF_WINDOW_BLOCKS
    assert (
        block_pool.get_num_free_blocks()
        == free_after_prefill + NUM_OUT_OF_WINDOW_BLOCKS
    )


def test_swa_free_immediate_when_sync():
    scheduler = _create_swa_scheduler(async_scheduling=False)
    request = create_requests(
        num_requests=1,
        num_tokens=NUM_PROMPT_TOKENS,
        block_size=BLOCK_SIZE,
    )[0]
    scheduler.add_request(request)

    prefill_output = scheduler.schedule()
    scheduler.update_from_output(
        prefill_output, _make_model_runner_output(prefill_output)
    )
    assert request.num_in_flight_tokens == 0

    scheduler.schedule()

    assert _num_null_blocks(scheduler, request.request_id) == NUM_OUT_OF_WINDOW_BLOCKS


def test_swa_admission_cap_accounts_for_overlapping_batches():
    scheduler = create_scheduler(
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=512,
        async_scheduling=True,
        kv_cache_spec=SlidingWindowSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=512,
        ),
    )
    manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]

    assert manager._max_admission_blocks_per_request == 97


def test_connector_finish_frees_on_processed_token_basis():
    scheduler = _create_swa_scheduler(
        async_scheduling=True,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    request = create_requests(
        num_requests=1,
        num_tokens=NUM_PROMPT_TOKENS,
        block_size=BLOCK_SIZE,
    )[0]
    scheduler.add_request(request)

    prefill_output = scheduler.schedule()
    scheduler.schedule()

    scheduler._connector_finished(request)
    assert _num_null_blocks(scheduler, request.request_id) == 0

    scheduler.update_from_output(
        prefill_output, _make_model_runner_output(prefill_output)
    )
    scheduler._connector_finished(request)
    assert _num_null_blocks(scheduler, request.request_id) == NUM_OUT_OF_WINDOW_BLOCKS
