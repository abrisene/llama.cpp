"""Server-level coverage for the persistent token-aligned prefix cache.

These tests deliberately use a token-array prompt.  That makes the block
boundary part of the test input instead of relying on a particular text
tokenization shape for the tiny fixture model.
"""

from pathlib import Path

import pytest

from utils import *


BLOCK_SIZE = 256
CAPTURE_TOKENS = BLOCK_SIZE * 2 + 1


def _new_prefix_cache_server(cache_root: Path) -> ServerProcess:
    server = ServerPreset.tinyllama2()
    # The preset's 512-token context is split across two slots.  Keep one slot
    # and a larger context so that two minimum-size blocks fit in one request.
    server.n_ctx = 1024
    server.n_slots = 1
    server.n_batch = 32
    server.n_predict = 1
    server.temperature = 0.0
    server.cache_disk_path = str(cache_root)
    server.cache_disk_size_mib = 64
    server.cache_block_size = BLOCK_SIZE
    server.cache_capture_mode = "always"
    server.cache_ram = 64
    # The existing slot-action route gates erase behind --slot-save-path,
    # even though this test only uses erase to make the destination sequence
    # empty before a persistent restore.  Point it at the pytest-owned parent.
    server.slot_save_path = str(cache_root.parent)
    server.endpoint_props = True
    return server


def _token_prompt(server: ServerProcess, count: int) -> list[int]:
    # A repeated, space-prefixed word gives us substantially more than the
    # requested number of tokens.  The server accepts a token array directly,
    # so truncating it gives an exact input length without a BOS/tokenizer
    # ambiguity in the completion request.
    if server.process is None:
        server.start()
    text = " prefix" * (count * 4)
    response = server.make_request("POST", "/tokenize", data={
        "content": text,
        "add_special": True,
    })
    assert response.status_code == 200
    tokens = response.body["tokens"]
    assert len(tokens) >= count
    return tokens[:count]


def _prefix_props(server: ServerProcess) -> dict:
    response = server.make_request("GET", "/props")
    assert response.status_code == 200
    prefix_cache = response.body["prefix_cache"]
    assert prefix_cache["enabled"] is True
    assert prefix_cache["block_size"] == BLOCK_SIZE
    assert "store" in prefix_cache
    return prefix_cache


def _wait_for_disk_artifacts(server: ServerProcess, minimum: int = 2) -> dict:
    deadline = time.monotonic() + 30
    latest = _prefix_props(server)
    while time.monotonic() < deadline:
        latest = _prefix_props(server)
        if latest["store"]["disk_artifacts"] >= minimum:
            return latest
        time.sleep(0.05)
    raise AssertionError(f"persistent prefix artifacts were not published: {latest}")


def _completion(server: ServerProcess, tokens: list[int], id_slot: int = 0) -> dict:
    response = server.make_request("POST", "/completion", data={
        "prompt": tokens,
        "id_slot": id_slot,
        "cache_prompt": True,
        "n_predict": 1,
        "temperature": 0.0,
    })
    assert response.status_code == 200
    assert "timings" in response.body
    return response.body


def _completion_response(server: ServerProcess, tokens: list[int], **extra) -> ServerResponse:
    data = {
        "prompt": tokens,
        "id_slot": 0,
        "cache_prompt": True,
        "n_predict": 1,
        "temperature": 0.0,
    }
    data.update(extra)
    return server.make_request("POST", "/completion", data=data)


def _erase_slot(server: ServerProcess) -> None:
    response = server.make_request("POST", "/slots/0?action=erase")
    assert response.status_code == 200


def _capture(server: ServerProcess, prompt: list[int]) -> dict:
    if server.process is None:
        server.start()
    first = _completion(server, prompt)
    stored = _wait_for_disk_artifacts(server)
    assert stored["capture_attempts"] > 0
    assert stored["store"]["published"] >= 2
    return first


def test_prefix_cache_cold_capture_and_warm_restore(tmp_path: Path):
    server = _new_prefix_cache_server(tmp_path / "cache")
    prompt = _token_prompt(server, CAPTURE_TOKENS)

    first = _capture(server, prompt)
    assert first["timings"]["cache_n"] == 0
    assert first["timings"]["prompt_n"] >= BLOCK_SIZE * 2

    _erase_slot(server)
    before = _prefix_props(server)["store"]
    second = _completion(server, prompt)
    after = _prefix_props(server)["store"]

    # A non-aligned suffix remains a normal prompt evaluation, while both
    # complete prefix blocks are restored from the hot tier.
    assert second["timings"]["cache_n"] >= BLOCK_SIZE * 2
    assert second["timings"]["prompt_n"] < first["timings"]["prompt_n"]
    assert after["hot_hits"] > before["hot_hits"]


def test_prefix_cache_restart_uses_disk_artifacts(tmp_path: Path):
    cache_root = tmp_path / "cache"
    first_server = _new_prefix_cache_server(cache_root)
    prompt = _token_prompt(first_server, CAPTURE_TOKENS)
    _capture(first_server, prompt)
    first_server.stop()  # flushes the bounded writer before the restart

    server = _new_prefix_cache_server(cache_root)
    server.start()
    restarted = _completion(server, prompt)
    props = _prefix_props(server)

    assert restarted["timings"]["cache_n"] >= BLOCK_SIZE * 2
    assert props["store"]["disk_hits"] >= 2


def test_prefix_cache_exact_block_aligned_prompt_walks_back_one_block(tmp_path: Path):
    server = _new_prefix_cache_server(tmp_path / "cache")
    captured_prompt = _token_prompt(server, CAPTURE_TOKENS)
    _capture(server, captured_prompt)

    # Clearing hot forces this request through the persistent lookup path, and
    # the exact 512-token prompt exercises the deliberate "leave one fresh
    # token" rule: a complete boundary is not restored, so it walks back to
    # the previous block (256 tokens).
    clear_hot = server.make_request("POST", "/props", data={"action": "prefix_cache.clear_hot"})
    assert clear_hot.status_code == 200
    assert _prefix_props(server)["store"]["hot_artifacts"] == 0
    _erase_slot(server)

    exact = _completion(server, captured_prompt[: BLOCK_SIZE * 2])
    props = _prefix_props(server)
    assert exact["timings"]["cache_n"] == BLOCK_SIZE
    assert exact["timings"]["prompt_n"] == BLOCK_SIZE
    assert props["tokens_restored"] >= BLOCK_SIZE
    assert props["store"]["disk_hits"] >= 1


@pytest.mark.parametrize("mutation", ["missing", "corrupt"])
def test_prefix_cache_missing_or_corrupt_artifacts_fail_closed(tmp_path: Path, mutation: str):
    server = _new_prefix_cache_server(tmp_path / "cache")
    prompt = _token_prompt(server, CAPTURE_TOKENS)
    _capture(server, prompt)

    clear_hot = server.make_request("POST", "/props", data={"action": "prefix_cache.clear_hot"})
    assert clear_hot.status_code == 200
    artifact_files = list((tmp_path / "cache").rglob("*.bin"))
    assert artifact_files
    if mutation == "missing":
        for artifact in artifact_files:
            artifact.unlink()
    else:
        # Corrupt every immutable artifact so neither the full chain nor a
        # shorter candidate can be restored.  The store must turn each one
        # into a miss instead of handing partial bytes to llama state restore.
        for artifact in artifact_files:
            payload = bytearray(artifact.read_bytes())
            assert payload
            payload[-1] ^= 0x01
            artifact.write_bytes(payload)

    _erase_slot(server)
    failed = _completion(server, prompt)
    props = _prefix_props(server)
    assert failed["timings"]["cache_n"] == 0
    assert failed["timings"]["prompt_n"] >= BLOCK_SIZE * 2
    if mutation == "corrupt":
        assert props["store"]["corrupt"] > 0


def test_prefix_cache_props_and_clear_actions(tmp_path: Path):
    server = _new_prefix_cache_server(tmp_path / "cache")
    prompt = _token_prompt(server, CAPTURE_TOKENS)
    _capture(server, prompt)

    before = _prefix_props(server)
    assert before["disk_size_mib"] == 64
    assert before["hot_size_mib"] == 64
    assert before["store"]["disk_artifacts"] >= 2
    assert before["store"]["hot_artifacts"] >= 2

    clear_hot = server.make_request("POST", "/props", data={"action": "prefix_cache.clear_hot"})
    assert clear_hot.status_code == 200
    after_hot = _prefix_props(server)
    assert after_hot["store"]["hot_artifacts"] == 0
    assert after_hot["store"]["disk_artifacts"] >= 2

    clear_disk = server.make_request("POST", "/props", data={"action": "prefix_cache.clear_disk"})
    assert clear_disk.status_code == 200
    after_disk = _prefix_props(server)
    assert after_disk["store"]["disk_artifacts"] == 0
    assert after_disk["store"]["disk_bytes"] == 0


def test_prefix_cache_repeat_mode_first_observation_publishes_zero_then_second_cold_observation_captures(tmp_path: Path):
    server = _new_prefix_cache_server(tmp_path / "cache")
    server.cache_capture_mode = "repeat"
    prompt = _token_prompt(server, CAPTURE_TOKENS)

    first = _completion_response(server, prompt)
    assert first.status_code == 200
    first_props = _prefix_props(server)
    assert first_props["capture_mode"] == "repeat"
    assert first_props["boundaries_observed"] >= 2
    assert first_props["first_observations"] >= 2
    assert first_props["store"]["published"] == 0
    assert first.body["persistent_cache"]["published_blocks"] == 0

    _erase_slot(server)
    second = _completion_response(server, prompt)
    assert second.status_code == 200
    stored = _wait_for_disk_artifacts(server)
    assert stored["repeat_admissions"] >= 2
    assert stored["store"]["published"] >= 2
    assert second.body["persistent_cache"]["staged_blocks"] >= 2
    assert second.body["persistent_cache"]["published_blocks"] == 0


def test_prefix_cache_repeat_mode_captures_second_observation_with_resident_slot_reuse(tmp_path: Path):
    server = _new_prefix_cache_server(tmp_path / "cache")
    server.cache_capture_mode = "repeat"
    prompt = _token_prompt(server, CAPTURE_TOKENS)

    first = _completion_response(server, prompt)
    assert first.status_code == 200
    assert first.body["persistent_cache"]["published_blocks"] == 0
    assert _prefix_props(server)["store"]["published"] == 0

    second = _completion_response(server, prompt)
    assert second.status_code == 200
    stored = _wait_for_disk_artifacts(server)
    assert stored["repeat_admissions"] >= 2
    assert stored["store"]["published"] >= 2
    assert second.body["persistent_cache"]["staged_blocks"] >= 2
    assert second.body["persistent_cache"]["published_blocks"] == 0
    assert second.body["persistent_cache"]["reason"] == ""


def test_prefix_cache_explicit_capture_first_observation_bypasses_frequency_admission(tmp_path: Path):
    server = _new_prefix_cache_server(tmp_path / "cache")
    server.cache_capture_mode = "explicit"
    prompt = _token_prompt(server, CAPTURE_TOKENS)

    response = _completion_response(server, prompt, cache_persist=True)
    assert response.status_code == 200
    stored = _wait_for_disk_artifacts(server)
    assert response.body["persistent_cache"]["requested"] is True
    assert response.body["persistent_cache"]["eligible"] is True
    assert response.body["persistent_cache"]["staged_blocks"] >= 2
    assert response.body["persistent_cache"]["durable"] is False
    assert stored["explicit_admissions"] >= 2
    assert stored["store"]["published"] >= 2


def test_prefix_cache_explicit_capture_does_not_bypass_ineligible_request_gates(tmp_path: Path):
    server = _new_prefix_cache_server(tmp_path / "cache")
    server.enable_ctx_shift = True
    prompt = _token_prompt(server, CAPTURE_TOKENS)

    response = _completion_response(server, prompt, cache_persist=True)
    assert response.status_code == 200
    props = _prefix_props(server)
    assert response.body["persistent_cache"]["requested"] is True
    assert response.body["persistent_cache"]["eligible"] is False
    assert response.body["persistent_cache"]["reason"] == "context_shift"
    assert props["store"]["published"] == 0


def test_prefix_cache_endpoint_returns_durable_artifacts_reused_after_restart(tmp_path: Path):
    cache_root = tmp_path / "cache"
    server = _new_prefix_cache_server(cache_root)
    server.cache_capture_mode = "explicit"
    prompt = _token_prompt(server, CAPTURE_TOKENS)

    precache = server.make_request("POST", "/cache/prefix", data={
        "prompt": prompt,
        "cache_prompt": True,
        "temperature": 0.0,
    })
    assert precache.status_code == 200
    assert precache.body["tokens_evaluated"] == CAPTURE_TOKENS
    assert precache.body["boundary"] == BLOCK_SIZE * 2
    assert precache.body["attention_blocks"] == 2
    assert precache.body["recurrent_sidecars"] >= 0
    assert precache.body["durable"] is True
    first_props = _prefix_props(server)
    assert first_props["explicit_precache_requests"] == 1
    assert first_props["explicit_precache_successes"] == 1
    assert first_props["explicit_precache_failures"] == 0
    assert first_props["store"]["disk_artifacts"] >= 2

    duplicate = server.make_request("POST", "/cache/prefix", data={
        "prompt": prompt,
        "cache_prompt": True,
        "temperature": 0.0,
    })
    assert duplicate.status_code == 200
    assert duplicate.body["tokens_evaluated"] == CAPTURE_TOKENS
    assert duplicate.body["boundary"] == BLOCK_SIZE * 2
    assert duplicate.body["attention_blocks"] == 2
    assert duplicate.body["recurrent_sidecars"] == precache.body["recurrent_sidecars"]
    assert duplicate.body["bytes_published"] == 0
    assert duplicate.body["durable"] is True
    second_props = _prefix_props(server)
    assert second_props["explicit_precache_requests"] == 2
    assert second_props["explicit_precache_successes"] == 2
    assert second_props["explicit_precache_failures"] == 0
    assert second_props["store"]["disk_artifacts"] >= 2
    server.stop()

    restarted = _new_prefix_cache_server(cache_root)
    restarted.cache_capture_mode = "explicit"
    restarted.start()
    restored = _completion(restarted, prompt)
    props = _prefix_props(restarted)
    assert restored["timings"]["cache_n"] >= BLOCK_SIZE * 2
    assert props["store"]["disk_hits"] >= 2


def test_prefix_cache_endpoint_counts_ineligible_failure(tmp_path: Path):
    server = _new_prefix_cache_server(tmp_path / "cache")
    server.enable_ctx_shift = True
    prompt = _token_prompt(server, CAPTURE_TOKENS)

    precache = server.make_request("POST", "/cache/prefix", data={
        "prompt": prompt,
        "cache_prompt": True,
        "temperature": 0.0,
    })
    assert precache.status_code == 400
    props = _prefix_props(server)
    assert props["explicit_precache_requests"] == 1
    assert props["explicit_precache_successes"] == 0
    assert props["explicit_precache_failures"] == 1
    assert props["store"]["published"] == 0
