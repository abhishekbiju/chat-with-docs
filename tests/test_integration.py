from __future__ import annotations

import json
from contextlib import asynccontextmanager


def test_ready_endpoint_returns_expected_shape(mocker, integration_client) -> None:
    class FakeProbe:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def get(self, url: str):
            class R:
                is_success = True

            return R()

    mocker.patch("app.main.httpx.AsyncClient", FakeProbe)

    r = integration_client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert "index_documents" in body
    assert "ollama_reachable" in body
    assert "ready" in body
    assert isinstance(body["index_documents"], int)
    assert body["index_documents"] > 0
    assert body["ollama_reachable"] is True
    assert body["ready"] is True


def test_chat_stream_sse_format_with_mocked_ollama(mocker, integration_client) -> None:
    lines = [
        '{"message":{"content":"Hel"},"done":false}',
        '{"message":{"content":"lo."},"done":false}',
        '{"message":{"content":""},"done":true}',
    ]

    class FakeStreamResp:
        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            for line in lines:
                yield line

    @asynccontextmanager
    async def fake_stream(*args, **kwargs):
        yield FakeStreamResp()

    rag = integration_client.app.state.rag
    mocker.patch.object(rag.httpx_client, "stream", side_effect=fake_stream)

    with integration_client.stream(
        "POST",
        "/chat/stream",
        json={"query": "What is the recovery code for vault nine?"},
    ) as r:
        assert r.status_code == 200
        raw = b"".join(r.iter_bytes())

    text = raw.decode()
    assert "data: " in text
    assert '"token"' in text
    assert "event: done" in text

    tokens_concat = ""
    done_payload: dict | None = None
    for block in text.strip().split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                payload_s = line.partition("data:")[2].strip()
                if not payload_s:
                    continue
                try:
                    obj = json.loads(payload_s)
                except json.JSONDecodeError:
                    continue
                if "token" in obj:
                    tokens_concat += obj["token"]
        if "event: done" in block:
            for line in block.split("\n"):
                if line.startswith("data:"):
                    payload_s = line.partition("data:")[2].strip()
                    if payload_s.startswith("{"):
                        done_payload = json.loads(payload_s)

    assert tokens_concat == "Hello."
    assert done_payload is not None
    assert done_payload.get("answer") == "Hello."
    assert isinstance(done_payload.get("sources"), list)
    assert len(done_payload["sources"]) >= 1
