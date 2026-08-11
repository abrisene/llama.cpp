from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_cache_prefix_route_is_registered_and_router_proxied():
    server_cpp = _read("tools/server/server.cpp")
    context_h = _read("tools/server/server-context.h")

    assert "server_http_context::handler_t post_cache_prefix;" in context_h
    assert 'ctx_http.post("/cache/prefix",             ex_wrapper(routes.post_cache_prefix));' in server_cpp
    assert "routes.post_cache_prefix           = models_routes->proxy_post;" in server_cpp


def test_cache_prefix_single_model_fallback_documents_missing_capture_seam():
    server_cpp = _read("tools/server/server.cpp")

    assert "cache_prefix_not_implemented_handler" in server_cpp
    assert "requires the server-side explicit prefix capture handler" in server_cpp
    assert "if (!routes.post_cache_prefix)" in server_cpp
