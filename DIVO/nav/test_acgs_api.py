"""LLM transport failures are isolated without hiding configuration errors."""
import httpx
from openai import APIConnectionError

from nav.curriculum.acgs_api import NavACGS


def test_recoverable_connection_error_becomes_failed_candidate():
    acgs = NavACGS(provider="openai")

    def fail_request(_system, _user):
        raise APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))

    acgs._call_llm = fail_request
    executor, reason = acgs.generate_code("system", "user")
    assert executor is None
    assert reason.startswith("llm_request_failed:APIConnectionError:")


def test_programming_or_configuration_error_is_not_hidden():
    acgs = NavACGS(provider="openai")

    def fail_request(_system, _user):
        raise RuntimeError("bad configuration")

    acgs._call_llm = fail_request
    try:
        acgs.generate_code("system", "user")
    except RuntimeError as exc:
        assert "bad configuration" in str(exc)
    else:
        raise AssertionError("non-recoverable errors must propagate")


def main():
    test_recoverable_connection_error_becomes_failed_candidate()
    test_programming_or_configuration_error_is_not_hidden()
    print("ALL PASS")


if __name__ == "__main__":
    main()
