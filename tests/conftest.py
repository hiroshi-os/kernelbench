import pytest


def pytest_addoption(parser):
    parser.addoption("--require-gpu", action="store_true", help="Fail if GPU is unavailable or any selected test skips")


def pytest_sessionstart(session):
    if session.config.getoption("--require-gpu"):
        from kernelbench.ops import gpu_status
        reason = gpu_status()
        if reason:
            raise pytest.UsageError(f"GPU validation required: {reason}")


def pytest_sessionfinish(session, exitstatus):
    if session.config.getoption("--require-gpu"):
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if session.testscollected == 0 or (reporter and reporter.stats.get("skipped")):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
