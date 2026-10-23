"""Bound subprocess trees, including torchrun worker processes."""

import os
import signal
import subprocess


def bounded_run(command, *, timeout=180, env=None, cwd=None):
    options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
               else {"start_new_session": True})
    process = subprocess.Popen(command, env=env, cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, **options)
    try:
        output, _ = process.communicate(timeout=timeout)
    except BaseException:
        if os.name == "nt":
            try:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, timeout=10, check=False)
            except (OSError, subprocess.SubprocessError):
                pass
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.communicate(timeout=10)
        raise
    result = subprocess.CompletedProcess(command, process.returncode, output)
    if process.returncode:
        print(output, end="")
    result.check_returncode()
    return result
