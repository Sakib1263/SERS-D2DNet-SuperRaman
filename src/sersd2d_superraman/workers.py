"""File-backed worker logs and bounded failure messages for long notebook jobs."""

from pathlib import Path
import os
import signal
import subprocess


def run_worker(command, log_path, *, env=None, cwd=None):
    """Keep logs on disk; cancel the Linux worker group when a notebook interrupts.

    On Windows the direct worker is terminated. Process-tree cancellation on that
    platform has not been certified. A killed kernel requires a fresh run directory.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy() if env is None else dict(env)
    owns_group = os.name == "posix" and environment.get("_SERS_WORKER_GROUP") != "1"
    if owns_group:
        # Nested cascade workers stay in the outer notebook worker's group.
        environment["_SERS_WORKER_GROUP"] = "1"
    with log_path.open("xb") as log:
        with subprocess.Popen(
            list(map(str, command)),
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=owns_group,
        ) as process:
            try:
                status = process.wait()
            except BaseException:
                try:
                    if owns_group:
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                except ProcessLookupError:
                    pass  # The worker may have exited at the interrupt boundary.
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        if owns_group:
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                    except ProcessLookupError:
                        pass
                    process.wait()
                raise
    if status:
        with log_path.open("rb") as log:
            log.seek(max(0, log_path.stat().st_size - 6000))
            tail = log.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Worker exited with status {status}. Log: {log_path}\n{tail}")
