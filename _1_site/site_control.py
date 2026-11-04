"""Start and stop this checkout's local website on Linux."""
import os
from pathlib import Path
import time
import signal
import sys

SITE = Path(__file__).resolve().parent


def is_site_process(pid):
    """Match Python entry points, never shells or unrelated port owners."""
    proc = Path("/proc") / str(pid)
    try:
        args = (proc / "cmdline").read_bytes().split(b"\0")
        args = [os.fsdecode(arg) for arg in args if arg]
        if not args or not Path(args[0]).name.startswith("python"):
            return False
        # Current launcher.
        if len(args) >= 3 and args[-2] == str(SITE / "site_control.py") and args[-1] == "launch" and "-c" not in args:
            return True
        # Original launch_site.sh used python -u -c with SITE as its last argument.
        if "-c" in args and args[-1] == str(SITE):
            code = args[args.index("-c") + 1]
            return "from app import app" in code and "make_server" in code
        # Also recognize python3 _1_site/app.py started by hand.
        scripts = [arg for arg in args[1:] if not arg.startswith("-")]
        if scripts and Path(scripts[0]).name == "app.py":
            script = Path(scripts[0])
            if not script.is_absolute():
                script = (proc / "cwd").resolve() / script
            return script.resolve() == SITE / "app.py"
        return False
    except (FileNotFoundError, ProcessLookupError):
        return False


def process_identity(pid):
    try:
        # Fields after the closing comm parenthesis start at field 3.
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return fields[19]  # /proc stat field 22: process start time
    except (FileNotFoundError, ProcessLookupError):
        return None


def wait_for_exit(pid, identity):
    deadline = time.monotonic() + 5
    while process_identity(pid) == identity:
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def stop_site():
    stopped = 0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
        except FileNotFoundError:
            continue
        pid = int(proc.name)
        identity = process_identity(pid)
        if identity is None or not is_site_process(pid):
            continue
        try:
            if process_identity(pid) != identity:
                continue
            os.kill(pid, signal.SIGTERM)
            if not wait_for_exit(pid, identity):
                if process_identity(pid) == identity and is_site_process(pid):
                    os.kill(pid, signal.SIGKILL)
                if not wait_for_exit(pid, identity):
                    raise RuntimeError(f"Site process {pid} did not exit")
        except ProcessLookupError:
            pass
        print(f"Stopped ASTRID site (PID {pid}).", flush=True)
        stopped += 1
    if not stopped:
        print("No running ASTRID site found.", flush=True)


def launch_site():
    stop_site()
    from werkzeug.serving import make_server
    from app import app

    with make_server("127.0.0.1", 8000, app, threaded=True) as server:
        url = "http://127.0.0.1:8000"
        print(f"\nASTRID is running: \033]8;;{url}\033\\{url}\033]8;;\033\\", flush=True)
        print("Ctrl+click the link in VS Code to open the site.")
        print("Press Ctrl+C to stop.\n", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    if sys.argv[1:] == ["launch"]:
        launch_site()
    elif sys.argv[1:] == ["stop"]:
        stop_site()
    else:
        raise SystemExit("Usage: site_control.py {launch|stop}")
