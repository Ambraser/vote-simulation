"""Server lifecycle helpers — auto-shutdown when all browser tabs close.

This module is a proper Python module (cached by the import system), so its
module-level state persists across Streamlit script reruns.  The thread is
therefore guaranteed to start only once per process lifetime.
"""

from __future__ import annotations

import os
import threading
import time

_started = False
_lock = threading.Lock()

# Seconds of zero active sessions before the process exits.
# After a browser tab closes, Streamlit keeps the session alive for
# `disconnectedSessionTTL` seconds (set in config.toml).  We add a further
# grace period here before actually exiting so a page-refresh doesn't kill
# the server.
_GRACE_SECONDS = 20


def _auto_shutdown_worker() -> None:
    """Exit the process when no browser sessions remain for _GRACE_SECONDS."""
    time.sleep(10)  # startup grace — wait for at least one session to connect

    no_session_since: float | None = None

    while True:
        time.sleep(2)
        try:
            from streamlit.runtime import get_instance

            runtime = get_instance()
            if runtime is None:
                continue

            sm = getattr(runtime, "_session_mgr", None)
            if sm is None:
                continue

            # Streamlit's internal API varies slightly across versions.
            sessions: list = []
            for method_name in ("list_sessions", "get_active_sessions"):
                if hasattr(sm, method_name):
                    try:
                        sessions = getattr(sm, method_name)()
                    except Exception:
                        pass
                    break

            if not sessions:
                if no_session_since is None:
                    no_session_since = time.time()
                elif time.time() - no_session_since > _GRACE_SECONDS:
                    # All browser tabs have been closed long enough → shut down.
                    os._exit(0)
            else:
                no_session_since = None  # at least one active session
        except Exception:
            no_session_since = None


def ensure_auto_shutdown_started() -> None:
    """Start the auto-shutdown watcher thread (idempotent — safe to call on every Streamlit rerun)."""
    global _started
    if _started:
        return
    with _lock:
        if _started:
            return
        _started = True
    t = threading.Thread(target=_auto_shutdown_worker, daemon=True)
    t.start()
