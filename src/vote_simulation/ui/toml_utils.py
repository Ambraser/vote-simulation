"""Bijection between the current UI state (session_state) and a valid TOML file.

The UI state is a flat dict with the following keys:

    output_base_path : str
    seed             : int
    generative_models: list[str]
    rule_codes       : list[str]
    candidates       : list[int]
    voters           : list[int]
    iterations       : int
    generator_params : dict[str, dict[str, object]]   # per-model sub-tables
    input_folder_path: str | None
"""

from __future__ import annotations

import copy
import os
import queue
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

# ---------------------------------------------------------------------------
# Shared thread utility
# ---------------------------------------------------------------------------


class QueueWriter:
    """Redirects sys.stdout to a queue — used by generation/simulation threads."""

    def __init__(self, q: queue.Queue[str]) -> None:
        self._q = q

    def write(self, text: str) -> None:
        text = text.strip()
        if text:
            self._q.put(text)

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_int(value: Any, field: str, default: int, warnings: list[str]) -> int:
    """Converts *value* to int (via float); logs a warning if necessary."""
    try:
        result = int(float(value))
        return result
    except (TypeError, ValueError):
        warnings.append(f"Field `{field}` invalid ({value!r}) — default used: {default}.")
        return default


def _coerce_int_list(value: Any, field: str, default: list[int], warnings: list[str]) -> list[int]:
    """Converts *value* to list[int]; accepts scalars and floats."""
    if not isinstance(value, list):
        warnings.append(f"Field `{field}`: scalar converted to list.")
        value = [value]
    try:
        result = sorted({int(float(v)) for v in value if float(v) > 0})
        if not result:
            warnings.append(f"Field `{field}` empty or invalid — default used: {default}.")
            return list(default)
        return result
    except (TypeError, ValueError) as exc:
        warnings.append(f"Field `{field}` invalid ({value!r}) — default used: {default}. ({exc})")
        return list(default)


def _coerce_str_list(value: Any, field: str, warnings: list[str]) -> list[str]:
    """Converts *value* to a normalised list[str] (strip + upper); accepts scalars."""
    if isinstance(value, str):
        warnings.append(f"Field `{field}`: scalar converted to list.")
        value = [value]
    if not isinstance(value, list):
        warnings.append(f"Field `{field}` ignored (unexpected type: {type(value).__name__}).")
        return []
    return [str(v).strip().upper() for v in value if str(v).strip()]


def _parse_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Extracts and coerces fields from a parsed TOML payload.

    Accepts two structures:
    - ``[simulation]`` wrapped section (standard format)
    - Keys at the root (flat format)

    Returns ``(state, warnings)``.
    """
    warnings: list[str] = []
    state: dict[str, Any] = copy.deepcopy(DEFAULT_STATE)

    # Resolve the simulation section
    simulation = payload.get("simulation", {})
    if not isinstance(simulation, dict):
        warnings.append("Section `[simulation]` has unexpected type — ignored.")
        simulation = {}

    # If no simulation keys found, try the root (flat TOML)
    KNOWN_KEYS = {
        "output_base_path",
        "seed",
        "generative_models",
        "rule_codes",
        "iterations",
        "candidates",
        "voters",
        "input_folder_path",
    }
    if not simulation and any(k in payload for k in KNOWN_KEYS):
        simulation = payload
        warnings.append("No `[simulation]` section found — reading keys from the root of the file.")

    # output_base_path
    if "output_base_path" in simulation:
        val = str(simulation["output_base_path"]).strip()
        state["output_base_path"] = val or DEFAULT_STATE["output_base_path"]

    # seed
    if "seed" in simulation:
        state["seed"] = _coerce_int(simulation["seed"], "seed", DEFAULT_STATE["seed"], warnings)

    # generative_models
    if "generative_models" in simulation:
        state["generative_models"] = _coerce_str_list(simulation["generative_models"], "generative_models", warnings)

    # rule_codes
    if "rule_codes" in simulation:
        state["rule_codes"] = _coerce_str_list(simulation["rule_codes"], "rule_codes", warnings)

    # candidates
    if "candidates" in simulation:
        state["candidates"] = _coerce_int_list(
            simulation["candidates"], "candidates", DEFAULT_STATE["candidates"], warnings
        )

    # voters
    if "voters" in simulation:
        state["voters"] = _coerce_int_list(simulation["voters"], "voters", DEFAULT_STATE["voters"], warnings)

    # iterations
    if "iterations" in simulation:
        val = _coerce_int(simulation["iterations"], "iterations", DEFAULT_STATE["iterations"], warnings)
        if val <= 0:
            warnings.append(
                f"Field `iterations` must be > 0 (received {val}) — default used: "
                f"{DEFAULT_STATE['iterations']}."
            )
        else:
            state["iterations"] = val

    # input_folder_path
    if "input_folder_path" in simulation:
        raw = simulation["input_folder_path"]
        state["input_folder_path"] = str(raw).strip() if raw else None

    # generator_params
    gen_params_section = payload.get("generator_params", {})
    if isinstance(gen_params_section, dict) and gen_params_section:
        gp: dict[str, dict[str, Any]] = {}
        for model, params in gen_params_section.items():
            key = str(model).strip().upper()
            if isinstance(params, dict):
                gp[key] = dict(params)
            else:
                warnings.append(
                    f"Section `generator_params.{model}` ignored (unexpected type: {type(params).__name__})."
                )
        state["generator_params"] = gp

    return state, warnings


# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_BASE_PATH = os.environ.get("VOTE_SIM_OUTPUT_BASE_PATH", "").strip()

DEFAULT_STATE: dict[str, Any] = {
    "output_base_path": _DEFAULT_OUTPUT_BASE_PATH,
    "seed": None,
    "generative_models": [],
    "rule_codes": [],
    "candidates": [],
    "voters": [],
    "iterations": 1000,
    "generator_params": {},
    "input_folder_path": None,
}

# ---------------------------------------------------------------------------
# State → TOML
# ---------------------------------------------------------------------------


def state_to_toml(state: dict[str, Any]) -> str:
    """Converts the current UI state to a valid TOML string.

    The produced document is directly readable by ``load_simulation_config()``.
    """
    sim: dict[str, Any] = {
        "generative_models": list(state.get("generative_models", [])),
        "rule_codes": list(state.get("rule_codes", [])),
        "iterations": int(state.get("iterations", 1000)),
    }
    output_path = state.get("output_base_path") or ""
    if output_path:
        sim["output_base_path"] = output_path
    seed = state.get("seed")
    if seed is not None:
        sim["seed"] = int(seed)
    doc: dict[str, Any] = {"simulation": sim}

    candidates = state.get("candidates")
    if candidates:
        doc["simulation"]["candidates"] = [int(c) for c in candidates]

    voters = state.get("voters")
    if voters:
        doc["simulation"]["voters"] = [int(v) for v in voters]

    input_folder = state.get("input_folder_path")
    if input_folder:
        doc["simulation"]["input_folder_path"] = str(input_folder)

    # Per-model generator params as sub-tables [generator_params.<MODEL>]
    generator_params = state.get("generator_params", {})
    if generator_params:
        doc["generator_params"] = {
            model.upper(): dict(params)
            for model, params in generator_params.items()
            if isinstance(params, dict) and params
        }

    return tomli_w.dumps(doc)


def write_temp_toml(state: dict[str, Any], base_dir: str | None = None) -> str:
    """Writes the current state to a temporary TOML file and returns its path.

    ``output_base_path`` is always converted to an absolute path before writing,
    so that ``load_simulation_config()`` does not interpret it relative to the
    /tmp/ folder of the temporary file.

    Args:
        state: Current UI state (session_state["cfg"]).
        base_dir: Base directory for resolving relative paths.
            If provided, a relative ``output_base_path`` is resolved from this
            directory (== the folder of the original TOML file).
            If ``None``, resolved from the current working directory.
    """
    resolved_state = dict(state)
    raw_path = resolved_state.get("output_base_path", "../data/")
    if base_dir is not None and not Path(raw_path).is_absolute():
        resolved_state["output_base_path"] = str((Path(base_dir) / raw_path).resolve())
    else:
        resolved_state["output_base_path"] = str(Path(raw_path).resolve())

    content = state_to_toml(resolved_state)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".toml",
        prefix="vote_sim_",
        delete=False,
        encoding="utf-8",
    ) as fh:
        fh.write(content)
        return fh.name


# ---------------------------------------------------------------------------
# TOML → State
# ---------------------------------------------------------------------------


def toml_to_state(toml_path: str | Path) -> dict[str, Any]:
    """Parses a TOML file and returns a complete UI state.

    Keys absent from the file are filled from ``DEFAULT_STATE``.
    Compatible with ``[simulation]``-section TOMLs and flat TOMLs.
    """
    path = Path(toml_path)
    if not path.is_file():
        raise FileNotFoundError(f"TOML file not found: {path}")

    raw = path.read_bytes()
    # Strip UTF-8 BOM if present
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    try:
        payload = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML file ({path.name}): {exc}") from exc

    state, _warnings = _parse_payload(payload)
    return state


def toml_bytes_to_state(raw_bytes: bytes) -> tuple[dict[str, Any], list[str]]:
    """Parses TOML bytes (Streamlit upload) and returns ``(state, warnings)``.

    Handles:
    - UTF-8 with or without BOM
    - latin-1 encoding as fallback
    - Wrapped ``[simulation]`` structure or flat TOML
    - Incorrect types (float → int, scalar → list, etc.)
    """
    warnings: list[str] = []

    # UTF-8 BOM
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raw_bytes = raw_bytes[3:]
        warnings.append("UTF-8 BOM detected and removed.")

    # Decoding
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw_bytes.decode("latin-1")
            warnings.append("Non-UTF-8 encoding detected — reading as latin-1.")
        except Exception as exc:
            raise ValueError("Cannot decode file (UTF-8 and latin-1 both failed).") from exc

    # Parse TOML
    try:
        payload = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Syntactically invalid TOML file: {exc}") from exc

    state, parse_warnings = _parse_payload(payload)
    return state, warnings + parse_warnings
