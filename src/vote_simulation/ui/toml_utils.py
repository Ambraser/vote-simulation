"""Bijection entre l'état courant de l'UI (session_state) et un fichier TOML valide.

L'état UI est un dict plat avec les clés suivantes :

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
    """Redirige sys.stdout vers une queue — utilisé par les threads de génération/simulation."""

    def __init__(self, q: "queue.Queue[str]") -> None:
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
    """Convertit *value* en int (via float) ; journalise un warning si nécessaire."""
    try:
        result = int(float(value))
        return result
    except (TypeError, ValueError):
        warnings.append(
            f"Champ `{field}` invalide ({value!r}) — valeur par défaut utilisée : {default}."
        )
        return default


def _coerce_int_list(value: Any, field: str, default: list[int], warnings: list[str]) -> list[int]:
    """Convertit *value* en liste[int] ; accepte les scalaires et les floats."""
    if not isinstance(value, list):
        warnings.append(f"Champ `{field}` : scalaire converti en liste.")
        value = [value]
    try:
        result = sorted({int(float(v)) for v in value if float(v) > 0})
        if not result:
            warnings.append(f"Champ `{field}` vide ou invalide — valeur par défaut utilisée : {default}.")
            return list(default)
        return result
    except (TypeError, ValueError) as exc:
        warnings.append(f"Champ `{field}` invalide ({value!r}) — valeur par défaut utilisée : {default}. ({exc})")
        return list(default)


def _coerce_str_list(value: Any, field: str, warnings: list[str]) -> list[str]:
    """Convertit *value* en liste[str] normalisée (strip + upper) ; accepte les scalaires."""
    if isinstance(value, str):
        warnings.append(f"Champ `{field}` : scalaire converti en liste.")
        value = [value]
    if not isinstance(value, list):
        warnings.append(f"Champ `{field}` ignoré (type inattendu : {type(value).__name__}).")
        return []
    return [str(v).strip().upper() for v in value if str(v).strip()]


def _parse_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Extrait et coerce les champs depuis un payload TOML parsé.

    Accepte deux structures :
    - ``[simulation]`` section wrappée (format standard)
    - Clés à la racine (format plat)

    Retourne ``(state, warnings)``.
    """
    warnings: list[str] = []
    state: dict[str, Any] = copy.deepcopy(DEFAULT_STATE)

    # Résoudre la section simulation
    simulation = payload.get("simulation", {})
    if not isinstance(simulation, dict):
        warnings.append("Section `[simulation]` de type inattendu — ignorée.")
        simulation = {}

    # Si aucune clé simulation trouvée, essayer la racine (TOML plat)
    KNOWN_KEYS = {"output_base_path", "seed", "generative_models", "rule_codes",
                  "iterations", "candidates", "voters", "input_folder_path"}
    if not simulation and any(k in payload for k in KNOWN_KEYS):
        simulation = payload
        warnings.append(
            "Aucune section `[simulation]` trouvée — lecture des clés à la racine du fichier."
        )

    # output_base_path
    if "output_base_path" in simulation:
        val = str(simulation["output_base_path"]).strip()
        state["output_base_path"] = val or DEFAULT_STATE["output_base_path"]

    # seed
    if "seed" in simulation:
        state["seed"] = _coerce_int(simulation["seed"], "seed", DEFAULT_STATE["seed"], warnings)

    # generative_models
    if "generative_models" in simulation:
        state["generative_models"] = _coerce_str_list(
            simulation["generative_models"], "generative_models", warnings
        )

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
        state["voters"] = _coerce_int_list(
            simulation["voters"], "voters", DEFAULT_STATE["voters"], warnings
        )

    # iterations
    if "iterations" in simulation:
        val = _coerce_int(simulation["iterations"], "iterations", DEFAULT_STATE["iterations"], warnings)
        if val <= 0:
            warnings.append(
                f"Champ `iterations` doit être > 0 (reçu {val}) — valeur par défaut utilisée : "
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
                    f"Section `generator_params.{model}` ignorée (type inattendu : {type(params).__name__})."
                )
        state["generator_params"] = gp

    return state, warnings

# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------

DEFAULT_STATE: dict[str, Any] = {
    "output_base_path": "../data/",
    "seed": 42,
    "generative_models": [],
    "rule_codes": [],
    "candidates": [3, 14],
    "voters": [11, 101, 1001],
    "iterations": 1000,
    "generator_params": {},
    "input_folder_path": None,
}

# ---------------------------------------------------------------------------
# State → TOML
# ---------------------------------------------------------------------------


def state_to_toml(state: dict[str, Any]) -> str:
    """Convertit l'état courant de l'UI en contenu TOML valide (str).

    Le document produit est directement lisible par ``load_simulation_config()``.
    """
    doc: dict[str, Any] = {
        "simulation": {
            "output_base_path": state.get("output_base_path", "../data/"),
            "seed": int(state.get("seed", 42)),
            "generative_models": list(state.get("generative_models", [])),
            "rule_codes": list(state.get("rule_codes", [])),
            "iterations": int(state.get("iterations", 1000)),
        }
    }

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
    """Écrit l'état courant dans un fichier TOML temporaire et retourne son chemin.

    ``output_base_path`` est systématiquement converti en chemin absolu avant
    l'écriture, de sorte que ``load_simulation_config()`` ne le réinterprète
    pas relativement au dossier /tmp/ du fichier temporaire.

    Args:
        state: État courant de l'UI (session_state["cfg"]).
        base_dir: Répertoire de base pour résoudre les chemins relatifs.
            Si fourni, ``output_base_path`` relatif est résolu depuis ce
            répertoire (== le dossier du fichier TOML d'origine).
            Si ``None``, résolution depuis le dossier de travail courant.
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
    """Parse un fichier TOML et retourne un état UI complet.

    Les clés absentes du fichier sont remplies par ``DEFAULT_STATE``.
    Compatible avec les TOML à section ``[simulation]`` et les TOML plats.
    """
    path = Path(toml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier TOML introuvable : {path}")

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
        raise ValueError(f"Fichier TOML invalide ({path.name}) : {exc}") from exc

    state, _warnings = _parse_payload(payload)
    return state


def toml_bytes_to_state(raw_bytes: bytes) -> tuple[dict[str, Any], list[str]]:
    """Parse des bytes TOML (upload Streamlit) et retourne ``(state, warnings)``.

    Gère :
    - UTF-8 avec ou sans BOM
    - Encodage latin-1 en fallback
    - Structure ``[simulation]`` wrappée ou TOML plat
    - Types incorrects (float → int, scalaire → liste, etc.)
    """
    warnings: list[str] = []

    # UTF-8 BOM
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raw_bytes = raw_bytes[3:]
        warnings.append("BOM UTF-8 détecté et supprimé.")

    # Décodage
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw_bytes.decode("latin-1")
            warnings.append("Encodage non-UTF-8 détecté — lecture en latin-1.")
        except Exception as exc:
            raise ValueError("Impossible de décoder le fichier (UTF-8 et latin-1 échoués).") from exc

    # Parse TOML
    try:
        payload = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Fichier TOML syntaxiquement invalide : {exc}") from exc

    state, parse_warnings = _parse_payload(payload)
    return state, warnings + parse_warnings
