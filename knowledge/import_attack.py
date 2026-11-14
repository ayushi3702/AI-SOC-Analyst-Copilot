"""Import official MITRE ATT&CK Enterprise STIX techniques into local JSON."""

import argparse
import json
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
    "master/enterprise-attack/enterprise-attack.json"
)
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "mitre_attack_techniques.json"
MAX_BUNDLE_BYTES = 100 * 1024 * 1024
MAX_TIMEOUT_SECONDS = 120


def _mitre_reference(item: dict) -> dict | None:
    if item.get("type") != "attack-pattern" or item.get("revoked") or item.get("x_mitre_deprecated"):
        return None
    external = next(
        (
            reference
            for reference in item.get("external_references", [])
            if reference.get("source_name") == "mitre-attack"
            and isinstance(reference.get("external_id"), str)
            and reference["external_id"].startswith("T")
        ),
        None,
    )
    if external is None:
        return None
    technique_id = external["external_id"]
    title = item.get("name")
    description = item.get("description")
    if not isinstance(title, str) or not isinstance(description, str):
        return None
    phases = sorted({
        phase.get("phase_name", "").replace("-", " ").title()
        for phase in item.get("kill_chain_phases", [])
        if phase.get("kill_chain_name") == "mitre-attack"
        and phase.get("phase_name")
    })
    platforms = sorted(set(item.get("x_mitre_platforms", [])))
    tactic_line = f"Tactics: {', '.join(phases)}." if phases else ""
    platform_line = f"Platforms: {', '.join(platforms)}." if platforms else ""
    source_url = external.get("url") or (
        "https://attack.mitre.org/techniques/"
        + technique_id.replace(".", "/")
        + "/"
    )
    return {
        "reference_id": f"mitre:{technique_id}",
        "title": title,
        "kind": "MITRE ATT&CK Enterprise technique",
        "citation": f"MITRE ATT&CK Enterprise, {technique_id}",
        "source_url": source_url,
        "text": " ".join(
            part for part in (
                technique_id,
                tactic_line,
                platform_line,
                description,
                "Reference context only; not evidence that this technique occurred.",
            ) if part
        ),
    }


def extract_enterprise_techniques(bundle: dict) -> list[dict]:
    if not isinstance(bundle, dict) or bundle.get("type") != "bundle":
        raise ValueError("Input must be a STIX bundle object")
    objects = bundle.get("objects")
    if not isinstance(objects, list):
        raise ValueError("STIX bundle is missing its objects list")
    techniques = {}
    for item in objects:
        if not isinstance(item, dict):
            continue
        reference = _mitre_reference(item)
        if reference is not None:
            techniques[reference["reference_id"]] = reference
    if not techniques:
        raise ValueError("No active MITRE ATT&CK Enterprise techniques found")
    return [techniques[key] for key in sorted(techniques)]


def _read_bundle(input_path: Path | None, timeout: int) -> dict:
    if input_path is not None:
        return json.loads(input_path.read_text(encoding="utf-8"))
    request = Request(
        SOURCE_URL,
        headers={"User-Agent": "AI-SOC-Analyst-Copilot/0.1"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = response.read(MAX_BUNDLE_BYTES + 1)
    if len(payload) > MAX_BUNDLE_BYTES:
        raise ValueError(
            f"MITRE ATT&CK bundle exceeds {MAX_BUNDLE_BYTES} byte download limit"
        )
    return json.loads(payload.decode("utf-8"))


def import_attack_corpus(
    output_path: Path = DEFAULT_OUTPUT,
    input_path: Path | None = None,
    timeout: int = 60,
) -> int:
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout must be an integer from 1 to {MAX_TIMEOUT_SECONDS}")
    output_path = Path(output_path)
    techniques = extract_enterprise_techniques(_read_bundle(input_path, timeout))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(techniques, temporary, ensure_ascii=False, indent=2)
        temporary.write("\n")
    temporary_path.replace(output_path)
    return len(techniques)


def main():
    parser = argparse.ArgumentParser(
        description="Build a local corpus from the official ATT&CK Enterprise STIX bundle."
    )
    parser.add_argument("--input", type=Path, help="Local STIX bundle JSON; otherwise fetch from MITRE's public repository.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    count = import_attack_corpus(args.output, args.input, args.timeout)
    print(f"Imported {count:,} ATT&CK techniques into {args.output}")


if __name__ == "__main__":
    main()
