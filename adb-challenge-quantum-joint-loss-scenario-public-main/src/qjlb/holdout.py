"""Generate private, reproducible numerical-estimation cases."""

import hashlib
import json
from pathlib import Path
import random
import secrets
import shutil

DATA_FILES = ("adb_terms.json", "audit.json", "seismic.npz", "flood.npz")
COUNTRIES = ["KAZ", "KGZ", "TJK", "TKM", "UZB"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(output, data, count=6, seed=None):
    if type(count) is not int or count < 2 or count > 100:
        raise ValueError("cases must be from 2 to 100")
    if seed is None:
        seed = secrets.randbits(128)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    data = Path(data).resolve()
    for name in DATA_FILES:
        if not (data / name).is_file():
            raise ValueError(f"missing data file: {name}")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "data").mkdir()
    (output / "cases").mkdir()
    for name in DATA_FILES:
        shutil.copyfile(data / name, output / "data" / name)
    rng = random.Random(seed)
    cases = []
    for i in range(count):
        tier = (i // 2) % 3
        if i % 2 == 0:
            principal = [rng.randint(40, 120) * 1e6 for _ in range(2)]
            eq = [round(p * rng.uniform(0.55, 0.9), 2) for p in principal]
            config = {
                "schema_version": 1,
                "track": "adb_drb",
                "terms_file": "../data/adb_terms.json",
                "countries": ["KGZ", "TJK"],
                "term_years": [1, 3, 10][tier],
                "earthquake_sublimit_usd": eq,
                "precipitation_sublimit_usd": [p - e for p, e in zip(principal, eq)],
                "earthquake_common_rate_multiplier": rng.uniform(0, 2.5),
                "precipitation_annual_trigger_probability": [
                    rng.uniform(0.005, 0.04) for _ in range(2)
                ],
                "precipitation_common_rate_share": rng.uniform(0, 0.5),
                "precipitation_rate_multiplier": rng.uniform(0.5, 2),
                "portfolio_threshold_usd": round(
                    sum(principal) * rng.uniform(0.3, 0.8), 2
                ),
                "assumptions": "Synthetic contract and sensitivity case using the public earthquake calibration. Independent perils and years; independent severity marks within shared earthquakes.",
            }
        else:
            countries = rng.sample(COUNTRIES, [2, 3, 5][tier])
            n = len(countries)
            attachment = [
                [round(rng.uniform(5e7, 4e8), 2) for _ in range(n)] for _ in range(2)
            ]
            limits = [rng.randint(40, 120) * 1e6 for _ in range(n)]
            config = {
                "schema_version": 2,
                "track": "worldbank_elt",
                "dataset_directory": "../data",
                "countries": countries,
                "peril_order": ["seismic", "flood"],
                "attachment_usd": attachment,
                "exhaustion_usd": [
                    [round(a + rng.uniform(2e8, 9e8), 2) for a in row]
                    for row in attachment
                ],
                "limit_usd": limits,
                "portfolio_threshold_usd": round(
                    sum(limits) * rng.uniform(0.3, 0.8), 2
                ),
                "contract_note": "Synthetic payments per catalogue row without a term cap. Joint exceedance requires a positive payment in every selected country.",
            }
        name = f"cases/case-{i + 1:03d}.json"
        (output / name).write_text(json.dumps(config, indent=2) + "\n")
        cases.append(name)
    manifest = {
        "schema_version": 1,
        "generator_version": "holdout-v1",
        "seed": seed,
        "cases": cases,
        "files": {
            str(p.relative_to(output)): digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    (output / "suite.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_suite(path):
    from .validation import read_json

    path = Path(path).resolve()
    manifest = read_json(path / "suite.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("generator_version") != "holdout-v1"
    ):
        raise ValueError("unsupported suite version")
    cases = manifest.get("cases")
    if (
        not isinstance(cases, list)
        or not cases
        or any(not isinstance(case, str) for case in cases)
        or len(cases) != len(set(cases))
    ):
        raise ValueError("suite must contain distinct cases")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("suite file hashes are required")
    for name, expected in files.items():
        file = path / name
        if (
            file.is_symlink()
            or not file.resolve().is_relative_to(path)
            or digest(file) != expected
        ):
            raise ValueError(f"suite file changed or outside suite: {name}")
    required = set(cases) | {f"data/{name}" for name in DATA_FILES}
    if not required <= files.keys() or any(
        not name.startswith("cases/") for name in cases
    ):
        raise ValueError("suite is missing required inputs")
    return manifest
