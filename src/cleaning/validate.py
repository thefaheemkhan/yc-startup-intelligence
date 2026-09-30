"""Structural validation of the raw payload. Errors stop the pipeline; warnings are reported."""
from __future__ import annotations

from collections import Counter

REQUIRED_FIELDS = {
    "id", "name", "slug", "batch", "status", "industry", "subindustry", "all_locations",
    "team_size", "launched_at", "tags", "regions", "stage", "one_liner", "long_description",
    "website", "url", "isHiring", "nonprofit", "top_company",
}
KNOWN_STATUSES = {"Active", "Inactive", "Acquired", "Public"}


def validate_raw(records: list[dict]) -> dict:
    """Return {'ok', 'errors', 'warnings', 'n_records'}."""
    errors: list[str] = []
    warnings: list[str] = []
    if not records:
        return {"ok": False, "errors": ["no records"], "warnings": [], "n_records": 0}

    field_counts = Counter(k for r in records for k in r)
    for f in sorted(REQUIRED_FIELDS):
        missing = len(records) - field_counts.get(f, 0)
        if missing == len(records):
            errors.append(f"required field absent from every record: {f}")
        elif missing:
            warnings.append(f"field '{f}' absent from {missing} records")

    ids = [r.get("id") for r in records]
    if len(set(ids)) != len(ids):
        warnings.append(f"{len(ids) - len(set(ids))} duplicate ids (cleaning keeps the last)")
    bad_status = Counter(r.get("status") for r in records if r.get("status") not in KNOWN_STATUSES)
    if bad_status:
        warnings.append(f"unexpected status values: {dict(bad_status)}")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "n_records": len(records)}
