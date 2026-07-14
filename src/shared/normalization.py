def is_empty_reference(value: str) -> bool:
    """Checks if a page reference string is effectively empty or 'N/A'."""
    val = str(value).strip().upper()
    return not val or val in {"N/A", "NA", "NONE"}

