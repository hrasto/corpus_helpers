import re

# --- file naming utils --- 

def to_camel_case(s: str) -> str:
    """Convert string (with spaces, underscores, hyphens) to camelCase."""
    parts = re.split(r'[\s_-]+', s.strip())
    if not parts:
        return ""
    return parts[0].lower() + "".join(word.capitalize() for word in parts[1:])

def safe_filename(name: str) -> str:
    """Replace unsafe filename characters with underscores."""
    return re.sub(r'[^A-Za-z0-9._-]', '_', name)

def dict_to_filename(data: dict) -> str:
    """Embed dict into a safe filename with camelCase keys and string values camelCased."""
    parts = []
    for k, v in data.items():
        key = to_camel_case(str(k))
        if isinstance(v, str):
            value = to_camel_case(v)
        else:
            value = str(v)
        parts.append(f"{key}-{value}")
    filename = "_".join(parts)
    return safe_filename(filename)
