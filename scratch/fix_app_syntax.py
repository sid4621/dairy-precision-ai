import os

filepath = "dashboard/src/App.jsx"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Fix the API_BASE declaration
content = content.replace(
    "const API_BASE = import.meta.env.VITE_API_URL || API_BASE';",
    "const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';"
)

# Fix fetch strings. Currently they look like: fetch(`${API_BASE}/api/infer_mastitis_sensor', {
# Let's replace ' with ` where appropriate.
# Actually, the string starts with ` but ends with ' because it was originally 'http://...'.
# So we just need to replace `${API_BASE}/api/SOMETHING' with `${API_BASE}/api/SOMETHING`
import re
# Regex to find `${API_BASE}/api/something' and replace the trailing ' with `
content = re.sub(r"(\$\{API_BASE\}/api/[a-zA-Z0-Z_]+)'", r"\1`", content)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("App.jsx syntax fixed.")
