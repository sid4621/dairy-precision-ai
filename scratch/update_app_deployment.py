import os

filepath = "dashboard/src/App.jsx"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Add the API_BASE definition at the top of the App component (around line 22)
if "const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';" not in content:
    content = content.replace(
        "function App() {", 
        "function App() {\n  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';"
    )

# Replace fetch calls
content = content.replace("'http://localhost:8000/api", "`${API_BASE}/api")
content = content.replace("'http://localhost:8000", "API_BASE")
# Also fix any backticks if they were already template literals (wait, none of them were template literals)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("App.jsx updated for deployment.")
