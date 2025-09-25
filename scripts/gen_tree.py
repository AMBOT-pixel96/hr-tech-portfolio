import os
from pathlib import Path

def generate_tree(start_path: str = ".", prefix: str = "") -> str:
    """Generate a folder tree like `tree` command."""
    tree_str = ""
    entries = sorted(Path(start_path).iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    for index, entry in enumerate(entries):
        connector = "└── " if index == len(entries) - 1 else "├── "
        tree_str += prefix + connector + entry.name + ("\n" if entry.is_file() else "/\n")
        if entry.is_dir() and entry.name not in {".git", "__pycache__", ".ipynb_checkpoints"}:
            extension = "    " if index == len(entries) - 1 else "│   "
            tree_str += generate_tree(str(entry), prefix + extension)
    return tree_str

if __name__ == "__main__":
    tree = "```plaintext\n" + "hr-tech-portfolio/\n" + generate_tree(".") + "```"
    with open("FOLDER_STRUCTURE.md", "w") as f:
        f.write(tree)
    print("✅ Folder structure saved to FOLDER_STRUCTURE.md")