import os
from pathlib import Path

def generate_tree(start_path=".", prefix=""):
    tree_str = ""
    items = sorted(os.listdir(start_path))
    pointers = ["├── "] * (len(items) - 1) + ["└── "]
    for pointer, item in zip(pointers, items):
        if item.startswith(".git"):  # skip git internals
            continue
        path = os.path.join(start_path, item)
        tree_str += prefix + pointer + item + "\n"
        if os.path.isdir(path):
            extension = "│   " if pointer == "├── " else "    "
            tree_str += generate_tree(path, prefix + extension)
    return tree_str

def update_readme(tree_output):
    readme_path = Path("README.md")
    content = readme_path.read_text(encoding="utf-8")

    start_tag = "<!-- REPO_TREE_START -->"
    end_tag = "<!-- REPO_TREE_END -->"

    new_block = f"{start_tag}\n```text\n{tree_output}```\n{end_tag}"

    if start_tag in content and end_tag in content:
        # Replace existing block
        start = content.index(start_tag)
        end = content.index(end_tag) + len(end_tag)
        content = content[:start] + new_block + content[end:]
    else:
        # Add block at the bottom
        content += "\n\n## 📂 Repository Structure\n" + new_block

    readme_path.write_text(content, encoding="utf-8")

if __name__ == "__main__":
    tree = generate_tree(".")
    update_readme(tree)
    print("✅ Repo tree injected into README.md")