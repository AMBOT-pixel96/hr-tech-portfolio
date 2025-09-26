import os
from pathlib import Path

def generate_tree(start_path=".", prefix="", max_items=3):
    tree_str = ""
    items = sorted([i for i in os.listdir(start_path) if not i.startswith(".git")])
    pointers = ["├── "] * (len(items) - 1) + ["└── "]

    # Show only first `max_items`, then collapse rest
    display_items = items[:max_items]
    hidden_count = len(items) - max_items if len(items) > max_items else 0

    for pointer, item in zip(pointers[:len(display_items)], display_items):
        path = os.path.join(start_path, item)
        tree_str += prefix + pointer + item + "\n"
        if os.path.isdir(path):
            extension = "│   " if pointer == "├── " else "    "
            tree_str += generate_tree(path, prefix + extension, max_items=max_items)

    # Add ellipsis line if there are hidden items
    if hidden_count > 0:
        tree_str += prefix + "└── ... ({} more)\n".format(hidden_count)

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
    print("✅ Repo tree (summarized) injected into README.md")
