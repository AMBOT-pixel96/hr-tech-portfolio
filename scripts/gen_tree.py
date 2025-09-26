import os
from pathlib import Path

# Max number of files to show inside each folder
MAX_FILES = 5  

def generate_tree(start_path=".", prefix=""):
    tree_str = ""
    items = sorted(os.listdir(start_path))
    # Skip hidden files and .git internals
    items = [i for i in items if not i.startswith(".git")]
    
    pointers = ["├── "] * (len(items) - 1) + ["└── "]
    for pointer, item in zip(pointers, items):
        path = os.path.join(start_path, item)

        # Always show folders
        if os.path.isdir(path):
            tree_str += prefix + pointer + item + "/\n"
            extension = "│   " if pointer == "├── " else "    "
            tree_str += generate_tree(path, prefix + extension)
        else:
            # For files → only show top 3–5 latest
            files = sorted(items, key=lambda x: os.path.getmtime(os.path.join(start_path, x)), reverse=True)
            files = [f for f in files if os.path.isfile(os.path.join(start_path, f))]
            for f in files[:MAX_FILES]:
                tree_str += prefix + pointer + f + "\n"
            break  # prevent flooding
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
    print("✅ Curated repo tree injected into README.md")
