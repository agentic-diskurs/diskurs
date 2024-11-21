import re
from pathlib import Path

# [Previous configuration code remains the same...]


def convert_to_mdx(content):
    """
    Convert Sphinx-generated markdown to MDX-compatible format by:
    1. Simplifying enum representations
    2. Removing default values that contain angle brackets
    """
    # Replace enum value representations
    content = re.sub(r'<(\w+)\.(\w+):\s*[\'"][^\'"]+[\'"]>', r"\1.\2", content)

    # Remove default values that contain angle brackets
    content = re.sub(r"\s*=\s*<[^>]+>", "", content)

    return content


def process_markdown_files(app, exception):
    """
    Process all markdown files in the build directory after Sphinx build is complete.
    """
    if exception is not None:  # Skip if build failed
        return

    build_dir = Path(app.outdir)

    for md_file in build_dir.glob("**/*.md"):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            converted_content = convert_to_mdx(content)

            with open(md_file, "w", encoding="utf-8") as f:
                f.write(converted_content)

            print(f"Processed {md_file}")
        except Exception as e:
            print(f"Error processing {md_file}: {str(e)}")
