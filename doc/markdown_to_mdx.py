import re
import argparse
from pathlib import Path

def convert_to_mdx(content):
    """
    Convert Sphinx-generated markdown to MDX-compatible format.
    """
    # Replace type annotations in class definitions
    content = re.sub(
        r'(class[^(]+\()([^)]+)(\))',
        lambda m: m.group(1) + re.sub(r':\s*([^,\)]+)', r' /* : \1 */', m.group(2)) + m.group(3),
        content
    )
    
    # Replace type annotations in class attributes
    content = re.sub(
        r'(\n\s*\w+)\s*:\s*([^\n=]+)(\s*=?\s*.*\n)',
        r'\1 /* : \2 */\3',
        content
    )
    
    # Remove trailing asterisks that might conflict with JSX
    content = re.sub(r'\*\s*$', '', content, flags=re.MULTILINE)
    
    # Fix code block references (optional, depends on your needs)
    content = re.sub(r'\[`([^`]+)`\]', r'[`\1`]', content)
    
    return content

def process_file(input_path, output_path=None):
    """
    Process a single markdown file and convert it to MDX.
    """
    if output_path is None:
        output_path = input_path
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    converted_content = convert_to_mdx(content)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(converted_content)

def main():
    parser = argparse.ArgumentParser(description='Convert Sphinx markdown to MDX')
    parser.add_argument('input', help='Input file or directory')
    parser.add_argument('--output', help='Output file or directory (optional)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    
    if input_path.is_file():
        process_file(input_path, output_path)
    elif input_path.is_dir():
        if output_path and not output_path.exists():
            output_path.mkdir(parents=True)
        
        for md_file in input_path.glob('**/*.md'):
            relative_path = md_file.relative_to(input_path)
            out_file = output_path / relative_path if output_path else md_file
            out_file.parent.mkdir(parents=True, exist_ok=True)
            process_file(md_file, out_file)

if __name__ == '__main__':
    main()
