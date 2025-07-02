#!/usr/bin/env python3
"""Build documentation using Sphinx."""

import subprocess
import sys
from pathlib import Path


def main():
    """Build documentation."""
    docs_dir = Path(__file__).parent
    build_dir = docs_dir / "_build"
    
    # Clean previous build
    if build_dir.exists():
        import shutil
        shutil.rmtree(build_dir)
    
    # Build documentation
    cmd = [
        sys.executable, "-m", "sphinx",
        "-b", "html",
        "-d", str(build_dir / "doctrees"),
        str(docs_dir),
        str(build_dir / "html")
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error building documentation:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    
    print("Documentation built successfully!")
    print(f"Output: {build_dir / 'html' / 'index.html'}")


if __name__ == "__main__":
    main()