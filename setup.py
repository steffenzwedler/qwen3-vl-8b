"""Setup script for qwen3-vl-8b package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="qwen3-vl-8b",
    version="0.1.0",
    description="Windows Screen Capture and Analysis using Qwen3-VL-8B",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/qwen3-vl-8b",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.37.0",
        "qwen-vl-utils>=0.0.2",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "mss>=9.0.0",
        "pywin32>=306",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "flash-attn": [
            "flash-attn>=2.5.0",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "vlm-screen=src.interactive_vlm:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Testing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Microsoft :: Windows",
    ],
)
