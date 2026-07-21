"""Package setup for AL-ST pipeline."""

from setuptools import setup, find_packages

setup(
    name="al-st-prediction",
    version="0.1.0",
    description="Active Learning for Spatial Transcriptomics Gene-Expression Prediction",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "scikit-learn>=1.0",
        "pandas>=1.3",
        "torch>=2.0",
        "torchvision>=0.15",
        "scanpy>=1.9",
        "anndata>=0.8",
        "Pillow>=9.0",
        "matplotlib>=3.5",
        "tqdm>=4.60",
    ],
    extras_require={
        "hest": ["hest", "huggingface_hub"],
        "openslide": ["openslide-python"],
        "pathology_fm": ["timm", "open_clip_torch"],
        "dinov2": ["timm"],
        "all": [
            "hest",
            "huggingface_hub",
            "openslide-python",
            "timm",
            "open_clip_torch",
            "tifffile",
            "squidpy",
        ],
    },
    entry_points={
        "console_scripts": [
            "al-st=src.run_experiment:main",
        ],
    },
)
