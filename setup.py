from setuptools import setup, find_packages

setup(
    name="dl-recsys-financial",
    version="1.0.0",
    description="Deep Learning Recommendation System for Financial Services",
    author="Jay Guwalani",
    author_email="jayguwalani@example.com",
    url="https://github.com/JayDS22/Deep-Learning-Recommendation-System-for-Financial-Services",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
    entry_points={
        "console_scripts": [
            "dl-recsys-train=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
