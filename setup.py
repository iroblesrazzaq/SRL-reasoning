from setuptools import setup, find_packages

setup(
    name="srl-reasoning",
    version="0.1.0",
    description="Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning",
    author="",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "accelerate",
        "vllm",
        "datasets",
        "pandas",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

