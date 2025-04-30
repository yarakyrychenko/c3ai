from setuptools import setup, find_packages

setup(
    name="c3ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",           
        "pandas",
        "openai==1.55.3",
        "torch",
        "transformers", 
        "accelerate",
        "datasets",
        "peft",
        "trl",

    ],
    author="Yara Kyrychenko",
    description="C3AI: A package for Crafting and Evaluating Constitutions for Constitusional AI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yarakyrychenko/c3ai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    include_package_data=True,  
    package_data={"c3ai": [
        "data/three_shots.jsonl", 
        "data/principles.csv", 
        "data/harmless_one_turn_train_100_sample.jsonl",
        "data/Rscripts/EGA.R",
        "data/Rscripts/bootEGA.R",
        ]}, 
)
