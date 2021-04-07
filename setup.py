from setuptools import setup
from pathlib import Path


ROOT = Path(__file__).parent
SRC_MODULE = ROOT / "src/sopredictable.py"


install_requires = [
    "typing-extensions;python_version < '3.8'",
]

extras_require = {"dev": ["pytest"]}


def setup_pkg():

    with SRC_MODULE.open("r") as f:
        mod_globals = {}
        exec(f.read(), mod_globals)

    version = mod_globals["__version__"]
    del mod_globals

    setup(
        name="sopredictable",
        version=version,
        author="Adriaan Lemmens (lemontheme)",
        author_email="lemontheme@gmail.com",
        url="https://github.com/lemontheme/sopredictable",
        package_dir={"": "src"},
        py_modules=["sopredictable"],
        install_requires=install_requires,
        extras_require=extras_require,
        python_requires=">=3.7",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
    )


if __name__ == "__main__":
    setup_pkg()
