import pathlib
import inspect

print(f"Module 'a.py' loaded from {__name__}")

def f(v):
    print(v.__module__)
    print(inspect.getmodule(v))


def g(fpath):
    print(fpath.resolve())


def g_(fpath: str):
    print(pathlib.Path(fpath).resolve())
