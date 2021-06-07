from __future__ import annotations

import collections
import importlib
import inspect
import copy
import dataclasses
from dataclasses import dataclass, field
from json import encoder
import typing
import zipfile
import sys
import json
import shutil
from pathlib import Path

# import pickle
import dill as pickle
# import cloudpickle as pickle

pickle.settings["recurse"] = True


from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
    cast,
)

if sys.version_info >= (3, 8):
    from typing import Protocol, TypedDict
else:
    from typing_extensions import Protocol, TypedDict

from sopredictable.utils import iterative_topological_sort


PREDICTOR_CLS_FILE = "predictor_cls.pkl"
META_FILE = "meta.json"
# CODE_ARCHIVE = "code.zip"
CONDA_FILE = "conda.yml"
ARTIFACTS_DIR = "artifacts"
ARTIFACTS_LOAD_ORDER_FILE = "load_order.json"
ENV_DIR = "env"
ENV_ID_FILE = "env_id.txt"
PY_DISTS_DIR = "dists"
CODE_DIR = "code"
ARTIFACT_DATA_DIR = "data"
ARTIFACT_CLS_FILE = "cls.pkl"
ARTIFACT_DEPS_FILE = "deps.json"
ARTIFACT_SAVE_FN_FILE = "save_fn.pkl"
ARTIFACT_LOAD_FN_FILE = "load_fn.pkl"
DEFAULT_ARTIFACT_FILE = "artifact.pkl"
DOCKERFILE_FILE = "Dockerfile"
REQUIREMENTS_TXT_FILE = "requirements.txt"
PICKLING_PROTOCOL = 4  # compatible with py37

PathLike = typing.Union[str, Path]
ArtifactName = str

_JSONValue = Union[
    bool, None, int, float, str, Dict[str, Any], List[Any]
]

JSONDict = Dict[str, _JSONValue]


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


@runtime_checkable
class Predictable(Protocol):
    def predict(self, batch: Sequence[Any]) -> Sequence[Any]:
        ...

    @classmethod
    def from_artifacts_dict(cls: Type[T], artifacts_dict: Dict[str, Any]) -> T:
        ...


@runtime_checkable
class ArtifactSaver(Protocol[T_contra]):
    def __call__(self, artifact: T_contra, data_dir: Path) -> None:
        ...


@runtime_checkable
class ArtifactLoader(Protocol[T_co]):
    def __call__(self, data_dir: Path, deps: Dict[str, Any]) -> T_co:
        ...


# Implements ArtifactSaver protocol
def save_as_pickle(artifact: Any, data_dir: Path) -> None:
    try:
        (data_dir / DEFAULT_ARTIFACT_FILE).write_bytes(
            pickle.dumps(artifact, protocol=PICKLING_PROTOCOL)
        )
    except pickle.PicklingError:
        raise ValueError(f"Unable to pickle {artifact}")


# Implements ArtifactLoader protocol
def load_from_pickle(data_dir: Path, deps: Dict[str, Any]) -> Any:
    print(f"loading with deps={deps}")
    _ = deps
    pickle_fpath = data_dir / DEFAULT_ARTIFACT_FILE
    try:
        return pickle.loads(pickle_fpath.read_bytes())
    except pickle.UnpicklingError:
        raise ValueError(f"Unable to load pickle from {pickle_fpath}")


@dataclass(frozen=True)
class _Rename:
    original: str
    new: str


def rename(
    original: str,
    new: str,
) -> _Rename:
    """
    Args:
        original: original name, e.g. of an artifact registered via
          `Saver.add_artifact()`.
        new: new name, e.g. of artifact expected on a given key in dict passed to `deps`
         parameter of functions implementing `ArtifactLoader` protocol.
    """
    return _Rename(original, new)


def placeholder():
    raise NotImplementedError


@dataclass(frozen=True)
class _PredictableInfo:
    # Tags are just strings.
    tags: Sequence[str] = field(default_factory=list)
    # Everything else gets serialized to a JSON object.
    hparams: JSONDict = field(default_factory=dict)
    metrics: JSONDict = field(default_factory=dict)
    dataset: JSONDict = field(default_factory=dict)
    task: JSONDict = field(default_factory=dict)
    misc: JSONDict = field(default_factory=dict)


# @dataclass(frozen=True)
# class _DefinedInfo:
#     name: str
#     module: str
#
#     @classmethod
#     def from_cls_or_func(cls, v):
#         if not (inspect.isclass(v) or inspect.isfunction(v)):
#             raise ValueError
#         mod = v.__module__
#         if mod == "__main__":
#             mod = sys.modules["__main__"].__name__
#         return _DefinedInfo(name=v.__name__, module=inspect.getmodulename)


@dataclass(frozen=True)
class GeneratedMeta:
    created_at: str
    sopredictable_version: str
    python_version: str
    os: str
    info: _PredictableInfo
    artifacts: List[_ArtifactMeta]
    artifact_load_order: List[str]
    env: _EnvironmentMeta


@dataclass(frozen=True)
class _ArtifactRecord:
    name: str
    value: Any
    save_fn: ArtifactSaver
    load_fn: ArtifactLoader
    deps: Tuple[_LoadDep, ...] = ()

    def __post_init__(self):
        assert self.value is not None
        assert isinstance(self.name, str) and len(self.name) > 0
        assert callable(self.save_fn)
        assert callable(self.load_fn)
        self._validate_deps()

    def _validate_deps(self):
        seen_names, seen_aliases = set(), set()
        for dep in self.deps:
            if dep.artifact_name in seen_names or dep.load_time_name in seen_aliases:
                raise ValueError(
                    "`deps` refers to the same artifact dependency more than once."
                )
            else:
                seen_names.add(dep.artifact_name)
                seen_aliases.add(dep.load_time_name)

    # def _save(self, artifact_dir: Path) -> None:
    #     artifact_dir.mkdir()
    #     artifact_data_dir = artifact_dir / ARTIFACT_DATA_DIR
    #     artifact_data_dir.mkdir()
    #     (artifact_dir / ARTIFACT_LOAD_FN_FILE).write_bytes(pickle.dumps(self.load_fn))
    #     (artifact_dir / ARTIFACT_DEPS_FILE).write_text(
    #         json.dumps([dataclasses.asdict(dep) for dep in self.deps])
    #     )
    #     self.save_fn(self.value, artifact_dir / ARTIFACT_DATA_DIR)

    def to_meta(self) -> _ArtifactMeta:
        return _ArtifactMeta(
            name=self.name,
            cls=type(self.value).__qualname__,
            save_fn=_qualname_from_user_callable(self.save_fn),
            load_fn=_qualname_from_user_callable(self.load_fn),
            deps=self.deps,
        )


@dataclass(frozen=True)
class _ArtifactMeta:
    name: str
    cls: str
    save_fn: str
    load_fn: str
    deps: Sequence[_LoadDep]


@dataclass(frozen=True)
class _EnvironmentMeta:
    pkgs: Optional[Sequence[str]] = None
    use_setup_py: bool = False
    py_dists: Optional[Sequence[str]] = None
    dockerfile: Optional[str] = None
    image_uri: Optional[str] = None
    conda_env: Optional[str] = None
    requirements: Optional[str] = None
    env_id: Optional[str] = None


def _qualname_from_user_callable(c: Callable) -> str:
    if inspect.isfunction(c):
        return c.__qualname__
    elif callable(c) and not inspect.isclass(c):
        return type(c).__qualname__
    else:
        raise ValueError


def _qualname_of_cls_or_fn_wrt_code_paths(v, code_paths: List[Path]) -> str:
    ...


@dataclass(frozen=True)
class _Environment:
    py_version: Optional[str] = None
    py_dist: Optional[Path] = None
    dockerfile: Optional[Path] = None
    image_uri: Optional[str] = None
    conda_env: Optional[Path] = None
    requirements: Optional[Path] = None
    env_id: Optional[str] = None

    def _to_meta(self) -> _EnvironmentMeta:
        return _EnvironmentMeta(
            py_dists=[self.py_dist.name] if self.py_dist else None,
            dockerfile=DOCKERFILE_FILE if self.dockerfile else None,
            image_uri=self.image_uri,
            conda_env=CONDA_FILE if self.conda_env else None,
            requirements=REQUIREMENTS_TXT_FILE if self.requirements else None,
            env_id=self.env_id,
        )

    def _save(self, env_dir: Path):
        if not env_dir.exists():
            env_dir.mkdir()
        if self.py_dist:
            dist_dir = env_dir / PY_DISTS_DIR
            shutil.copy(self.py_dist, dist_dir)
        for path_val, copy_dst in [
            (self.conda_env, CONDA_FILE),
            (self.requirements, REQUIREMENTS_TXT_FILE),
            (self.dockerfile, DOCKERFILE_FILE),
        ]:
            if path_val:
                src_path = Path(path_val).resolve()
                shutil.copy(src=src_path, dst=env_dir / copy_dst)
        for str_val, write_dst in [
            (self.env_id, "env_id.txt"),  # TODO: Refactor as path.
            (self.image_uri, "image_uri.txt"),
        ]:
            if str_val is not None:
                out_path: Path = env_dir / write_dst
                out_path.write_text(str_val)


@dataclass(frozen=True)
class _LoadDep:
    artifact_name: str
    load_time_name: Optional[str] = None

    def __post_init__(self):
        if self.load_time_name is None:
            object.__setattr__(self, "load_time_name", self.artifact_name)


# Saving
# ------


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Path):
            return str(o)
        else:
            return super().default(o)


class Saver:

    __slots__ = ("_predictor_cls", "_code", "_info", "_env", "_artifacts")

    _predictor_cls: Type[Predictable]
    _code: Optional[List[Path]]
    _info: _PredictableInfo
    _env: _Environment
    _artifacts: Tuple[_ArtifactRecord, ...]
    _options: Dict[str, bool]

    def __init__(
        self,
        predictor_cls: Type[Predictable],
        code: Union[PathLike, List[PathLike]],
    ) -> None:
        assert issubclass(
            predictor_cls, Predictable
        ), f"{predictor_cls} does not implement the `Predictable`"

        self._predictor_cls = predictor_cls
        self._env = _Environment()
        self._info = _PredictableInfo()
        self._artifacts = ()

        if code is None:
            self._code = None
        elif isinstance(code, (str, Path)):
            self._code = [Path(code).resolve()]
        elif isinstance(code, collections.Sequence):
            self._code = [Path(p).resolve() for p in code]
        else:
            raise ValueError

    def set_env(
        self,
        *,
        conda_env: PathLike = None,
        requirements: PathLike = None,
        py_dist: PathLike = None,
        dockerfile: PathLike = None,
        image_uri: str = None,
        env_id: str = None,
    ):
        current_env = self._env
        changes = {}
        for env_field, value, validator in [
            ("py_dist", py_dist, lambda x: Path(x)),
            ("dockerfile", dockerfile, lambda x: Path(x)),
            ("image_uri", image_uri, lambda x: Path(x)),
            ("conda_env", conda_env, lambda x: Path(x)),
            ("requirements", requirements, lambda x: Path(x)),
            ("env_id", env_id, None),
        ]:
            if value is not None:
                changes[env_field] = validator(value) if validator else value
        new_env = dataclasses.replace(current_env, **changes)
        return self._replace(_env=new_env)

    def set_info(
        self,
        *,
        tags: Sequence[str] = None,
        hparams: JSONDict = None,
        metrics: JSONDict = None,
        dataset: JSONDict = None,
        task: JSONDict = None,
        misc: JSONDict = None,
    ) -> Saver:
        current_info: _PredictableInfo = self._info
        changes = {}
        for info_field, value in [
            ("hparams", hparams),
            ("metrics", metrics),
            ("tags", tags),
            ("dataset", dataset),
            ("task", task),
            ("misc", misc),
        ]:
            if value is not None:
                changes[info_field] = value
        new_info = dataclasses.replace(current_info, **changes)
        return self._replace(_info=new_info)

    def add_artifact(
        self,
        name: str,
        artifact: T,
        save_fn: ArtifactSaver[T] = save_as_pickle,
        load_fn: ArtifactLoader[T] = load_from_pickle,
        deps: Sequence[Union[str, _Rename]] = (),
    ) -> Saver:
        deps_ = []
        for dep in deps:
            if isinstance(dep, str):
                deps_.append(_LoadDep(dep))
            elif isinstance(dep, _Rename):
                deps_.append(
                    _LoadDep(artifact_name=dep.original, load_time_name=dep.new)
                )
            else:
                raise TypeError(
                    f"`deps` member '{dep}' is neither a string nor an `_Alias`"
                )
        artifact_record = _ArtifactRecord(name, artifact, save_fn, load_fn, deps_)
        new_obj: Saver = self._replace(
            _artifacts=tuple((*self._artifacts, artifact_record))
        )
        return new_obj

    def save(self, path: PathLike, dry=False) -> None:
        if dry:
            raise NotImplementedError
        _save(Path(path), self)

    @property
    def info(self):
        return self._info

    @property
    def artifacts(self):
        return self._artifacts

    def _replace(self, **changes: Any) -> Saver:
        copy_of_self = copy.copy(self)
        for attr_name, value in changes.items():
            setattr(copy_of_self, attr_name, value)
        return copy_of_self


def _save(path: Path, saver: Saver) -> None:
    """
    Args:
        path: DIRECTORY
        saver (Saver): ...
    """
    # Prepare target directory
    dir_path = path
    if dir_path.exists():
        if sum(1 for _ in dir_path.iterdir()) > 0:
            raise FileExistsError(
                f"dir '{dir_path}' already exists. Caller should ensure it is removed."
            )
    else:
        dir_path.mkdir(parents=True)

    # Predictor class
    (dir_path / PREDICTOR_CLS_FILE).write_bytes(pickle.dumps(saver._predictor_cls))

    # get module path predictor to `code`
    predictor_module_name: str = saver._predictor_cls.__module__
    predictor_mod_path = None
    if predictor_module_name != "__main__":
        predictor_mod_path = predictor_module_name
    else:
        __main__mod = sys.modules[predictor_module_name]
        try:
            __main__mod_file = Path(__main__mod.__file__)
        except AttributeError:
            raise RuntimeError  # TODO: describe why sopredictable's Saver will not work in REPL
        for code_path in saver._code or ():
            try:
                rel_path = __main__mod_file.relative_to(code_path)
            except ValueError:
                raise RuntimeError
            if rel_path:
                rel_path_no_ext = rel_path.with_suffix("")
                full_mod_path = f"{code_path.name}.{'.'.join(rel_path_no_ext.parts)}"
                predictor_mod_path = full_mod_path

    # EXPERIMENTAL
    (dir_path / "predictor_cls.json").write_text(
        json.dumps({
            "name": saver._predictor_cls.__name__, "module": predictor_mod_path
        })
    )

    # Artifacts
    artifacts_dir = dir_path / ARTIFACTS_DIR
    artifacts_dir.mkdir()

    artifacts_meta: Dict[str, Any] = {}
    artifact_load_order = _compute_artifact_load_order(saver._artifacts)
    # - load order
    (artifacts_dir / ARTIFACTS_LOAD_ORDER_FILE).write_text(
        json.dumps(artifact_load_order)
    )

    artifacts_meta["load_order"] = artifact_load_order

    for artifact_record in saver._artifacts:
        artifact_dir = artifacts_dir / artifact_record.name
        artifact_data_dir = artifact_dir / ARTIFACT_DATA_DIR
        artifact_data_dir.mkdir(parents=True)
        (artifact_dir / ARTIFACT_LOAD_FN_FILE).write_bytes(
            pickle.dumps(artifact_record.load_fn)
        )
        (artifact_dir / ARTIFACT_DEPS_FILE).write_text(
            json.dumps([dataclasses.asdict(dep) for dep in artifact_record.deps])
        )
        artifact_record.save_fn(artifact_record.value, artifact_dir / ARTIFACT_DATA_DIR)

    # Environment
    env_dir = dir_path / ENV_DIR
    saver._env._save(env_dir)

    # Finally, all metadata.
    json_serializable_meta = dataclasses.asdict(_generate_meta(saver))
    print(f"{json_serializable_meta=}")
    (
        (dir_path / META_FILE).write_text(
            json.dumps(json_serializable_meta, indent=2, cls=EnhancedJSONEncoder)
        )
    )

    code_dir = dir_path / CODE_DIR
    code_dir.mkdir()
    _ignorer = shutil.ignore_patterns("__pycache__", "*.pyc", ".*")
    for path_ in saver._code or ():
        shutil.copytree(
            path_,
            code_dir / path_.name,
            ignore=_ignorer,
            copy_function=shutil.copy,
        )


def _generate_meta(saver: Saver):
    from datetime import datetime
    from sopredictable.about import __version__
    import platform

    info = saver._info
    meta = GeneratedMeta(
        info=info,
        created_at=datetime.now().astimezone().isoformat(timespec="milliseconds"),
        sopredictable_version=__version__,
        python_version=platform.python_version(),
        os=platform.platform(),
        artifacts=[artifact.to_meta() for artifact in saver._artifacts],
        artifact_load_order=_compute_artifact_load_order(saver._artifacts),
        env=saver._env._to_meta(),
    )
    print(meta)
    return meta


def _dependency_graph_from_artifacts(
    artifacts: Sequence[_ArtifactRecord],
) -> Dict[ArtifactName, Sequence[ArtifactName]]:
    return {
        artifact.name: [dep.artifact_name for dep in artifact.deps]
        for artifact in artifacts
    }


def _compute_artifact_load_order(
    artifacts: Sequence[_ArtifactRecord],
) -> List[ArtifactName]:
    dependency_graph = _dependency_graph_from_artifacts(artifacts)
    # TODO: handle cycles
    load_order = cast(List[str], iterative_topological_sort(dependency_graph))
    return load_order


# Loading
# -------


def load(path: PathLike) -> Union[Predictable, Tuple[Predictable, _PredictableInfo]]:
    predictor, _ = _load(path)
    return predictor


def load_with_info(path: PathLike) -> Tuple[Predictable, _PredictableInfo]:
    predictor, meta = _load(path)
    return predictor, meta["info"]  # TODO: fix type signature


def _load(dir_path: PathLike) -> Tuple[Predictable, GeneratedMeta]:
    dir_path = Path(dir_path)

    if not dir_path.exists():
        raise ValueError(f"`{dir_path}` does not exist.")

    pkgs_dir = dir_path / CODE_DIR
    # for path_ in pkgs_dir.iterdir():
    #     if path_.is_dir():
    sys.path.insert(0, str(pkgs_dir.resolve()))

    print(sys.modules)
    print(sys.path)

    predictor_cls_spec = json.loads((dir_path / "predictor_cls.json").read_text())
    predictor_module = importlib.import_module(predictor_cls_spec["module"])
    predictor_cls = getattr(predictor_module, predictor_cls_spec["name"])
    print(predictor_cls)
    # breakpoint()


    meta: GeneratedMeta = read_meta(dir_path / META_FILE)

    artifacts_dir = dir_path / ARTIFACTS_DIR

    artifact_load_order = json.loads(
        (artifacts_dir / ARTIFACTS_LOAD_ORDER_FILE).read_text()
    )
    loaded_artifacts: Dict[str, Any] = {}

    for artifact_name in artifact_load_order:
        artifact_dir = artifacts_dir / artifact_name
        load_fn = pickle.loads((artifact_dir / ARTIFACT_LOAD_FN_FILE).read_bytes())
        deps_data: List[dict] = json.loads(
            (artifact_dir / ARTIFACT_DEPS_FILE).read_text()
        )
        deps_arg = {
            dep["load_time_name"]: loaded_artifacts[dep["artifact_name"]]
            for dep in deps_data
        }
        artifact = load_fn(artifact_dir / ARTIFACT_DATA_DIR, deps=deps_arg)
        loaded_artifacts[artifact_name] = artifact

    # predictor_cls: Type[Predictable] = pickle.loads(
    #     (dir_path / PREDICTOR_CLS_FILE).read_bytes()
    # )
    predictor: Predictable = predictor_cls.from_artifacts_dict(loaded_artifacts)
    return predictor, meta


def read_meta(meta_json_path: Path) -> GeneratedMeta:  # noqa
    meta_data = json.loads(meta_json_path.read_text())
    return meta_data
    # env = _EnvironmentMeta(**meta_data.pop("env"))
    # info = _PredictableInfo(**meta_data.pop("info"))
    # artifacts = {}
    # for af_name, af_data in meta_data.pop("artifacts").items():
    #     deps = [_LoadDep(**dep_data) for dep_data in af_data.pop("deps")]
    #     artifacts[af_name] = _ArtifactMeta(deps=deps, **af_data)
    # return GeneratedMeta(info=info, artifacts=artifacts, env=env, **meta_data)


__all__ = ["Predictable", "load", "load_with_info", "Saver"]
