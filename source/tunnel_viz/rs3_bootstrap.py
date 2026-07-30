"""Make RS3 protobuf stubs importable under PyInstaller.

RS3's generated gRPC/protobuf modules use bare imports such as
``import BeamsResultsQueryService_pb2``. Those only work when
``rs3/generatedFiles`` is on ``sys.path`` as real files on disk.

In a frozen app those modules live as ``rs3.generatedFiles.*`` inside the
archive, so bare imports fail. This installer aliases the package modules
under their bare names (and still adds the on-disk folder when present).
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from typing import Sequence


_INSTALLED = False


class _Rs3Pb2AliasFinder(importlib.abc.MetaPathFinder):
    """Resolve bare ``*_pb2`` / ``*_pb2_grpc`` to ``rs3.generatedFiles.*``."""

    def __init__(self) -> None:
        self._busy: set[str] = set()

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: object | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if "." in fullname:
            return None
        if not (fullname.endswith("_pb2") or fullname.endswith("_pb2_grpc")):
            return None

        real_name = f"rs3.generatedFiles.{fullname}"
        if fullname in self._busy or real_name in self._busy:
            return None

        existing = sys.modules.get(real_name)
        if existing is not None:
            sys.modules[fullname] = existing
            return importlib.util.spec_from_loader(
                fullname,
                _ExistingModuleLoader(existing),
                origin=getattr(existing, "__file__", None),
            )

        self._busy.add(real_name)
        try:
            real_spec = importlib.util.find_spec(real_name)
        finally:
            self._busy.discard(real_name)

        if real_spec is None or real_spec.loader is None:
            return None

        return importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(real_name, real_spec.loader),
            origin=getattr(real_spec, "origin", None),
        )


class _ExistingModuleLoader(importlib.abc.Loader):
    def __init__(self, module: object) -> None:
        self._module = module

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> object:
        return self._module

    def exec_module(self, module: object) -> None:
        return None


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, real_name: str, real_loader: importlib.abc.Loader) -> None:
        self._real_name = real_name
        self._real_loader = real_loader

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> object | None:
        module = importlib.import_module(self._real_name)
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module: object) -> None:
        return None


def ensure_rs3_protobuf_imports() -> None:
    """Install import aliases for RS3 generated protobuf modules."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    if not any(isinstance(finder, _Rs3Pb2AliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Rs3Pb2AliasFinder())

    try:
        gf = importlib.import_module("rs3.generatedFiles")
        gf_dir = getattr(gf, "__path__", [None])[0]
        if gf_dir and gf_dir not in sys.path:
            sys.path.insert(0, gf_dir)
    except Exception:
        pass
