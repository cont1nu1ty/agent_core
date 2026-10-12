"""Run core contracts with host imports and external I/O forbidden.

Use --wheel to extract and test a built wheel outside the repository. Dependencies
must already exist in the selected Python environment; this never installs them.
"""
from argparse import ArgumentParser
import importlib.abc
from pathlib import Path
import os
import sys
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

PACKAGE = Path(__file__).resolve().parents[1]


def main():
    parser = ArgumentParser()
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    forbidden = {"state_grid", "crewai", "fastapi", "psycopg", "neo4j"}
    attempts = []

    class BlockHost(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in forbidden:
                attempts.append(fullname)
                raise ImportError(f"Core isolation forbids {fullname}")
            return None

    def deny_io(event, _args):
        if event in {"socket.connect", "socket.getaddrinfo", "sqlite3.connect", "subprocess.Popen"}:
            attempts.append(event)
            raise RuntimeError(f"Core isolation forbids {event}")

    with TemporaryDirectory(prefix="agent-core-wheel-") as temporary:
        if args.wheel:
            with ZipFile(args.wheel.resolve()) as wheel:
                names = wheel.namelist()
                if not any(name.startswith("agent_core/") for name in names):
                    raise AssertionError("wheel has no core package")
                if any(name.startswith(("state_grid/", "agent_core_tests/")) for name in names):
                    raise AssertionError("wheel includes host or tests")
                wheel.extractall(temporary)
            sys.path.insert(0, temporary)
        else:
            sys.path.insert(0, str(PACKAGE / "src"))
        sys.path.insert(0, str(PACKAGE / "tests"))
        os.chdir(temporary)
        sys.meta_path.insert(0, BlockHost())
        sys.addaudithook(deny_io)
        import agent_core
        if args.wheel and not Path(agent_core.__file__).is_relative_to(temporary):
            raise AssertionError("did not import the built wheel")
        suite = unittest.defaultTestLoader.discover(str(PACKAGE / "tests"), pattern="test_*.py")
        if suite.countTestCases() == 0:
            raise AssertionError("core contract test collection is empty")
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        print({"source": "wheel" if args.wheel else "checkout", "tests": result.testsRun,
               "failures": len(result.failures), "errors": len(result.errors),
               "skipped": len(result.skipped), "blocked_attempts": attempts})
        return int(not result.wasSuccessful() or bool(attempts))


if __name__ == "__main__":
    raise SystemExit(main())
