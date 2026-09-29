from tools.inspect_installed_build import fingerprint


def test_packaged_comparison_ignores_install_path_and_line_numbers():
    before = compile("def value():\n    return 12\n", "a/source.py", "exec")
    after = compile("\n\ndef value():\n    return 12\n", "b/installed.py", "exec")
    assert fingerprint(before) == fingerprint(after)
    different = compile("def value():\n    return 13\n", "a/source.py", "exec")
    assert fingerprint(before) != fingerprint(different)


def test_inspection_does_not_inherit_its_own_future_flags(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys
    from tools.inspect_installed_build import inspect_build
    source = b'"A package without future annotations."\nvalue = 1\n'
    (tmp_path / "orion_core").mkdir()
    (tmp_path / "orion_core/__init__.py").write_bytes(source)
    executable = tmp_path / "test.exe"
    executable.write_bytes(b"fixture")
    archive = SimpleNamespace(toc={"orion_core": None},
        extract=lambda _: compile(source, "original.py", "exec", dont_inherit=True))
    readers = SimpleNamespace(CArchiveReader=lambda _: SimpleNamespace(
        toc={"PYZ.pyz": None}, open_embedded_archive=lambda _: archive))
    monkeypatch.setitem(sys.modules, "PyInstaller.archive.readers", readers)
    assert inspect_build(executable, tmp_path)["source_matches"]


def test_comparison_ignores_constant_interning_and_object_sharing():
    shared = "a reasonably long constant"
    independent = (shared + "x")[:-1]
    assert shared == independent and shared is not independent
    code = compile("x = 'value'", "source.py", "exec")
    before = code.replace(co_consts=(shared, shared))
    after = code.replace(co_consts=(shared, independent))
    assert fingerprint(before) == fingerprint(after)


def test_a_module_this_python_cannot_compile_is_reported_not_fatal(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import hashlib
    import sys
    from tools.inspect_installed_build import inspect_build
    (tmp_path / "orion_core").mkdir()
    (tmp_path / "orion_core/__init__.py").write_bytes(b"value = 1\n")
    (tmp_path / "orion_core/newer.py").write_bytes(b"def f(:\n    pass\n")
    executable = tmp_path / "test.exe"
    executable.write_bytes(b"fixture" * 1000)
    archive = SimpleNamespace(toc={"orion_core": None, "orion_core.newer": None},
        extract=lambda _: compile("value = 1\n", "x.py", "exec", dont_inherit=True))
    readers = SimpleNamespace(CArchiveReader=lambda _: SimpleNamespace(
        toc={"PYZ.pyz": None}, open_embedded_archive=lambda _: archive))
    monkeypatch.setitem(sys.modules, "PyInstaller.archive.readers", readers)
    result = inspect_build(executable, tmp_path)
    assert result["unreadable_modules"] == ["orion_core.newer"]
    assert result["matching_modules"] == ["orion_core"]
    assert not result["source_matches"]
    assert result["sha256"] == hashlib.sha256(b"fixture" * 1000).hexdigest()
