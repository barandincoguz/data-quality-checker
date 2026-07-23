from __future__ import annotations

import json
import stat
import zipfile

import pytest

from data_quality_checker.config import SecurityConfig
from data_quality_checker.errors import UnsafeArchive
from data_quality_checker.safezip import inspect_archive, read_json_records, validate_zip_info


def limits(**overrides) -> SecurityConfig:
    values = {
        "max_zip_entries": 10,
        "max_zip_entry_bytes": 10_000,
        "max_zip_total_bytes": 20_000,
        "max_zip_compression_ratio": 200.0,
    }
    values.update(overrides)
    return SecurityConfig(**values)


def write_zip(path, entries, *, compression=zipfile.ZIP_DEFLATED) -> None:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)


@pytest.mark.parametrize("name", ["../escape.json", "/absolute.json", "C:/drive.json"])
def test_rejects_zip_slip_and_absolute_paths(tmp_path, name) -> None:
    path = tmp_path / "bad.zip"
    write_zip(path, [(name, "{}")])
    with pytest.raises(UnsafeArchive, match="path"):
        inspect_archive(path, limits())


def test_rejects_normalized_duplicate_paths(tmp_path) -> None:
    path = tmp_path / "duplicate.zip"
    write_zip(path, [("DATA.json", "{}"), ("data.json", "{}")])
    with pytest.raises(UnsafeArchive, match="duplicate"):
        inspect_archive(path, limits())


def test_rejects_symlink_entry(tmp_path) -> None:
    path = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("link.json")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(link, "target")
    with pytest.raises(UnsafeArchive, match="symlink"):
        inspect_archive(path, limits())


def test_rejects_encrypted_flag_without_attempting_read() -> None:
    info = zipfile.ZipInfo("secret.json")
    info.flag_bits |= 0x1
    with pytest.raises(UnsafeArchive, match="encrypted"):
        validate_zip_info(info, limits())


def test_rejects_entry_total_ratio_and_entry_count_limits(tmp_path) -> None:
    oversized = tmp_path / "oversized.zip"
    write_zip(oversized, [("data.json", json.dumps({"x": "123456789"}))])
    with pytest.raises(UnsafeArchive, match="exceeds"):
        inspect_archive(oversized, limits(max_zip_entry_bytes=5))

    bomb = tmp_path / "bomb.zip"
    write_zip(bomb, [("data.json", json.dumps({"x": "0" * 9000}))])
    with pytest.raises(UnsafeArchive, match="compression ratio"):
        inspect_archive(bomb, limits(max_zip_compression_ratio=2.0))

    many = tmp_path / "many.zip"
    write_zip(many, [(f"{i}.json", "{}") for i in range(3)])
    with pytest.raises(UnsafeArchive, match="maximum"):
        inspect_archive(many, limits(max_zip_entries=2))


def test_rejects_bad_json_and_never_extracts(tmp_path) -> None:
    path = tmp_path / "bad-json.zip"
    write_zip(path, [("nested/data.json", "not-json")])
    with pytest.raises(UnsafeArchive, match="invalid JSON"):
        read_json_records(path, limits())
    assert not (tmp_path / "nested").exists()


def test_reads_json_wrappers_and_jsonl(tmp_path) -> None:
    path = tmp_path / "records.zip"
    write_zip(
        path,
        [
            ("a.json", json.dumps({"annotations": [{"document_id": "1"}]})),
            ("b.jsonl", '{"document_id":"2"}\n{"document_id":"3"}\n'),
        ],
    )
    audit, records = read_json_records(path, limits())
    assert audit.json_entry_count == 2
    assert [record.payload["document_id"] for record in records] == ["1", "2", "3"]
