import json

import pytest

from eaddit.cli import main


def _write_fixture(tmp_path, sample_data):
    p = tmp_path / "fixture.json"
    p.write_text(json.dumps(sample_data), encoding="utf-8")
    return str(p)


def test_ingest_query_info_round_trip(tmp_path, sample_data, capsys):
    fixture = _write_fixture(tmp_path, sample_data)
    store_path = str(tmp_path / "store.json")

    rc = main([
        "ingest", "python",
        "--source", "json", "--fixture", fixture,
        "--store", store_path,
        "--min-score", "1",
        "--dim", "128",
    ])
    assert rc == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["posts_seen"] == 2
    assert stats["chunks_added"] > 0

    rc = main([
        "info", "--store", store_path,
    ])
    assert rc == 0
    info = json.loads(capsys.readouterr().out)
    assert info["chunks"] == stats["chunks_added"]
    assert info["dimension"] == 128
    assert "python" in info["subreddits"]

    rc = main([
        "query", "how do I read large CSV files",
        "--store", store_path,
        "--dim", "128",
        "--top-k", "2",
    ])
    assert rc == 0
    results = json.loads(capsys.readouterr().out)
    assert len(results) == 2
    assert all("chunk" in r and "score" in r for r in results)


def test_query_prompt_format(tmp_path, sample_data, capsys):
    fixture = _write_fixture(tmp_path, sample_data)
    store_path = str(tmp_path / "store.json")
    main([
        "ingest", "python",
        "--source", "json", "--fixture", fixture,
        "--store", store_path,
        "--dim", "64",
    ])
    capsys.readouterr()

    rc = main([
        "query", "python tips",
        "--store", store_path,
        "--dim", "64",
        "--format", "prompt",
        "--top-k", "1",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Question: python tips" in out
    assert "[Excerpt 1]" in out


def test_query_min_score_filter(tmp_path, sample_data, capsys):
    fixture = _write_fixture(tmp_path, sample_data)
    store_path = str(tmp_path / "store.json")
    main([
        "ingest", "python",
        "--source", "json", "--fixture", fixture,
        "--store", store_path,
        "--dim", "64",
    ])
    capsys.readouterr()
    rc = main([
        "query", "python",
        "--store", store_path,
        "--dim", "64",
        "--top-k", "20",
        "--min-score", "100",
    ])
    assert rc == 0
    results = json.loads(capsys.readouterr().out)
    assert results == []  # no chunk meets that score floor


def test_ingest_requires_fixture_for_json_source(tmp_path):
    with pytest.raises(SystemExit):
        main([
            "ingest", "python",
            "--source", "json",
            "--store", str(tmp_path / "store.json"),
        ])


def test_query_without_store_fails(tmp_path):
    with pytest.raises(SystemExit):
        main([
            "query", "x",
            "--store", str(tmp_path / "missing.json"),
        ])


def test_dim_mismatch_rejected_on_query(tmp_path, sample_data):
    fixture = _write_fixture(tmp_path, sample_data)
    store_path = str(tmp_path / "store.json")
    main([
        "ingest", "python",
        "--source", "json", "--fixture", fixture,
        "--store", store_path,
        "--dim", "64",
    ])
    with pytest.raises(SystemExit):
        main([
            "query", "anything",
            "--store", store_path,
            "--dim", "128",
        ])
