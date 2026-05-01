import json

from eaddit.collector import JSONFixtureCollector


def test_collect_returns_posts_and_comments(collector):
    posts, comments = collector.collect("python", limit=10, min_score=-100)
    assert {p.id for p in posts} == {"p1", "p2", "p3"}
    assert {c.id for c in comments} == {"c1", "c2", "c3", "c4"}


def test_collect_filters_by_min_score(collector):
    posts, comments = collector.collect("python", min_score=1)
    assert {p.id for p in posts} == {"p1", "p2"}
    # Comments belonging to dropped posts are also dropped, plus low-score ones.
    assert {c.id for c in comments} == {"c1", "c2", "c3"}


def test_collect_unknown_subreddit_returns_empty(collector):
    posts, comments = collector.collect("nope")
    assert posts == [] and comments == []


def test_collect_respects_limit_and_drops_orphan_comments(sample_data):
    coll = JSONFixtureCollector(sample_data)
    posts, comments = coll.collect("python", limit=1)
    assert len(posts) == 1
    # Only comments for the kept post survive.
    kept_post = posts[0].id
    assert all(c.post_id == kept_post for c in comments)


def test_collector_loads_from_path(tmp_path, sample_data):
    p = tmp_path / "data.json"
    p.write_text(json.dumps(sample_data), encoding="utf-8")
    coll = JSONFixtureCollector(p)
    posts, _ = coll.collect("python", min_score=-100)
    assert len(posts) == 3
