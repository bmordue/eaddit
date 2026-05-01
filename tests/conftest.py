"""Pytest fixtures shared across the test suite."""

from __future__ import annotations

import pytest

from eaddit.collector import JSONFixtureCollector


SAMPLE_DATA = {
    "python": {
        "posts": [
            {
                "id": "p1",
                "title": "How to learn list comprehensions",
                "body": "I'm new to Python and want quick tips on list comprehensions.",
                "score": 42,
                "url": "https://example.com/p1",
                "created_utc": 1_700_000_000,
                "subreddit": "python",
                "author": "alice",
            },
            {
                "id": "p2",
                "title": "Fastest way to read large CSV files",
                "body": "What library do you use for big CSVs?",
                "score": 10,
                "url": "https://example.com/p2",
                "created_utc": 1_700_000_500,
                "subreddit": "python",
                "author": "bob",
            },
            {
                "id": "p3",
                "title": "Low quality post",
                "body": "spam spam spam",
                "score": -5,
                "url": "https://example.com/p3",
                "created_utc": 1_699_000_000,
                "subreddit": "python",
                "author": "spammer",
            },
        ],
        "comments": [
            {
                "id": "c1",
                "post_id": "p1",
                "parent_id": "p1",
                "body": "Use [x for x in xs if cond] — it's pythonic and fast.",
                "score": 8,
                "created_utc": 1_700_000_100,
                "author": "carol",
                "depth": 1,
            },
            {
                "id": "c2",
                "post_id": "p1",
                "parent_id": "c1",
                "body": "Agreed. Generator expressions are even better for huge inputs.",
                "score": 4,
                "created_utc": 1_700_000_200,
                "author": "dave",
                "depth": 2,
            },
            {
                "id": "c3",
                "post_id": "p2",
                "parent_id": "p2",
                "body": "Try pandas.read_csv with chunksize for very large CSV files.",
                "score": 12,
                "created_utc": 1_700_000_600,
                "author": "erin",
                "depth": 1,
            },
            {
                "id": "c4",
                "post_id": "p2",
                "parent_id": "p2",
                "body": "downvoted nonsense",
                "score": -3,
                "created_utc": 1_700_000_700,
                "author": "troll",
                "depth": 1,
            },
        ],
    }
}


@pytest.fixture
def sample_data():
    # Return a deep-ish copy so tests can mutate without bleeding into siblings.
    import copy

    return copy.deepcopy(SAMPLE_DATA)


@pytest.fixture
def collector(sample_data):
    return JSONFixtureCollector(sample_data)
