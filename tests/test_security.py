import pytest
import argparse
from eaddit.chunker import Chunker, MAX_TEXT_LENGTH
from eaddit.cli import cmd_query, MAX_QUERY_LENGTH
from eaddit.models import Post

def test_chunker_max_length_enforced():
    chunker = Chunker()
    # Create a post with text exceeding the limit
    long_text = "a" * (MAX_TEXT_LENGTH + 1)
    post = Post(
        id="test",
        title="title",
        body=long_text,
        score=1,
        url="http://example.com",
        created_utc=0,
        subreddit="test",
        author="test"
    )

    with pytest.raises(ValueError, match="Input text too long"):
        chunker.chunk_post(post)

def test_cli_query_max_length_enforced():
    args = argparse.Namespace(
        text="a" * (MAX_QUERY_LENGTH + 1),
        dim=256,
        store="nonexistent_store.json"
    )

    with pytest.raises(SystemExit, match="Query too long"):
        cmd_query(args)
