"""The LLM port: output validation, and the cache stamp.

**Raw model text is never trusted** (D10). Two things enforce that, and both are tested
here rather than discovered in production: extraction survives the wrappers models actually
emit, and validation rejects the shapes that would otherwise reach the domain.

The cache tests cover D2's requirement that judgments are persisted data. The property that
matters is not "it caches" but **that the key includes `{provider, model, prompt_version}`**
— without it, changing the default model silently serves judgments a different model made.
"""

import json

import pytest

from paperthread.llm.base import SchemaError, extract_json, validate
from paperthread.llm.cache import LLMCache

SCHEMA = {
    "type": "object",
    "required": ["papers"],
    "properties": {
        "papers": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["id", "role", "confidence"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "role": {"type": "string", "enum": ["foundation", "survey"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        }
    },
}

VALID = {"papers": [{"id": "n1", "role": "survey", "confidence": 0.9}]}


class TestExtractJson:
    def test_bare_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_markdown_fence(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_unlabelled_fence(self):
        assert extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_preamble_and_trailing_prose(self):
        """Reasoning models emit a preamble despite being told not to. Tolerating the
        wrapper is safe because the payload is schema-validated regardless."""
        assert extract_json('Sure! Here it is:\n{"a": [1,2]}\nHope that helps.') == {"a": [1, 2]}

    @pytest.mark.parametrize("text", ["", "   ", "no json here", "[1, 2, 3]"])
    def test_rejects_unusable_responses(self, text):
        with pytest.raises(ValueError):
            extract_json(text)


class TestValidate:
    def test_accepts_valid(self):
        validate(VALID, SCHEMA)

    def test_missing_required_field(self):
        with pytest.raises(SchemaError, match="confidence"):
            validate({"papers": [{"id": "n1", "role": "survey"}]}, SCHEMA)

    def test_value_outside_enum(self):
        with pytest.raises(SchemaError, match="not one of"):
            validate({"papers": [{"id": "n1", "role": "invented", "confidence": 0.5}]}, SCHEMA)

    def test_number_out_of_range(self):
        with pytest.raises(SchemaError, match=">= 0|<= 1"):
            validate({"papers": [{"id": "n1", "role": "survey", "confidence": 1.7}]}, SCHEMA)

    def test_boolean_is_not_a_number(self):
        """`bool` subclasses `int` in Python, so a naive isinstance check accepts `true`
        as a confidence score and it silently becomes 1.0."""
        with pytest.raises(SchemaError, match="expected a number"):
            validate({"papers": [{"id": "n1", "role": "survey", "confidence": True}]}, SCHEMA)

    def test_wrong_container_type(self):
        with pytest.raises(SchemaError, match="expected an array"):
            validate({"papers": {"id": "n1"}}, SCHEMA)

    def test_min_items_enforced(self):
        with pytest.raises(SchemaError, match="at least 1 items"):
            validate({"papers": []}, SCHEMA)

    def test_error_message_names_the_path(self):
        """Validation errors are fed back to the model as repair instructions, so they
        have to say WHERE the problem is."""
        with pytest.raises(SchemaError, match=r"\$\.papers\[0\]\.role"):
            validate({"papers": [{"id": "n1", "role": "bogus", "confidence": 0.5}]}, SCHEMA)


class TestCache:
    def test_roundtrip(self, tmp_path):
        cache = LLMCache(tmp_path)
        key = cache.key(
            provider="p", model="m", prompt_version="1", role="r", system="s", user="u"
        )
        assert cache.get(key) is None
        cache.put(key, {"data": VALID})
        assert cache.get(key) == {"data": VALID}

    @pytest.mark.parametrize(
        "field", ["provider", "model", "prompt_version", "role", "system", "user"]
    )
    def test_every_stamp_component_changes_the_key(self, tmp_path, field):
        """The point of the stamp: changing the model must MISS, not silently serve a
        judgment some other model made (D10)."""
        cache = LLMCache(tmp_path)
        base = dict(provider="p", model="m", prompt_version="1", role="r", system="s", user="u")
        changed = {**base, field: "different"}
        assert cache.key(**base) != cache.key(**changed)

    def test_disabled_cache_never_reads_or_writes(self, tmp_path):
        cache = LLMCache(tmp_path, enabled=False)
        key = cache.key(provider="p", model="m", prompt_version="1", role="r", system="s", user="u")
        cache.put(key, {"data": VALID})
        assert cache.get(key) is None
        assert not list(tmp_path.rglob("*.json"))

    def test_corrupt_entry_is_a_miss_not_an_error(self, tmp_path):
        """Losing a cached judgment costs one API call. Raising costs the user their path."""
        cache = LLMCache(tmp_path)
        key = cache.key(provider="p", model="m", prompt_version="1", role="r", system="s", user="u")
        cache.put(key, {"data": VALID})
        path = next(tmp_path.rglob("*.json"))
        path.write_text("{ truncated")
        assert cache.get(key) is None

    def test_prompt_text_edit_invalidates_without_a_version_bump(self, tmp_path):
        """Version bumps are for humans reading provenance; the hash is what protects the
        cache when someone edits a prompt and forgets."""
        cache = LLMCache(tmp_path)
        original = cache.key(
            provider="p", model="m", prompt_version="1", role="r", system="old prompt", user="u"
        )
        edited = cache.key(
            provider="p", model="m", prompt_version="1", role="r", system="new prompt", user="u"
        )
        assert original != edited
