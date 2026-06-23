"""Unit tests for the output validators (self-correction framework).

Pure / containerless. Covers ShortLabelValidator, ProperNameValidator,
BoundedTextValidator, ForEachValidator (detect + normalize), and the Violation
model.
"""

from agentrunner.output_validation import (
    BoundedTextValidator,
    ForEachValidator,
    ProperNameValidator,
    ShortLabelValidator,
    Violation,
)

# ---------------------------------------------------------------------------
# Fixtures / shared test data
# ---------------------------------------------------------------------------

# The exact value from the BOU-1743 origin bug:
# "Stonecarver Who Chips Away At Rock And Stone Until The Answers Emerge"
# Word count = 14, len = 68 chars — but spec says 105 char.
# Use a 105-char phrase that also has 14 words:
_LONG_VALUE = "Stonecarver Who Chips Away At Rock And Stone Until The Answers Will Emerge From" + " " * 27
# Make it exactly 105 non-space chars:
LONG_PHRASE = "Stonecarver Who Chips Away At Rock And Stone Until The Answers Will Emerge From Depths"
# Verify our test value (15 words, well over max_words=3 and max_chars=40):
assert len(LONG_PHRASE.split()) > 3, f"expected >3 words, got {len(LONG_PHRASE.split())}"
assert len(LONG_PHRASE) > 40, f"expected >40 chars"

SHORT_LABEL = "Stonecarver"


# ---------------------------------------------------------------------------
# ShortLabelValidator tests
# ---------------------------------------------------------------------------


def test_short_label_detects_long_phrase():
    """14-word phrase should flag a violation."""
    val = ShortLabelValidator(field="character_class", max_words=3, max_chars=40)

    class FakeOutput:
        character_class = LONG_PHRASE

    violations = val.validate(FakeOutput())
    assert violations, "Expected at least one violation for a 14-word phrase"
    assert violations[0].field == "character_class"


def test_short_label_passes_short_value():
    """Single word 'Stonecarver' should pass."""
    val = ShortLabelValidator(field="character_class", max_words=3, max_chars=40)

    class FakeOutput:
        character_class = SHORT_LABEL

    violations = val.validate(FakeOutput())
    assert violations == [], f"Expected no violations for {SHORT_LABEL!r}, got {violations}"


def test_short_label_normalize_caps_length():
    """normalize() must return a string ≤ max_chars and ≤ max_words words."""
    max_words = 3
    max_chars = 40

    val = ShortLabelValidator(field="character_class", max_words=max_words, max_chars=max_chars)

    class FakeOutput:
        character_class = LONG_PHRASE

    normalized = val.normalize(FakeOutput())
    result = normalized.character_class
    assert isinstance(result, str), "normalized value should be a str"
    assert len(result) <= max_chars, (
        f"normalize() returned {len(result)} chars, expected ≤ {max_chars}: {result!r}"
    )
    assert len(result.split()) <= max_words, (
        f"normalize() returned {len(result.split())} words, expected ≤ {max_words}: {result!r}"
    )
    # Explicit guard for VARCHAR(100) constraint
    assert len(result) < 100, (
        f"normalize() output MUST be < 100 chars to fit character_profiles.character_class "
        f"VARCHAR(100), got {len(result)}: {result!r}"
    )


def test_short_label_normalize_on_dict():
    """normalize() must handle plain dict outputs."""
    val = ShortLabelValidator(field="character_class", max_words=3, max_chars=40)
    output = {"character_class": LONG_PHRASE, "other_field": "preserved"}
    result = val.normalize(output)
    assert isinstance(result, dict)
    assert result["other_field"] == "preserved", "other fields must be preserved"
    assert len(result["character_class"]) <= 40


def test_short_label_normalize_on_pydantic_model():
    """normalize() must handle Pydantic model outputs via model_copy."""
    from pydantic import BaseModel

    class FakeModel(BaseModel):
        character_class: str
        name: str = "Aria"

    val = ShortLabelValidator(field="character_class", max_words=3, max_chars=40)
    m = FakeModel(character_class=LONG_PHRASE)
    result = val.normalize(m)
    assert isinstance(result, FakeModel), "Should return a FakeModel instance"
    assert len(result.character_class) <= 40
    assert result.name == "Aria", "Other fields must be preserved"


def test_short_label_skips_none_value():
    """validate/normalize should be no-ops when field is None."""
    val = ShortLabelValidator(field="character_class", max_words=3, max_chars=40)

    class FakeOutput:
        character_class = None

    assert val.validate(FakeOutput()) == []
    result = val.normalize(FakeOutput())
    assert result.character_class is None


def test_short_label_skips_missing_field():
    """validate/normalize should be no-ops when field is absent."""
    val = ShortLabelValidator(field="character_class", max_words=3, max_chars=40)

    class FakeOutput:
        pass

    assert val.validate(FakeOutput()) == []


def test_short_label_flags_sentence_punctuation():
    """Values containing '.' should be flagged."""
    val = ShortLabelValidator(field="character_class", max_words=3, max_chars=40)

    class FakeOutput:
        character_class = "Warrior. Fighter"

    violations = val.validate(FakeOutput())
    assert violations, "Expected a violation for value containing '.'"


def test_short_label_flags_comma():
    """Values containing ',' should be flagged."""
    val = ShortLabelValidator(field="character_class", max_words=3, max_chars=40)

    class FakeOutput:
        character_class = "Shield, Warrior"

    violations = val.validate(FakeOutput())
    assert violations, "Expected a violation for value containing ','"


# ---------------------------------------------------------------------------
# ProperNameValidator tests
# ---------------------------------------------------------------------------


def test_proper_name_detects_sentence_name():
    """A sentence-y long name should be flagged."""
    val = ProperNameValidator(field="name", max_words=4)

    class FakeOutput:
        name = "Joren Stark The Brave Warrior Of The North"

    violations = val.validate(FakeOutput())
    assert violations, "Expected violation for long sentence-y name"
    assert violations[0].field == "name"


def test_proper_name_detects_period():
    """A name with a period should be flagged."""
    val = ProperNameValidator(field="name", max_words=4)

    class FakeOutput:
        name = "Dr. Joren Stark"

    violations = val.validate(FakeOutput())
    assert violations, "Expected violation for name with period"


def test_proper_name_passes_short_name():
    """'Joren Stark' (2 words, no punctuation) should pass."""
    val = ProperNameValidator(field="name", max_words=4)

    class FakeOutput:
        name = "Joren Stark"

    violations = val.validate(FakeOutput())
    assert violations == [], f"Expected no violations, got {violations}"


def test_proper_name_normalize():
    """normalize() should keep the first max_words tokens."""
    val = ProperNameValidator(field="name", max_words=4)

    class FakeOutput:
        name = "Joren Stark The Brave Warrior Of The North"

    result = val.normalize(FakeOutput())
    words = result.name.split()
    assert len(words) <= 4, f"Expected ≤4 words, got {words}"
    assert result.name.startswith("Joren"), "Name should start with 'Joren'"


# ---------------------------------------------------------------------------
# Violation model basic tests
# ---------------------------------------------------------------------------


def test_violation_model_fields():
    v = Violation(
        field="character_class", kind="too_long", message="too long", suggested_fix="shorten it"
    )
    assert v.field == "character_class"
    assert v.kind == "too_long"
    assert v.message == "too long"
    assert v.suggested_fix == "shorten it"


def test_violation_model_optional_fix():
    v = Violation(field="name", kind="name_has_period", message="contains period")
    assert v.suggested_fix is None


def test_normalize_output_clears_all_violations():
    # Codex review fix #2: normalize() must produce a value that PASSES validate()
    # (AgentRunner does not re-validate), including punctuation-only violations.
    v = ShortLabelValidator(field="character_class")
    for bad in (
        "Shield, Warrior",          # punctuation, within word/char limits
        "Warrior. Fighter",         # period
        "Stonecarver Who Chips Away Loose Debris From The Throne",  # too many words
        "A" * 80,                   # too long
    ):
        out = v.normalize({"character_class": bad})
        assert v.validate(out) == [], f"normalize() left a violation for {bad!r}: {out}"
        assert len(out["character_class"]) < 100  # prod VARCHAR(100) guard


def test_short_label_violation_kinds_are_stable():
    v = ShortLabelValidator(field="character_class")
    assert v.validate({"character_class": "one two three four five"})[0].kind == "too_many_words"
    assert v.validate({"character_class": "X" * 80})[0].kind == "too_long"
    assert v.validate({"character_class": "Shield, Warrior"})[0].kind == "has_punctuation"


# ---------------------------------------------------------------------------
# BoundedTextValidator tests (BOU-1752)
# ---------------------------------------------------------------------------


def test_bounded_text_flags_over_limit():
    """A value longer than max_chars is flagged exactly once as too_long."""
    val = BoundedTextValidator(field="title", max_chars=10)
    violations = val.validate({"title": "A" * 40})
    assert len(violations) == 1
    assert violations[0].field == "title"
    assert violations[0].kind == "too_long"


def test_bounded_text_allows_multi_word_within_limit():
    """Multi-word phrases and punctuation are fine if within the char bound
    (unlike ShortLabelValidator / ProperNameValidator)."""
    val = BoundedTextValidator(field="name", max_chars=255)
    out = {"name": "The Sunken Temple of Forgotten Gods, Vol. 2"}
    assert val.validate(out) == []


def test_bounded_text_normalize_caps_on_word_boundary():
    """normalize() truncates to <= max_chars, preferring a word boundary."""
    val = BoundedTextValidator(field="title", max_chars=20)
    out = val.normalize({"title": "Rescue the Kidnapped Duchess from the Tower"})
    result = out["title"]
    assert len(result) <= 20, f"expected <=20 chars, got {len(result)}: {result!r}"
    # Word boundary: should not end mid-word ("Kidnap")
    assert not result.endswith("Kidnap"), f"should cut on a word boundary: {result!r}"
    assert result.startswith("Rescue the")


def test_bounded_text_normalize_hard_cap_when_no_space():
    """A single over-long token with no space is hard-truncated to max_chars."""
    val = BoundedTextValidator(field="name", max_chars=10)
    out = val.normalize({"name": "X" * 40})
    assert len(out["name"]) == 10


def test_bounded_text_normalize_clears_violation():
    """normalize() output must PASS validate() (AgentRunner does not re-validate)."""
    val = BoundedTextValidator(field="title", max_chars=15)
    for bad in ("Y" * 80, "many short words strung together far past the cap"):
        out = val.normalize({"title": bad})
        assert val.validate(out) == [], f"normalize left a violation for {bad!r}: {out}"


def test_bounded_text_skips_none_and_non_str():
    val = BoundedTextValidator(field="title", max_chars=10)
    assert val.validate({"title": None}) == []
    assert val.validate({"title": 12345}) == []
    # normalize is a no-op for None
    assert val.normalize({"title": None})["title"] is None


def test_bounded_text_normalize_on_pydantic_model():
    from pydantic import BaseModel

    class FakeModel(BaseModel):
        title: str
        other: str = "preserved"

    val = BoundedTextValidator(field="title", max_chars=12)
    result = val.normalize(FakeModel(title="A very long campaign title indeed"))
    assert isinstance(result, FakeModel)
    assert len(result.title) <= 12
    assert result.other == "preserved"


# ---------------------------------------------------------------------------
# ForEachValidator tests (BOU-1752)
# ---------------------------------------------------------------------------


def test_for_each_validates_nested_list_with_index_qualified_field():
    """One over-limit sub-object in a list yields an index-qualified violation."""
    from pydantic import BaseModel

    class Act(BaseModel):
        title: str

    class Output(BaseModel):
        acts: list[Act]

    out = Output(acts=[Act(title="Short"), Act(title="Mid"), Act(title="Z" * 80)])
    val = ForEachValidator("acts", lambda o: o.acts, BoundedTextValidator("title", 20))
    violations = val.validate(out)
    assert len(violations) == 1
    assert violations[0].field == "acts[2].title"
    assert violations[0].kind == "too_long"


def test_for_each_normalize_truncates_only_offender_in_place():
    """normalize() truncates only the offending sub-object; siblings unchanged,
    and the change is reflected on the parent container."""
    from pydantic import BaseModel

    class Act(BaseModel):
        title: str

    class Output(BaseModel):
        acts: list[Act]

    out = Output(acts=[Act(title="Keep Me"), Act(title="Z" * 80)])
    val = ForEachValidator("acts", lambda o: o.acts, BoundedTextValidator("title", 20))
    result = val.normalize(out)
    assert result.acts[0].title == "Keep Me", "sibling must be unchanged"
    assert len(result.acts[1].title) <= 20, "offender must be truncated on the parent"
    # And the normalized output passes validate()
    assert val.validate(result) == []


def test_for_each_over_dict_of_models():
    """selector can flatten a dict-of-models (campaign location_registry shape)."""
    from pydantic import BaseModel

    class Loc(BaseModel):
        name: str

    class Output(BaseModel):
        location_registry: dict[str, Loc]

    out = Output(
        location_registry={
            "loc:a": Loc(name="Tavern"),
            "loc:b": Loc(name="N" * 300),
        }
    )
    val = ForEachValidator(
        "location_registry",
        lambda o: list(o.location_registry.values()),
        BoundedTextValidator("name", 255),
    )
    violations = val.validate(out)
    assert len(violations) == 1
    assert violations[0].field.endswith(".name")
    result = val.normalize(out)
    assert len(result.location_registry["loc:b"].name) <= 255
    assert result.location_registry["loc:a"].name == "Tavern"


def test_for_each_over_flattened_dict_of_lists_of_dicts():
    """selector can flatten residents (Dict[ref, List[resident_dict]]) and the
    inner validator reads a dict key — mutation is in place on the dicts."""
    residents = {
        "loc:a": [{"role": "Innkeeper"}, {"role": "Suspiciously Verbose Tavern Keeper Of Renown"}],
        "loc:b": [{"role": "Guard"}],
    }
    val = ForEachValidator(
        "residents",
        lambda o: [r for lst in o.values() for r in lst],
        ShortLabelValidator(field="role"),
    )
    violations = val.validate(residents)
    assert violations, "expected the over-long role to be flagged"
    val.normalize(residents)
    # The offending dict was mutated in place
    assert len(residents["loc:a"][1]["role"].split()) <= 3
    assert residents["loc:a"][0]["role"] == "Innkeeper"


def test_for_each_doubly_nested_acts_goals_flatten():
    """A flatten selector reaches acts[].goals[].title (the campaign quests.title
    source) — only the over-limit goal title is bounded, siblings untouched."""
    from pydantic import BaseModel

    class Goal(BaseModel):
        title: str

    class Act(BaseModel):
        goals: list[Goal]

    class Output(BaseModel):
        acts: list[Act]

    out = Output(
        acts=[
            Act(goals=[Goal(title="Find the key"), Goal(title="Q" * 400)]),
            Act(goals=[Goal(title="Escape")]),
        ]
    )
    val = ForEachValidator(
        "act_goals",
        lambda o: [g for act in o.acts for g in act.goals],
        BoundedTextValidator("title", 255),
    )
    violations = val.validate(out)
    assert len(violations) == 1
    assert violations[0].field == "act_goals[1].title"
    result = val.normalize(out)
    assert len(result.acts[0].goals[1].title) <= 255
    assert result.acts[0].goals[0].title == "Find the key"
    assert result.acts[1].goals[0].title == "Escape"


def test_for_each_selector_raising_is_safe():
    """A selector that raises (shape drift) → no violation, no raise."""
    def bad_selector(_o):
        raise KeyError("unexpected shape")

    val = ForEachValidator("x", bad_selector, BoundedTextValidator("f", 10))
    assert val.validate({"anything": 1}) == []
    # normalize returns output unchanged, no raise
    out = {"anything": 1}
    assert val.normalize(out) is out


def test_for_each_empty_and_none_selector_results():
    val = ForEachValidator("x", lambda o: o.get("items"), BoundedTextValidator("f", 10))
    assert val.validate({"items": None}) == []
    assert val.validate({"items": []}) == []
