"""Output validators for agent structured outputs.

Provides a Protocol for post-generation constraint checking, plus two built-in
validators used by the pilot NPC-generation paths (BOU-1743 / self-correction).

Usage in AgentRunner.run():
    output_validators=[ShortLabelValidator(field="character_class")]

Correction loop behaviour (implemented in agent_runner.py):
    1. validate() is called on each validator; violations are collected.
    2. If any violations: agent is re-prompted once with a structured correction
       message embedding the violation list.
    3. If still violating after max_corrections: normalize() is applied to each
       validator in sequence to produce a safe deterministic fallback.

All validators are pure / stateless / no-LLM.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Violation model
# ---------------------------------------------------------------------------


class Violation(BaseModel):
    """A single constraint violation emitted by an OutputValidator.

    `kind` is a STABLE, value-independent category (e.g. "too_many_words") used
    in the prompt-gap event signature so dedup groups recurring failures of the
    same shape — unlike `message`, which embeds the offending value/count.
    """

    field: str
    kind: str
    message: str
    suggested_fix: Optional[str] = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class OutputValidator(Protocol):
    """Contract for a post-generation constraint validator.

    name:       stable identifier used in the prompt-gap event signature.
    validate(): returns a (possibly empty) list of Violation objects.
    normalize(): returns a *corrected* copy/object — deterministic, no LLM.
                 MUST NOT raise; must return a valid object even if the field
                 value is completely missing or None.
    """

    name: str

    def validate(self, output: Any) -> list[Violation]: ...

    def normalize(self, output: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_field(output: Any, field: str) -> Any:
    """Read a field from a Pydantic model, dict, or plain object."""
    if isinstance(output, dict):
        return output.get(field)
    return getattr(output, field, None)


def _set_field(output: Any, field: str, value: Any) -> Any:
    """Return a copy/mutated version of output with field set to value.

    Handles:
    - Pydantic v2 models (model_copy)
    - Pydantic v1 models (copy)
    - dicts
    - plain objects (setattr in-place, returns same object)
    """
    if isinstance(output, dict):
        result = dict(output)
        result[field] = value
        return result
    # Pydantic v2
    if hasattr(output, "model_copy"):
        return output.model_copy(update={field: value})
    # Pydantic v1
    if hasattr(output, "copy"):
        return output.copy(update={field: value})
    # Plain object — mutate in place
    setattr(output, field, value)
    return output


# ---------------------------------------------------------------------------
# ShortLabelValidator
# ---------------------------------------------------------------------------


class ShortLabelValidator:
    """Validates that a string field is a short label (not a sentence).

    Flags if:
    - word count > max_words, OR
    - character length > max_chars, OR
    - value contains '.' or ',' (sentence-level punctuation)

    normalize() takes the first max_words whitespace tokens and hard-truncates to
    max_chars characters.

    IMPORTANT: max_chars MUST remain < 100 to protect character_profiles.character_class
    VARCHAR(100). The default of 40 is well within that bound.
    """

    def __init__(
        self,
        field: str,
        max_words: int = 3,
        max_chars: int = 40,
    ) -> None:
        self.field = field
        self.max_words = max_words
        self.max_chars = max_chars
        self.name = f"short_label:{field}"

    def validate(self, output: Any) -> list[Violation]:
        value = _get_field(output, self.field)
        if value is None or not isinstance(value, str):
            return []
        words = value.split()
        violations: list[Violation] = []
        if len(words) > self.max_words:
            violations.append(
                Violation(
                    field=self.field,
                    kind="too_many_words",
                    message=(
                        f"must be ≤{self.max_words} words, was {len(words)}: {value!r}"
                    ),
                    suggested_fix=(
                        f"Use a short label like '{' '.join(words[:self.max_words])}'"
                    ),
                )
            )
        elif len(value) > self.max_chars:
            violations.append(
                Violation(
                    field=self.field,
                    kind="too_long",
                    message=(
                        f"must be ≤{self.max_chars} characters, was {len(value)}: {value!r}"
                    ),
                    suggested_fix=f"Shorten to {self.max_chars} chars or fewer.",
                )
            )
        elif "." in value or "," in value:
            violations.append(
                Violation(
                    field=self.field,
                    kind="has_punctuation",
                    message=(
                        f"must not contain sentence punctuation: {value!r}"
                    ),
                    suggested_fix="Remove periods and commas; use a bare label.",
                )
            )
        return violations

    def normalize(self, output: Any) -> Any:
        value = _get_field(output, self.field)
        if value is None or not isinstance(value, str):
            return output
        # Strip sentence punctuation FIRST so the result also clears the
        # has_punctuation check (validate() flags '.'/',' — a split+truncate
        # alone would leave "Warrior. Fighter" still failing, and AgentRunner
        # does not re-validate after normalize()).
        cleaned = value.replace(".", " ").replace(",", " ")
        # Take first max_words tokens, then hard-truncate to max_chars to
        # guarantee the length bound regardless of token sizes.
        tokens = cleaned.split()
        clipped = " ".join(tokens[: self.max_words])[: self.max_chars]
        return _set_field(output, self.field, clipped)


# ---------------------------------------------------------------------------
# ProperNameValidator
# ---------------------------------------------------------------------------


class ProperNameValidator:
    """Validates that a name field looks like a proper name (not a sentence).

    Flags if:
    - word count > max_words, OR
    - value contains '.'

    normalize() takes the first max_words tokens.
    """

    def __init__(
        self,
        field: str = "name",
        max_words: int = 4,
    ) -> None:
        self.field = field
        self.max_words = max_words
        self.name = f"proper_name:{field}"

    def validate(self, output: Any) -> list[Violation]:
        value = _get_field(output, self.field)
        if value is None or not isinstance(value, str):
            return []
        words = value.split()
        violations: list[Violation] = []
        if len(words) > self.max_words:
            violations.append(
                Violation(
                    field=self.field,
                    kind="name_too_many_words",
                    message=(
                        f"name must be ≤{self.max_words} words, was {len(words)}: {value!r}"
                    ),
                    suggested_fix=(
                        f"Use a short name like '{' '.join(words[:self.max_words])}'"
                    ),
                )
            )
        elif "." in value:
            violations.append(
                Violation(
                    field=self.field,
                    kind="name_has_period",
                    message=f"name must not contain periods: {value!r}",
                    suggested_fix="Remove periods from the name.",
                )
            )
        return violations

    def normalize(self, output: Any) -> Any:
        value = _get_field(output, self.field)
        if value is None or not isinstance(value, str):
            return output
        # Drop periods (validate() flags them) before clipping to max_words.
        tokens = value.replace(".", " ").split()
        clipped = " ".join(tokens[: self.max_words])
        return _set_field(output, self.field, clipped)


# ---------------------------------------------------------------------------
# BoundedTextValidator
# ---------------------------------------------------------------------------


class BoundedTextValidator:
    """Validates that a string field fits within a character bound.

    Unlike ShortLabelValidator / ProperNameValidator this imposes NO word cap and
    does NOT flag punctuation — names and titles can legitimately be multi-word
    phrases ("The Sunken Temple of Forgotten Gods", "Half-Elf"). Its sole job is
    to keep an LLM-authored free-text value inside a length-bounded DB column
    (e.g. locations.name / quests.title VARCHAR(255), character_profiles.race
    VARCHAR(100)).

    normalize() truncates to max_chars, preferring the last word boundary within
    the cap, and always guarantees len(result) <= max_chars.
    """

    def __init__(self, field: str, max_chars: int) -> None:
        self.field = field
        self.max_chars = max_chars
        self.name = f"bounded_text:{field}"

    def validate(self, output: Any) -> list[Violation]:
        value = _get_field(output, self.field)
        if value is None or not isinstance(value, str):
            return []
        if len(value) > self.max_chars:
            return [
                Violation(
                    field=self.field,
                    kind="too_long",
                    message=(
                        f"must be ≤{self.max_chars} characters, was {len(value)}: {value!r}"
                    ),
                    suggested_fix=f"Shorten to {self.max_chars} characters or fewer.",
                )
            ]
        return []

    def normalize(self, output: Any) -> Any:
        value = _get_field(output, self.field)
        if value is None or not isinstance(value, str):
            return output
        if len(value) <= self.max_chars:
            return _set_field(output, self.field, value)
        capped = value[: self.max_chars]
        # Prefer a word boundary: drop the trailing partial word if there's a
        # space to cut at (and the cut leaves a non-empty result).
        if " " in capped:
            head = capped.rsplit(" ", 1)[0].rstrip()
            if head:
                capped = head
        # Hard guarantee regardless of the above.
        return _set_field(output, self.field, capped[: self.max_chars])


# ---------------------------------------------------------------------------
# ForEachValidator
# ---------------------------------------------------------------------------


class ForEachValidator:
    """Combinator that applies a flat inner validator to every sub-object yielded
    by a selector, so the framework can reach nested collection fields.

    The existing validators read a single top-level field. Generators like
    campaign_generator (location_registry: Dict[str, LocationOutput], acts: list)
    and location_residents (Dict[ref, List[resident_dict]]) carry the bounded
    field on nested elements. ForEachValidator bridges that gap:

    - ``selector(output) -> list`` returns the MUTABLE sub-objects to validate
      (Pydantic v2 models are mutable; resident rows are dicts). The selector owns
      the container shape (``list(d.values())``, a list attribute, a flattened
      dict-of-lists, etc.).
    - ``validate`` runs ``inner.validate`` on each sub-object and re-emits every
      violation with an index-qualified field (``"<label>[<i>].<field>"``) so the
      prompt-gap signature still dedups by the inner ``kind``.
    - ``normalize`` runs ``inner.normalize`` on each sub-object and writes the
      corrected value back IN PLACE so the parent container reflects it (the
      selector yields references inside ``output``).

    A selector that raises on shape drift is caught: validate() returns [] and
    normalize() returns output unchanged. Never raises.
    """

    def __init__(self, name: str, selector: Any, inner: Any) -> None:
        self._label = name
        self.selector = selector
        self.inner = inner
        self.name = f"for_each:{name}:{getattr(inner, 'name', 'inner')}"

    def _items(self, output: Any) -> list:
        try:
            items = self.selector(output)
        except Exception:  # noqa: BLE001 — shape drift must not break generation
            return []
        if items is None:
            return []
        try:
            return list(items)
        except Exception:  # noqa: BLE001
            return []

    def validate(self, output: Any) -> list[Violation]:
        violations: list[Violation] = []
        for i, sub in enumerate(self._items(output)):
            for v in self.inner.validate(sub):
                violations.append(
                    Violation(
                        field=f"{self._label}[{i}].{v.field}",
                        kind=v.kind,
                        message=v.message,
                        suggested_fix=v.suggested_fix,
                    )
                )
        return violations

    def normalize(self, output: Any) -> Any:
        inner_field = getattr(self.inner, "field", None)
        for sub in self._items(output):
            corrected = self.inner.normalize(sub)
            # inner.normalize mutates dicts/plain objects in place but returns a
            # NEW object for Pydantic models (model_copy). Write the corrected
            # field back onto the ORIGINAL sub-object so the parent container —
            # which still holds the original reference — reflects the change.
            if corrected is not sub and inner_field is not None:
                new_value = _get_field(corrected, inner_field)
                if isinstance(sub, dict):
                    sub[inner_field] = new_value
                else:
                    try:
                        setattr(sub, inner_field, new_value)
                    except Exception:  # noqa: BLE001 — frozen/immutable: best-effort
                        pass
        return output
