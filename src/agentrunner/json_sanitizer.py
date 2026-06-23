"""JSON sanitizer for cleaning malformed JSON from LLM outputs."""

import re
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Alternation that matches EITHER a complete JSON string literal (group 1, with
# \-escapes) OR a structural trailing comma before a closing }/] (group 2). The
# string branch consumes string contents so commas inside strings (e.g.
# "choose A,] then B") are matched-and-kept, never rewritten.
_TRAILING_COMMA_RE = re.compile(r'("(?:[^"\\]|\\.)*")|,(\s*[}\]])')


def _strip_structural_trailing_commas(json_str: str) -> str:
    """Remove trailing commas before a closing ``}``/``]`` — STRUCTURAL only.

    String-aware via ``_TRAILING_COMMA_RE``: a string literal is matched whole and
    returned unchanged, so a comma inside a string value is never touched; only a
    comma immediately preceding a closing brace/bracket (outside any string) is
    dropped.
    """
    return _TRAILING_COMMA_RE.sub(lambda m: m.group(1) if m.group(1) else m.group(2), json_str)


def sanitize_json_string(json_str: str) -> str:
    """Sanitize a JSON string by removing/escaping control characters.

    Args:
        json_str: Raw JSON string that may contain control characters

    Returns:
        Cleaned JSON string
    """
    if not json_str:
        return json_str

    # Strip a leading markdown code fence (``` or ```json) — several open-source
    # LLMs wrap structured output in a markdown block despite "no markdown"
    # instructions. The trailing fence is removed later by the brace-balance
    # truncation, so only the leading fence needs explicit handling.
    leading = json_str.lstrip()
    if leading.startswith("```"):
        newline_idx = leading.find("\n")
        if newline_idx != -1:
            json_str = leading[newline_idx + 1:]
        else:
            json_str = leading[3:]

    # First, try to identify if this is a broken JSON with embedded content
    # Pattern: JSON that ends prematurely with extra content after
    pattern = r'^(\{[^}]*"[^"]*"):.*?(Combat State:|combat_state).*?\}.*?(\{.*?\})$'
    match = re.search(pattern, json_str, re.DOTALL)
    if match:
        # Try to extract the main JSON and ignore the broken parts
        main_json = json_str[:json_str.rfind('}')+1]
        json_str = main_json

    # Remove any control characters (0x00-0x1F) except for \t, \n, \r which we'll escape
    # First, temporarily replace valid escaped sequences
    json_str = json_str.replace('\\n', '__ESCAPED_N__')
    json_str = json_str.replace('\\r', '__ESCAPED_R__')
    json_str = json_str.replace('\\t', '__ESCAPED_T__')
    json_str = json_str.replace('\\"', '__ESCAPED_QUOTE__')
    json_str = json_str.replace('\\\\', '__ESCAPED_BACKSLASH__')

    # Now handle actual control characters in string values
    # This regex finds strings and processes them
    def clean_string_value(match):
        string_content = match.group(1)
        # Replace actual newlines, tabs, etc. with escaped versions
        string_content = string_content.replace('\n', '\\n')
        string_content = string_content.replace('\r', '\\r')
        string_content = string_content.replace('\t', '\\t')
        # Remove other control characters
        string_content = ''.join(char for char in string_content if ord(char) >= 32 or char in ['\t'])
        return f'"{string_content}"'

    # Apply to all string values
    json_str = re.sub(r'"([^"]*)"', clean_string_value, json_str)

    # Restore the valid escaped sequences
    json_str = json_str.replace('__ESCAPED_N__', '\\n')
    json_str = json_str.replace('__ESCAPED_R__', '\\r')
    json_str = json_str.replace('__ESCAPED_T__', '\\t')
    json_str = json_str.replace('__ESCAPED_QUOTE__', '\\"')
    json_str = json_str.replace('__ESCAPED_BACKSLASH__', '\\\\')

    # Strip trailing commas before a closing brace/bracket — open-weight models
    # (notably Nemotron-120B) frequently emit `"k": v,}` or `[a, b,]`, which
    # strict json.loads rejects ("Expecting property name enclosed in double
    # quotes"). This is the single most common structured-output JSON defect and
    # is what the agents-SDK validate_json patch relies on this sanitiser to fix.
    # String-aware (NOT a blind regex): a comma inside a string value — e.g.
    # "choose A,] then B" — is left untouched; only structural trailing commas go.
    json_str = _strip_structural_trailing_commas(json_str)

    # Remove any trailing content after the last valid }
    # Find the last } that would close the JSON object
    brace_count = 0
    last_valid_pos = -1
    for i, char in enumerate(json_str):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                last_valid_pos = i
                break

    if last_valid_pos > -1:
        json_str = json_str[:last_valid_pos + 1]

    return json_str


def _drop_trailing_string_token(s: str) -> str:
    """Remove a complete trailing ``"..."`` token (string-aware, escape-aware).

    ``s`` is expected to end with the closing quote of a string. Scans backward to
    the matching *unescaped* opening quote and returns ``s`` without that token.
    Returns ``s`` unchanged if no opening quote is found.
    """
    if not s.endswith('"'):
        return s
    i = len(s) - 2
    while i >= 0:
        if s[i] == '"':
            backslashes = 0
            b = i - 1
            while b >= 0 and s[b] == "\\":
                backslashes += 1
                b -= 1
            if backslashes % 2 == 0:  # this quote is unescaped → string start
                return s[:i]
        i -= 1
    return s


def _trim_incomplete_object_tail(s: str, stack: list[str]) -> str:
    """Trim a dangling/incomplete trailing member so ``s`` is ready for closers.

    Handles the truncation-mid-member cases that a bare separator strip cannot:
    a key with no value (``{"a": 1, "b":``), or a partial/whole key with no colon
    (``{"a": 1, "b"``). The whole member is dropped — not just the trailing
    separator — so the remaining object parses once balanced. A trailing string in
    *value* position (preceded by ``:``) or any array element is kept.
    """
    while True:
        s = s.rstrip()
        if not s:
            return s
        last = s[-1]
        if last == ",":
            s = s[:-1]
            continue
        if last == ":":
            # Key with no value → drop the colon, then the key string itself.
            s = _drop_trailing_string_token(s[:-1].rstrip())
            continue
        if last == '"' and stack and stack[-1] == "{":
            # A complete string at the top of an object is a dangling key unless a
            # ':' precedes it (then it's a value we should keep).
            before = _drop_trailing_string_token(s).rstrip()
            if not before or before[-1] in ",{":
                s = before
                continue
        return s


def repair_truncated_json(json_str: str) -> str:
    """Best-effort repair of JSON truncated at EOF.

    Targets the BOU-1145 degeneration pattern: a model (notably Nemotron) emits a
    syntactically-valid JSON object prefix, then floods whitespace tokens
    (``"\\n \\n \\n ..."``) until it hits ``max_tokens`` and stops without ever
    closing the object. ``sanitize_json_string``'s brace-balance truncation cannot
    help here — it only trims content *after* the last balanced ``}``, and in this
    case the object never closes, so it returns the broken string unchanged.

    The repair is deliberately conservative: it only completes structure the model
    *finished* emitting, and never fabricates content the model did not produce.

    1. Strip the trailing whitespace flood.
    2. If the output was cut **inside a value string** (a string in value or array
       position is still open at EOF), refuse to repair — closing the quote would
       turn a half-written value into a complete-looking one and persist truncated
       data. Return the input unchanged so validation fails and the caller's
       retry/fallback path runs. A string cut in **key** position is instead dropped
       whole (keys carry no persisted value).
    3. Refuse when more than one container is still open, or when the lone open
       container was never filled (ends at its opener). Both mean the cut landed
       mid-content; balancing would fabricate missing inner items (a partial
       list/object) or an empty container.
    4. Otherwise drop a dangling/incomplete trailing member (a key with no value, or
       a key with no colon) and append the single ``]`` / ``}`` that closes the lone
       outermost container.

    Recovering *structure* stays decoupled from *accepting* the result: required
    schema fields are the gate. When the model emitted its required fields before the
    whitespace flood, the repaired JSON validates; when truncation dropped a required
    field, pydantic rejects the parseable-but-incomplete object and the existing
    retry/fallback takes over — the repair never lets a defaulted value silently
    stand in for content the model never produced (BOU-1145).

    Best-effort and total: never raises; returns the input unchanged when it cannot
    safely repair.
    """
    if not json_str:
        return json_str

    s = json_str.rstrip()
    if not s:
        return json_str

    stack: list[str] = []  # open '{' / '[' characters, in order
    in_string = False
    escaped = False
    string_open_idx = -1  # index of the opening quote of the currently-open string

    for i, ch in enumerate(s):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            string_open_idx = i
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()

    # Already structurally complete — nothing to repair.
    if not in_string and not stack:
        return s

    if in_string:
        # Cut mid-string. Only a string in *key* position (an object member whose
        # key is being typed) is safe to drop; a truncated *value* must not be
        # fabricated into a complete one.
        before = s[:string_open_idx].rstrip()
        prev = before[-1] if before else ""
        if stack and stack[-1] == "{" and prev in ",{":
            s = before  # partial key → drop it; fall through to trim + balance
        else:
            # Truncated value (object value after ':' or any array element) — refuse.
            return json_str

    # Only the whitespace-flood case is safe to complete: a single still-open
    # container (the outermost object/array) whose every nested container was already
    # closed. More than one unclosed container means the model was cut mid-nested
    # structure — balancing would fabricate the missing inner content (a partial
    # list/object diff), so refuse and let validation → retry handle it (BOU-1145).
    if len(stack) > 1:
        return json_str

    if not in_string and s.rstrip().endswith(","):
        return json_str

    # Drop a dangling/incomplete trailing member (whole key/separator, not just the
    # trailing punctuation).
    repaired = _trim_incomplete_object_tail(s, stack)

    # If the lone container was just opened (the text now ends at an opener), it was
    # never filled — closing it would fabricate an empty array/object. Refuse.
    trimmed = repaired.rstrip()
    if trimmed and trimmed[-1] in "{[":
        return json_str

    # Append the closing ]/} needed to balance the single still-open container.
    closers = {"{": "}", "[": "]"}
    for opener in reversed(stack):
        repaired += closers[opener]

    # Drop any structural trailing comma the truncation may have left before a closer.
    return _strip_structural_trailing_commas(repaired)


def parse_json_safely(json_str: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Parse JSON string safely with sanitization and fallback.

    Args:
        json_str: JSON string to parse
        fallback: Fallback dictionary to return if parsing fails

    Returns:
        Parsed JSON as dictionary, or fallback if parsing fails
    """
    if not json_str:
        return fallback or {}

    try:
        # First attempt: parse as-is
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.debug(f"Initial JSON parse failed: {e}")

        try:
            # Second attempt: sanitize and parse
            sanitized = sanitize_json_string(json_str)
            return json.loads(sanitized)
        except json.JSONDecodeError as e2:
            repaired: Optional[str] = None
            try:
                # Third attempt: repair EOF-truncated JSON (whitespace flood, an
                # unterminated string, or unbalanced braces) then sanitize + parse.
                repaired = sanitize_json_string(repair_truncated_json(json_str))
                return json.loads(repaired)
            except Exception:
                # Broad on purpose: a transform raising a non-JSONDecodeError must
                # not skip the fallback path below.
                logger.exception('JSON parse failed even after sanitization and repair')
                logger.debug(f"Original JSON: {json_str[:500]}...")
                logger.debug(f"Sanitized JSON: {sanitized[:500]}...")
                if repaired is not None:
                    logger.debug(f"Repaired JSON: {repaired[:500]}...")

                if fallback:
                    logger.info("Using fallback response")
                    return fallback
                raise e2


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON object from text that may contain additional content.

    Args:
        text: Text that may contain JSON along with other content

    Returns:
        Extracted JSON string or None
    """
    # Look for JSON object boundaries
    start_idx = text.find('{')
    if start_idx == -1:
        return None

    # Find matching closing brace
    brace_count = 0
    end_idx = -1

    for i in range(start_idx, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break

    if end_idx > start_idx:
        return text[start_idx:end_idx + 1]

    return None
