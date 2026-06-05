"""
safety.py — adversarial input handling for QueryLens.

Pattern-based screening for known prompt-injection and system-probing
attempts. Lives separately from decision.py so the safety policy can
evolve independently — new patterns get added here without touching
the decision logic.

Two functions:
  - detect_injection(text) — scans for known injection patterns
  - is_safe(text)          — convenience boolean wrapper

Wired into DecisionLayer.validate_query() so a flagged query produces
a rejected SearchDecision with a generic "I can't help with that"
warning. The same code path as length-validation rejections — Layer 3
doesn't need to know there's a separate safety subsystem.

Coverage: this catches the most common direct injection patterns. It
does NOT defend against:
  - Indirect prompt injection via malicious passages (defence: prompts.py
    explicitly tells the LLM to ignore passage instructions).
  - Multi-turn social engineering (no defence here — n/a for stateless
    QueryLens).
  - Novel attacks not in the pattern list (defence: update this list as
    new patterns are observed in telemetry).
"""

import re
from typing import List, Optional


# Each pattern is a compiled case-insensitive regex. Add new ones at the
# bottom — order doesn't matter for detection, only for the first-match
# report in detect_injection().
INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        # Direct instruction-override attempts.
        # The (modifier\s+)+ pattern catches chained modifiers like
        # "ignore all previous instructions" or "ignore the above prior rules".
        r"ignore\s+(the\s+)?((previous|prior|above|all|earlier)\s+)+(instructions?|prompts?|rules?)",
        r"disregard\s+(the\s+|all\s+)?(instructions?|prompts?|rules?|above)",
        r"forget\s+(everything|all\s+(previous|prior)|your\s+(instructions?|rules?))",
        r"override\s+(the\s+)?(system|instructions?|rules?)",

        # Role-swap / jailbreak
        r"you\s+are\s+(now\s+)?(a\s+|an\s+)?(?!helpful)",   # crude but catches "you are now DAN"
        r"(act|behave|respond)\s+as\s+(if\s+you\s+were\s+)?(a\s+|an\s+)?(?!a\s+helpful)",
        r"pretend\s+(to\s+be|you\s+are|that\s+you)",
        r"roleplay\s+as",

        # System-prompt extraction
        r"(repeat|print|show|reveal|tell\s+me)\s+(your\s+)?(system|initial)\s+(prompt|instructions?|message)",
        r"what\s+(are\s+)?(your\s+)?(system|initial|original)\s+(prompt|instructions?)",
        r"</?\s*(system|instructions?|prompt|user|assistant)\s*/?>",

        # System-state probing
        r"(list|show|dump|reveal)\s+(all\s+)?(your\s+|the\s+)?(files?|folders?|directories|env|environment\s+variables|secrets?|credentials?|api\s+keys?)",
        r"new\s+(instructions?|system\s+prompt|rules?)\s*[:=]",
        r"system\s*[:>=]",
    ]
]


def detect_injection(text: str) -> Optional[str]:
    """
    Scan text for known injection patterns.

    Returns the pattern string of the first match, or None if no pattern
    fires. Returning the pattern (not just a bool) makes telemetry useful —
    we can see which categories of attack are actually being attempted.

    Args:
        text: User query, retrieved passage, or any other untrusted text.

    Returns:
        First matching pattern as a string, or None.
    """
    if not text:
        return None
    for pat in INJECTION_PATTERNS:
        if pat.search(text):
            return pat.pattern
    return None


def is_safe(text: str) -> bool:
    """Convenience wrapper. True if no injection pattern matched."""
    return detect_injection(text) is None
