"""Optional LLM mode for the Go-to-Market Studio (Anthropic Claude API).

Off by default. It is used only when PHYSICSVETO_MOCK=0, the `anthropic` package is installed and
ANTHROPIC_API_KEY is set. Otherwise the app uses the template text in gtm/content.py.

The model rewrites the template brief / talk track for one vertical and audience. It is given the
measured numbers as a JSON block and told to use only those numbers, so it cannot invent results.
Everything it returns is labelled "LLM-drafted" in the app and should be reviewed before use.
Server-side model fallbacks are enabled (fallbacks="default"), so if the primary model is
unavailable the API may answer with a fallback model; the served model is reported back.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

MODEL = os.environ.get("PHYSICSVETO_LLM_MODEL", "claude-opus-5-5")

SYSTEM = (
    "You write go-to-market material for PhysicsVeto, a demo workbench that pairs fast AI physics surrogates "
    "with a physics-based veto (PDE residual, boundary, conservation, out-of-distribution and ensemble checks). "
    "Rules: use only numbers that appear in the MEASURED_RESULTS block, quoted exactly, with their hardware label; "
    "never invent customers, benchmarks, partnerships or endorsements; do not use any company's brand as if "
    "PhysicsVeto were its product; say plainly that the demo runs on CPU at 64x64 when you cite speed. "
    "Write in plain, direct English for the named audience. Output Markdown only."
)

TASKS = {
    "solution_brief": "Rewrite this one-page solution brief for the audience below. Keep it to one page.",
    "talk_track": "Rewrite this 5-minute demo talk track for the audience below. Keep the timed sections.",
    "outreach_email": "Write a short first-contact email (under 150 words) for the audience below, based on "
                      "the brief. No subject-line hype, one clear ask.",
}


@dataclass
class LLMResult:
    text: str
    model: str
    note: str = ""


def available() -> tuple[bool, str]:
    """Whether LLM mode can run, and why not if it cannot."""
    if os.environ.get("PHYSICSVETO_MOCK", "1") == "1":
        return False, "Mock mode is on (PHYSICSVETO_MOCK=1)."
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, "The `anthropic` package is not installed (pip install anthropic)."
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False, "ANTHROPIC_API_KEY is not set."
    return True, ""


def build_prompt(task: str, vertical: str, audience: str, template_text: str, measured: dict) -> str:
    return (
        f"{TASKS[task]}\n\nVertical: {vertical}\nAudience: {audience}\n\n"
        f"<MEASURED_RESULTS>\n{json.dumps(measured, indent=2)}\n</MEASURED_RESULTS>\n\n"
        f"<TEMPLATE>\n{template_text}\n</TEMPLATE>"
    )


def generate(task: str, vertical: str, audience: str, template_text: str, measured: dict,
             client=None, max_tokens: int = 4000) -> LLMResult:
    """Call Claude. Raises RuntimeError with a readable message on any failure."""
    import anthropic

    client = client or anthropic.Anthropic()
    try:
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=SYSTEM,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": build_prompt(task, vertical, audience, template_text, measured)}],
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.RateLimitError as e:
        raise RuntimeError("Rate limited by the API; try again in a minute.") from e
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise RuntimeError("Could not reach the Anthropic API.") from e
    if resp.stop_reason == "refusal":
        raise RuntimeError("The model declined this request; the template text is shown instead.")
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if not text:
        raise RuntimeError("The model returned no text.")
    note = "Output was cut at the token limit." if resp.stop_reason == "max_tokens" else ""
    return LLMResult(text=text, model=getattr(resp, "model", MODEL), note=note)
