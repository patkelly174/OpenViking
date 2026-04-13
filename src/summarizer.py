import json
import litellm
from dataclasses import dataclass


@dataclass
class L0Summary:
    """Dual-purpose summary for a single file or symbol."""
    search_text: str   # dense keyword blurb — embedded for retrieval
    display_text: str  # structured prose — returned to the agent


class Summarizer:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    async def generate_l0(self, content: str, rel_path: str = "") -> L0Summary:
        """
        Generate a dual-purpose summary for a file or symbol chunk.

        search_text: keyword-dense, synonym-rich — optimised for embedding similarity.
        display_text: structured, line-referenced, actionable — shown to agents.
        """
        if not content or content.strip() == "":
            return L0Summary(
                search_text="empty non-functional file no code",
                display_text="Empty or non-functional file.",
            )

        prompt = (
            "Analyse the following source file and produce a JSON object with two keys:\n\n"
            '- "search_text": A dense, keyword-rich phrase (≤30 words) optimised for semantic '
            "search. Include synonyms, concept names, and related terms an engineer might use "
            "when searching for this code. Do NOT use full sentences.\n"
            '- "display_text": A structured summary (≤120 words) for an AI agent. Include: '
            "primary purpose, key exports/functions/classes with their `path:line` references, "
            "notable side effects, and any important dependencies. Use bullet points.\n\n"
            "Respond with ONLY valid JSON — no markdown fences, no extra text.\n\n"
            f"File: {rel_path}\n"
            f"Content:\n{content}"
        )

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        raw = (response.choices[0].message.content or "").strip()  # type: ignore[union-attr]

        # Strip markdown fences if the model adds them despite instructions
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            parsed = json.loads(raw)
            return L0Summary(
                search_text=str(parsed.get("search_text", "")),
                display_text=str(parsed.get("display_text", "")),
            )
        except (json.JSONDecodeError, KeyError):
            # Fallback: treat the whole response as display_text
            return L0Summary(search_text=raw[:200], display_text=raw)

    async def generate_l1(self, folder_name: str, child_l0s: list[tuple[str, str]]) -> str:
        """
        Synthesise a directory-level overview from child display_text summaries.

        Returns a structured map useful as orientation context for an agent.
        """
        children_str = "\n".join(
            f"- {name}: {summary}" for name, summary in child_l0s
        )
        prompt = (
            f"Generate a structural overview of the directory '{folder_name}' "
            "for an AI coding agent. Use the child summaries below.\n\n"
            f"{children_str}\n\n"
            "Your response must include:\n"
            "1. 2-3 sentences on the directory's overall purpose and role in the project.\n"
            "2. The primary entry points and public interfaces exposed by this directory.\n"
            "3. The key data flow through this directory (inputs → transforms → outputs).\n"
            "4. A bullet list of key files/symbols with their full relative paths and "
            "`path:line` markers from the child summaries where available.\n\n"
            "Be specific and actionable. An agent reading this should know exactly which "
            "file to open next without further exploration."
        )

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        return (response.choices[0].message.content or "").strip()  # type: ignore[union-attr]