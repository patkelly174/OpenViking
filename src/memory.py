import json
import os
from pathlib import Path

import litellm


class Memory:
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.memories_dir = self.project_root / ".ov_brain" / "memories"
        self.memories_dir.mkdir(parents=True, exist_ok=True)

    async def extract_truths(
        self, transcript: str, model: str = "gpt-5-nano"
    ) -> list[dict]:
        prompt = (
            "Extract key truths, decisions, and preferences from this conversation "
            "transcript. Return ONLY a valid JSON object with a 'truths' key containing "
            "an array of objects, each with:\n"
            "  - 'key': a short slug (alphanumeric and hyphens only, e.g. 'auth-approach')\n"
            "  - 'content': the truth as a single sentence\n\n"
            f"Transcript:\n{transcript}"
        )
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences that some LLMs add
        if raw.startswith("```"):
            lines = raw.split("\n")
            # Remove first line (```json or ```) and last line (```)
            raw = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned non-JSON response: {raw!r}") from exc
        truths = data.get("truths", [])

        for truth in truths:
            key = truth.get("key", "").strip()
            content = truth.get("content", "").strip()
            if key and content:
                self._upsert(key, content)

        return truths

    def _upsert(self, key: str, content: str):
        safe_key = "".join(
            c if c.isalnum() or c == "-" else "_" for c in key
        )
        (self.memories_dir / f"{safe_key}.md").write_text(
            f"# {key}\n\n{content}\n"
        )

    def list_truths(self) -> list[dict]:
        return [
            {"key": f.stem, "content": f.read_text()}
            for f in sorted(self.memories_dir.glob("*.md"))
        ]
