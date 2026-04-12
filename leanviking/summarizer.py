import litellm

class Summarizer:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    async def generate_l0(self, content: str) -> str:
        """
        Takes raw file content and returns a one-sentence summary.
        """
        prompt = f"Summarize the following file content in exactly one sentence. Focus on the primary purpose of the code:\n\n{content}"

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    async def generate_l1(self, folder_name: str, child_l0s: list) -> str:
        """
        Takes a folder name and a list of (filename, summary) tuples,
        and returns a structural map of the directory.
        """
        children_str = "\n".join([f"- {name}: {summary}" for name, summary in child_l0s])
        prompt = (
            f"Generate a structural map/overview of the directory '{folder_name}' based on its contents:\n\n"
            f"{children_str}\n\n"
            f"Return a concise overview of the directory's purpose and structure."
        )

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
