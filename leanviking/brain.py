import os

class Brain:
    def __init__(self, project_root=None):
        self.project_root = project_root or os.getcwd()
        self.brain_dir = os.path.join(self.project_root, ".ov_brain")

    def resolve_uri(self, uri: str) -> str:
        if uri.startswith("viking://"):
            path = uri[len("viking://"):]
            return os.path.join(self.project_root, path)
        return uri
