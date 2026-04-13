"""Tree-sitter based symbol chunker.

Walks source files and extracts top-level functions, methods, and classes as
discrete chunks. Each chunk becomes its own L0 entry in the index, keyed by
a URI with a #symbol anchor (e.g. viking://src/auth.py#verify_token).

Supports: Python, JavaScript/TypeScript, Java, C++, Rust, Go, C#, PHP, Lua.
Falls back to whole-file chunking for unsupported or unparseable files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Mapping from file extension to (tree-sitter module name, language name)
_EXT_LANG: dict[str, tuple[str, str]] = {
    ".py": ("tree_sitter_python", "python"),
    ".js": ("tree_sitter_javascript", "javascript"),
    ".jsx": ("tree_sitter_javascript", "javascript"),
    ".ts": ("tree_sitter_typescript", "typescript"),
    ".tsx": ("tree_sitter_typescript", "tsx"),
    ".java": ("tree_sitter_java", "java"),
    ".cpp": ("tree_sitter_cpp", "cpp"),
    ".cc": ("tree_sitter_cpp", "cpp"),
    ".cxx": ("tree_sitter_cpp", "cpp"),
    ".rs": ("tree_sitter_rust", "rust"),
    ".go": ("tree_sitter_go", "go"),
    ".cs": ("tree_sitter_c_sharp", "c_sharp"),
    ".php": ("tree_sitter_php", "php"),
    ".lua": ("tree_sitter_lua", "lua"),
}

# Tree-sitter node types that represent a top-level symbol worth chunking.
# Keyed by language name.
_SYMBOL_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition",
                   "arrow_function", "lexical_declaration"},
    "typescript": {"function_declaration", "class_declaration", "method_definition",
                   "interface_declaration", "type_alias_declaration"},
    "tsx": {"function_declaration", "class_declaration", "method_definition"},
    "java": {"class_declaration", "method_declaration", "interface_declaration",
             "constructor_declaration", "enum_declaration"},
    "cpp": {"function_definition", "class_specifier"},
    "rust": {"function_item", "impl_item", "struct_item", "enum_item", "trait_item",
             "mod_item"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "c_sharp": {"class_declaration", "method_declaration", "interface_declaration",
                "struct_declaration", "enum_declaration", "constructor_declaration"},
    "php": {"function_definition", "class_declaration", "method_declaration"},
    "lua": {"function_definition", "local_function"},
}

# How many bytes to retain per symbol chunk (prevents oversized LLM context)
_MAX_CHUNK_BYTES = 8_000


@dataclass
class SymbolChunk:
    """A single symbol extracted from a source file."""
    rel_path: str          # e.g. "src/auth.py"
    anchor: str            # e.g. "verify_token" or "AuthClass.verify_token"
    start_line: int        # 1-based
    end_line: int          # 1-based inclusive
    source: str            # raw source text of the symbol
    uri: str               # full viking:// URI with anchor


def _get_language(ext: str):
    """Return a configured tree-sitter Language or None for unsupported extensions."""
    entry = _EXT_LANG.get(ext)
    if entry is None:
        return None, None
    module_name, lang_name = entry
    try:
        import importlib
        mod = importlib.import_module(module_name)
        from tree_sitter import Language
        # tree-sitter >= 0.23 uses Language(module.language())
        lang = Language(mod.language())
        return lang, lang_name
    except Exception:
        return None, None


def _extract_name(node, source_bytes: bytes) -> str:
    """Best-effort symbol name extraction from a tree-sitter node."""
    # Walk direct children for a name/identifier node
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier"):
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return f"line_{node.start_point[0] + 1}"


def chunk_file(rel_path: str, source_text: str) -> list[SymbolChunk]:
    """Chunk a source file into per-symbol chunks.

    Returns a list of SymbolChunk objects. Falls back to a single whole-file
    chunk if the language is unsupported or the file fails to parse.
    """
    ext = Path(rel_path).suffix.lower()
    lang, lang_name = _get_language(ext)

    if lang is None or not source_text.strip():
        # Unsupported — emit a single whole-file chunk with no anchor
        lines = source_text.splitlines()
        return [SymbolChunk(
            rel_path=rel_path,
            anchor="",
            start_line=1,
            end_line=len(lines),
            source=source_text[:_MAX_CHUNK_BYTES],
            uri=f"viking://{rel_path}",
        )]

    try:
        from tree_sitter import Parser
        parser = Parser(lang)
        source_bytes = source_text.encode("utf-8")
        tree = parser.parse(source_bytes)
    except Exception:
        lines = source_text.splitlines()
        return [SymbolChunk(
            rel_path=rel_path,
            anchor="",
            start_line=1,
            end_line=len(lines),
            source=source_text[:_MAX_CHUNK_BYTES],
            uri=f"viking://{rel_path}",
        )]

    symbol_types = _SYMBOL_NODE_TYPES.get(lang_name, set())
    source_bytes = source_text.encode("utf-8")
    chunks: list[SymbolChunk] = []
    seen_anchors: dict[str, int] = {}

    def _visit(node, parent_name: Optional[str] = None):
        if node.type in symbol_types:
            name = _extract_name(node, source_bytes)
            anchor = f"{parent_name}.{name}" if parent_name else name

            # Deduplicate anchors within the same file
            if anchor in seen_anchors:
                seen_anchors[anchor] += 1
                anchor = f"{anchor}_{seen_anchors[anchor]}"
            else:
                seen_anchors[anchor] = 0

            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            symbol_src = source_bytes[node.start_byte:node.end_byte].decode(
                "utf-8", errors="replace"
            )

            # Add line numbers as a prefix to help the summarizer reference them,
            # then cap the total to _MAX_CHUNK_BYTES.
            lines = symbol_src.splitlines()
            numbered = "\n".join(
                f"{start_line + i}: {line}" for i, line in enumerate(lines)
            )[:_MAX_CHUNK_BYTES]

            chunks.append(SymbolChunk(
                rel_path=rel_path,
                anchor=anchor,
                start_line=start_line,
                end_line=end_line,
                source=numbered,
                uri=f"viking://{rel_path}#{anchor}",
            ))
            # Recurse into children with this name as parent
            for child in node.children:
                _visit(child, parent_name=name)
        else:
            for child in node.children:
                _visit(child, parent_name=parent_name)

    _visit(tree.root_node)

    if not chunks:
        # File parsed but no recognised symbols — fall back to whole-file chunk
        lines = source_text.splitlines()
        return [SymbolChunk(
            rel_path=rel_path,
            anchor="",
            start_line=1,
            end_line=len(lines),
            source=source_text[:_MAX_CHUNK_BYTES],
            uri=f"viking://{rel_path}",
        )]

    return chunks
