from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from .util import relative_files


ATTRIBUTE_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"((?:\\.|[^"\\])*)"')
TRANSLATABLE_ATTRIBUTES = {
    "name",
    "text",
    "title",
    "content",
    "subtitle",
    "options",
    "option",
    "choice",
}


@dataclass(frozen=True)
class StoryLine:
    skeleton: str
    attributes: dict[str, str]
    tail: str


def _unescape_story_value(value: str) -> str:
    return value.replace(r'\"', '"').replace(r"\\", "\\")


def parse_story_line(line: str) -> StoryLine:
    stripped = line.strip()
    if not stripped:
        return StoryLine("empty", {}, "")
    if not stripped.startswith("[") or "]" not in stripped:
        return StoryLine("plain|tail", {}, stripped)

    closing = stripped.find("]")
    inner = stripped[1:closing]
    tail = stripped[closing + 1 :].strip()
    command = re.split(r"[\s(=]", inner, maxsplit=1)[0].casefold()
    attributes = {
        name.casefold(): _unescape_story_value(value)
        for name, value in ATTRIBUTE_RE.findall(inner)
    }
    skeleton = "|".join(
        (
            command,
            ",".join(sorted(attributes)),
            "tail" if tail else "",
        )
    )
    return StoryLine(skeleton, attributes, tail)


def _emit_segment_pairs(
    source: StoryLine,
    target: StoryLine,
    add: Callable[[str, str, str], None],
    provenance: str,
) -> int:
    count = 0
    if source.tail and target.tail:
        add(source.tail, target.tail, provenance + ":tail")
        count += 1
    for name in source.attributes.keys() & target.attributes.keys() & TRANSLATABLE_ATTRIBUTES:
        source_value = source.attributes[name]
        target_value = target.attributes[name]
        if name in {"options", "option", "choice"}:
            source_parts = source_value.split(";")
            target_parts = target_value.split(";")
            if len(source_parts) == len(target_parts):
                for source_part, target_part in zip(source_parts, target_parts):
                    add(source_part.strip(), target_part.strip(), provenance + f":{name}")
                    count += 1
        else:
            add(source_value, target_value, provenance + f":{name}")
            count += 1
    return count


def _translation_skeleton(line: StoryLine) -> tuple[str, tuple[str, ...], bool] | None:
    """Describe only the line structure that contributes visible text."""
    attributes = tuple(
        sorted(name for name in line.attributes if name in TRANSLATABLE_ATTRIBUTES)
    )
    # Comments stored after non-text commands do not appear in the game and
    # should not block alignment of the visible text lines.
    visible_tail = bool(line.tail and not line.tail.startswith("//"))
    if not visible_tail and not attributes:
        return None
    command = line.skeleton.split("|", 1)[0]
    return command, attributes, visible_tail


def collect_story_pairs(
    source_root: Path,
    target_root: Path,
    add: Callable[[str, str, str], None],
    *,
    max_files: int | None = None,
    allow_fuzzy: bool = False,
    add_nontext_aligned: Callable[[str, str, str], None] | None = None,
) -> dict[str, int]:
    source_files = relative_files(source_root, "*.txt")
    target_files = relative_files(target_root, "*.txt")
    common = sorted(source_files.keys() & target_files.keys())
    common = [
        key
        for key in common
        if "report" not in Path(key).name.casefold()
        and not key.startswith("[uc]info/")
    ]
    if max_files is not None:
        common = common[:max_files]

    stats = {
        "source_files": len(source_files),
        "target_files": len(target_files),
        "common_files": len(common),
        "exact_structure_files": 0,
        "nontext_structure_files": 0,
        "fuzzy_structure_files": 0,
        "skipped_structure_mismatch_files": 0,
        "matched_lines": 0,
        "emitted_segments": 0,
    }
    for file_index, key in enumerate(common, start=1):
        if file_index == 1 or file_index % 100 == 0 or file_index == len(common):
            print(f"[story] file {file_index}/{len(common)}: {key}", flush=True)
        source_lines = [
            parse_story_line(line)
            for line in source_files[key].read_text(encoding="utf-8-sig", errors="replace").splitlines()
        ]
        target_lines = [
            parse_story_line(line)
            for line in target_files[key].read_text(encoding="utf-8-sig", errors="replace").splitlines()
        ]
        source_skeleton = [line.skeleton for line in source_lines]
        target_skeleton = [line.skeleton for line in target_lines]
        aligned_pairs: list[tuple[int, int]] | None = None
        use_nontext_callback = False
        if source_skeleton == target_skeleton:
            stats["exact_structure_files"] += 1
            aligned_pairs = [(index, index) for index in range(len(source_lines))]
        else:
            source_text = [
                (index, _translation_skeleton(line))
                for index, line in enumerate(source_lines)
            ]
            target_text = [
                (index, _translation_skeleton(line))
                for index, line in enumerate(target_lines)
            ]
            source_text = [(index, signature) for index, signature in source_text if signature]
            target_text = [(index, signature) for index, signature in target_text if signature]
            if [signature for _, signature in source_text] == [
                signature for _, signature in target_text
            ] and source_text and add_nontext_aligned is not None:
                stats["nontext_structure_files"] += 1
                aligned_pairs = [
                    (source_index, target_index)
                    for (source_index, _), (target_index, _) in zip(source_text, target_text)
                ]
                use_nontext_callback = True
        if aligned_pairs is None and allow_fuzzy:
            stats["fuzzy_structure_files"] += 1
            blocks = SequenceMatcher(
                None,
                source_skeleton,
                target_skeleton,
                autojunk=False,
            ).get_matching_blocks()
            aligned_pairs = [
                (block.a + offset, block.b + offset)
                for block in blocks
                for offset in range(block.size)
            ]
        elif aligned_pairs is None:
            stats["skipped_structure_mismatch_files"] += 1
            continue
        for source_index, target_index in aligned_pairs:
            source_line = source_lines[source_index]
            target_line = target_lines[target_index]
            stats["matched_lines"] += 1
            callback = (
                add_nontext_aligned
                if use_nontext_callback and add_nontext_aligned is not None
                else add
            )
            stats["emitted_segments"] += _emit_segment_pairs(
                source_line,
                target_line,
                callback,
                f"story:{key}:{source_index + 1}",
            )
    return stats
