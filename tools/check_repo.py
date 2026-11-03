"""Проверка репозитория перед публикацией на GitHub.

Считает файлы, которые реально попадут в git (с учётом .gitignore), и сравнивает
их с лимитами GitHub:

  * push через git  — 100 МБ на файл (жёсткий лимит), 50 МБ (предупреждение);
  * веб-загрузка    — 25 МБ на файл, 100 файлов за один drag-and-drop;
  * репозиторий     — рекомендуется < 1 ГБ;
  * Releases        — 2 ГБ на ассет (для крупных весов).

Запуск:  python tools/check_repo.py
Код возврата 1, если лимиты нарушены.
"""
from __future__ import annotations

import os
import subprocess
import sys

try:  # консоль Windows по умолчанию cp1251
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

LIMITS = {
    "file_push_mb": 100.0,     # жёсткий лимит git push
    "file_web_mb": 25.0,       # лимит веб-загрузки
    "files_web_batch": 100,    # файлов за один drag-and-drop
    "repo_mb": 1024.0,         # рекомендуемый размер репозитория
}


def tracked_files() -> list[str]:
    """Файлы, которые попадут в коммит (git add -A --dry-run)."""
    out = subprocess.run(
        ["git", "add", "-A", "--dry-run"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    ).stdout
    paths = []
    for line in out.splitlines():
        if line.startswith("add '"):
            paths.append(line[5:-1])
    return paths


def main() -> int:
    try:
        paths = tracked_files()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("не git-репозиторий: запустите из корня репозитория")
        return 1

    sizes = [(os.path.getsize(p), p) for p in paths if os.path.exists(p)]
    total = sum(s for s, _ in sizes)
    biggest = max(sizes, default=(0, "-"))
    over_push = [(s, p) for s, p in sizes if s > LIMITS["file_push_mb"] * 1e6]
    over_web = [(s, p) for s, p in sizes if s > LIMITS["file_web_mb"] * 1e6]

    print(f"файлов: {len(sizes)}")
    print(f"суммарно: {total / 1e6:.1f} МБ (лимит рекомендации: {LIMITS['repo_mb']:.0f} МБ)")
    print(f"самый большой: {biggest[0] / 1e6:.2f} МБ — {biggest[1]}")
    print(f"файлов > {LIMITS['file_push_mb']:.0f} МБ (push): {len(over_push)}")
    print(f"файлов > {LIMITS['file_web_mb']:.0f} МБ (веб): {len(over_web)}")
    batches = (len(sizes) + LIMITS["files_web_batch"] - 1) // LIMITS["files_web_batch"]
    print(f"партий при веб-загрузке по {LIMITS['files_web_batch']} файлов: {batches}")

    ok = True
    if over_push:
        ok = False
        print("\nОШИБКА: эти файлы не пройдут git push, вынесите их в Releases или LFS:")
        for s, p in over_push:
            print(f"  {s / 1e6:.1f} МБ  {p}")
    if total > LIMITS["repo_mb"] * 1e6:
        ok = False
        print("\nОШИБКА: репозиторий больше рекомендуемого размера")
    print("\nИТОГ:", "можно публиковать" if ok else "нужна чистка")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
