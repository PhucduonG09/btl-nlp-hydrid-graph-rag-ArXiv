"""Kiểm tra an toàn TRƯỚC khi git add/commit/push. Offline, chỉ đọc.

Chạy:  python -X utf8 scripts/check_before_push.py
Thoát mã 1 nếu: .env có thể bị đưa lên; tìm thấy chuỗi giống API key/token trong file sắp push;
file >= 95 MB (GitHub chặn 100 MB). Không in giá trị key, chỉ in tên file và dòng.
"""
import re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 95 * 1024 * 1024
SCAN_MAX = 40 * 1024 * 1024
PATTERNS = {
    "sk-key": re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{32,}"),
    "github-token": re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{30,}"),
    "hf-token": re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}"),
    "google-key": re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{30,}"),
    "bearer": re.compile(r"Bearer\s+[A-Za-z0-9._-]{30,}"),
}
TEXT_EXT = {".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".yaml", ".yml", ".ipynb", ".cmd", ".ps1",
            ".toml", ".cfg", ".ini", ".html", ".tex", ".graphml", ".example", ""}


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout


def main():
    # tracked + untracked-not-ignored = đúng tập sẽ được `git add -A`
    files = [f for f in git("ls-files", "-c", "-o", "--exclude-standard").splitlines() if f.strip()]
    bad = []
    if any(Path(f).name == ".env" for f in files):
        bad.append(".env nằm trong danh sách sẽ push (kiểm tra .gitignore)")
    total = 0
    for f in files:
        p = ROOT / f
        if not p.is_file():
            continue
        size = p.stat().st_size
        total += size
        if size >= LIMIT:
            bad.append(f"{f}: {size/1e6:.0f} MB >= 95 MB (GitHub chặn file > 100 MB)")
        if size > SCAN_MAX or p.suffix.lower() not in TEXT_EXT:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            for name, rx in PATTERNS.items():
                if rx.search(line):
                    bad.append(f"{f}:{i}: giống {name} (giá trị không in ra)")
    print(f"Số file sẽ push: {len(files)} | tổng dung lượng: {total/1e6:.1f} MB")
    big = sorted(((ROOT / f).stat().st_size, f) for f in files if (ROOT / f).is_file())[-5:]
    for s, f in reversed(big):
        print(f"  {s/1e6:6.1f} MB  {f}")
    if bad:
        print("\nKHÔNG AN TOÀN, sửa trước khi push:")
        for b in bad:
            print("  -", b)
        sys.exit(1)
    print("\nOK: không thấy key/token, không có file quá lớn, .env không nằm trong danh sách.")


if __name__ == "__main__":
    main()
