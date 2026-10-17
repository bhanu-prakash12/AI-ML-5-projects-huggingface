# Laptop Setup (macOS Apple Silicon) — minimum software

Disk needed: ~15 GB free (models + datasets + venvs).

## 1. Install the toolchain (once)
```bash
xcode-select --install                      # git + compilers (skip if already installed)
# Homebrew (skip if `brew --version` works): https://brew.sh
brew install libomp                         # OpenMP runtime needed by LightGBM/XGBoost
curl -LsSf https://astral.sh/uv/0.5.11/install.sh | sh
uv python install 3.11
```
- Claude Code: `curl -fsSL https://claude.ai/install.sh | bash`

## 2. Check
```bash
git --version; uv --version; uv python find 3.11; claude --version
ls "$(brew --prefix libomp)/lib/libomp.dylib"
```

## 3. Pre-download the small models (optional, saves waiting later)
Models are cached in ~/.cache/huggingface and loaded directly by each project (no server).
```bash
uvx --from huggingface-hub==0.27.0 huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct   # P2 (~3 GB)
uvx --from huggingface-hub==0.27.0 huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct   # P3 (~1 GB)
uvx --from huggingface-hub==0.27.0 huggingface-cli download Qwen/Qwen2-VL-2B-Instruct    # P5 (~4.5 GB)
```

## 4. Start
```bash
cd ai-eng-projects && git init && git add . && git commit -m "kit"
claude                      # then type /model and choose Sonnet
```
First message to Claude Code:
```
Read PROGRESS.md and SETUP.md. Run the checks in SETUP.md §2, write the versions into
PROGRESS.md carry-over notes, commit, then wait. Next I will run /plan p1.
```

## 5. Loop per project
`/plan pN` → answer questions → `/clear` → `/build pN 1` → `/clear` → `/build pN 2` … →
`/verify pN` → `/clear` → `/readme pN`.
Build phases: p1 = 4, p2 = 5, p3 = 5, p4 = 6, p5 = 5 (the last spec phase is /verify + /readme).

## 6. Move to GitHub later
```bash
git remote add origin https://github.com/<you>/ai-eng-projects.git
git push -u origin main
```
data/, artifacts/, mlruns/, chroma/ and *.db are gitignored, so the repo stays small and
anyone re-creates them with `make data train`. For Codespaces later, ask Claude Code to add
a .devcontainer (Linux needs the CPU torch wheel index; models load the same way on CPU).
