<#
 Đưa repo lên GitHub an toàn. Chạy từ gốc repo, trong môi trường env_AI (cần `python` và `git`):
   conda activate env_AI
   powershell -ExecutionPolicy Bypass -File scripts\publish_to_github.ps1
 Thêm -IncludeEmbeddings để đưa cả cache embedding 61 MB (data/index/emb_scale_*.npy) cho người chấm khỏi phải embed lại.
 Thêm -DryRun để chỉ kiểm tra và xem danh sách, không commit/push.
#>
param([switch]$IncludeEmbeddings, [switch]$DryRun, [string]$Message = "Hybrid Graph-RAG ArXiv: ma nguon, benchmark v2, ket qua va tai lieu (final)")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

# 1) Kiểm tra an toàn (key, .env, file quá lớn)
python -X utf8 scripts/check_before_push.py
if ($LASTEXITCODE -ne 0) { throw "Kiem tra an toan that bai. Sua cac muc o tren roi chay lai." }

# 2) .env phải bị ignore và không được theo dõi
git check-ignore -q .env
if ($LASTEXITCODE -ne 0) { throw ".env chua bi .gitignore chan." }
if (git ls-files .env) { throw ".env dang bi theo doi boi git. Chay: git rm --cached .env" }

# 3) Danh tính commit
if (-not (git config user.name) -or -not (git config user.email)) {
  throw "Chua cau hinh git. Chay: git config user.name `"Ten ban`"  va  git config user.email `"email@cua.ban`""
}

# 4) Stage
git add -A
if ($IncludeEmbeddings) { git add -f -- data/index/emb_scale_6cdeb8a3dc.npy }
git status --short | Select-Object -First 25
$n = (git diff --cached --name-only | Measure-Object).Count
Write-Host "`nSo file se commit: $n"
if ($DryRun) { Write-Host "DryRun: dung o day (chua commit). De huy stage: git reset"; exit 0 }

$ans = Read-Host "Commit va push len origin/main? (y/N)"
if ($ans -ne "y") { Write-Host "Da dung. De huy stage: git reset"; exit 0 }

# 5) Commit + push
git commit -m $Message
git branch -M main
git push -u origin main
Write-Host "`nXong. Mo: $(git remote get-url origin)"
