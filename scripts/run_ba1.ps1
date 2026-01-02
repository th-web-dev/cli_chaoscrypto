$ErrorActionPreference = 'Stop'

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Resolve-Path (Join-Path $RootDir '..')
$OutDir = Join-Path $RootDir 'out\ba1'

New-Item -ItemType Directory -Force -Path (Join-Path $OutDir 'bench') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $OutDir 'analyze') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $OutDir 'report\plots') | Out-Null

$metaPath = Join-Path $OutDir 'run_meta.txt'
$lines = @(
    "date_utc=$([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'))",
    (& python --version),
    "os=$([System.Environment]::OSVersion.VersionString)",
    "machine=$env:COMPUTERNAME",
    "user=$env:USERNAME",
    "pwd=$RootDir"
)
$lines | Out-File -FilePath $metaPath -Encoding utf8

& python -m pip freeze | Out-File -FilePath $metaPath -Encoding utf8 -Append

& python -m chaoscrypto.cli.app benchmark `
  --config (Join-Path $RootDir 'examples\ba1_benchmark.yaml') `
  --out (Join-Path $OutDir 'bench\results.csv') `
  --out-json (Join-Path $OutDir 'bench\results.json')

& python -m chaoscrypto.cli.app analyze `
  --config (Join-Path $RootDir 'examples\ba1_analyze.yaml') `
  --out (Join-Path $OutDir 'analyze\analysis.csv') `
  --out-json (Join-Path $OutDir 'analyze\analysis.json')

& python -m chaoscrypto.cli.app report `
  --bench-csv (Join-Path $OutDir 'bench\results.csv') `
  --analysis-csv (Join-Path $OutDir 'analyze\analysis.csv') `
  --out (Join-Path $OutDir 'report\report.md') `
  --plots-dir (Join-Path $OutDir 'report\plots') `
  --no-timestamp
