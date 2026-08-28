# Start llama-server in router mode (on-demand model loading).
#
# Usage:  powershell -ExecutionPolicy Bypass -File F:\clients\text\llama.cpp\start-router.ps1
#         ... -Port 1243          (default)
#         ... -PerSlotCtx 32000   cap KV per slot so one request can't starve others
#         ... -ModelsMemoryMax 80 unload models when total VRAM exceeds N GiB

param(
    [int]$Port = 1243,
    [int]$PerSlotCtx = 0,
    [int]$ModelsMemoryMax = 0
)

$repo   = Split-Path -Parent $MyInvocation.MyCommand.Path
$server = Join-Path $repo "build\bin\llama-server.exe"
$preset = Join-Path $repo "models-preset.ini"

if (-not (Test-Path $server)) { throw "llama-server.exe not found — run build-cuda.ps1 first" }

$srvArgs = @(
    "--port", $Port,
    "--models-dir", "D:\models\text\models",
    "--models-preset", $preset
)

if ($PerSlotCtx -gt 0) {
    $srvArgs += "--kv-unified-per-slot"
    $srvArgs += $PerSlotCtx
}

if ($ModelsMemoryMax -gt 0) {
    $srvArgs += "--models-memory-max"
    $srvArgs += $ModelsMemoryMax
}

Write-Host "Starting router on :$Port"
Write-Host "  preset : $preset"
Write-Host "  models : D:\models\text\models"
if ($PerSlotCtx -gt 0)    { Write-Host "  kv/slot: $PerSlotCtx tokens" }
if ($ModelsMemoryMax -gt 0) { Write-Host "  mem cap: ${ModelsMemoryMax} GiB" }
Write-Host ""

& $server @srvArgs
