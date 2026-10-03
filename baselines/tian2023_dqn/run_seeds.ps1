param(
    [string]$Python = 'python',
    [int[]]$Seeds = @(0, 1, 2),
    [string]$Device = 'auto',
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
if ($Seeds.Count -eq 0 -or @($Seeds | Select-Object -Unique).Count -ne $Seeds.Count) {
    throw 'Provide at least one seed, without duplicates.'
}
$runs = @(foreach ($seed in $Seeds) {
    if ($seed -lt 0 -or $seed -gt 4292) { throw "Invalid seed: $seed" }
    [PSCustomObject]@{
        Seed = $seed
        Output = Join-Path $PSScriptRoot "runs/seed_$seed"
        Arguments = @('-u', '-m', 'baselines.tian2023_dqn.train', '--seed', "$seed", '--device', $Device)
    }
})
if ($DryRun) { $runs; return }
foreach ($run in $runs) {
    if (Test-Path -LiteralPath $run.Output) { throw "Output already exists: $($run.Output)" }
}
Push-Location $projectRoot
try {
    & $Python -c 'import torch, numpy, pygame, PIL, dotenv'
    if ($LASTEXITCODE -ne 0) { throw 'Missing baseline dependency.' }
    foreach ($run in $runs) {
        Write-Host "Starting Tian DQN seed $($run.Seed)" -ForegroundColor Cyan
        $trainArgs = $run.Arguments
        & $Python @trainArgs
        if ($LASTEXITCODE -ne 0) { throw "Seed $($run.Seed) failed; remaining runs were not started." }
    }
} finally { Pop-Location }
