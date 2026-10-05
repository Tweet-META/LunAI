param(
    [int]$NumEnvs = 8
)

$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
Get-Command python -ErrorAction Stop | Out-Null
& python -c 'import pygame, torch, dotenv, numpy'
if ($LASTEXITCODE -ne 0) {
    throw 'The active Python environment is missing a training dependency.'
}

New-Item -ItemType Directory -Force -Path 'checkpoints', 'training_logs' | Out-Null

$variants = @(
    @{ Name = 'full'; Config = 'config.json' },
    @{ Name = 'red_blue'; Config = 'config_red_blue.json' },
    @{ Name = 'red_only'; Config = 'config_red_only.json' }
)
$seeds = @(0, 100, 200)

foreach ($variant in $variants) {
    foreach ($seed in $seeds) {
        $runName = "stage1_$($variant.Name)_seed$seed"
        $modelPath = "checkpoints/$runName.pt"
        $logPath = "training_logs/$runName.csv"
        $consolePath = "training_logs/$runName.console.txt"

        $trainArgs = @(
            'rl/train_ppo_cnn.py',
            '--config', $variant.Config,
            '--max-total-frame-steps', '300000',
            '--num-envs', "$NumEnvs",
            '--seed', "$seed",
            '--learning-rate-final', '0.0003',
            '--entropy-coef-final', '0.004',
            '--model-path', $modelPath,
            '--log-path', $logPath
        )
        if (Test-Path -LiteralPath $modelPath) {
            $trainArgs += @('--load-path', $modelPath)
            Write-Host "Resuming $runName"
        } else {
            Write-Host "Starting $runName"
        }

        & python @trainArgs *> $consolePath
        if ($LASTEXITCODE -ne 0) {
            throw "$runName failed. See $consolePath"
        }
        Write-Host "Finished $runName"
    }
}

Write-Host 'All nine stage-one runs finished.'
