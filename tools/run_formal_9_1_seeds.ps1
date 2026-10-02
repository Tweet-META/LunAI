param(
    [string]$Python = 'python',
    [ValidateNotNullOrEmpty()]
    [ValidateRange(0, 2147483647)]
    [int[]]$Seeds = @(1, 2),
    [switch]$NoPccm,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

$variants = @(
    @{ Name = 'lunai_v9.1'; Scales = 'full'; PCC = '0' }
    @{ Name = 'lunai_v9.1_pcc'; Scales = 'full'; PCC = '0.1' }
    @{ Name = 'lunai_v9.1_noyellow'; Scales = 'red_blue'; PCC = '0' }
    @{ Name = 'lunai_v9.1_noyellow_pcc'; Scales = 'red_blue'; PCC = '0.1' }
)
$observationMode = 'trajectory'
if ($NoPccm) {
    $variants = @(@{ Name = 'lunai_v9.1_nopccm'; Scales = 'full'; PCC = '0' })
    $observationMode = 'occupancy_only'
}
if (@($Seeds | Select-Object -Unique).Count -ne $Seeds.Count) {
    throw 'Seeds must not contain duplicates.'
}

# Match seed 0: anneal through 1M, then hold the endpoint values through 2M.
$common = @(
    '-m', 'rl.train_ppo_cnn'
    '--episodes', '1000000'
    '--max-steps', '1800'
    '--num-envs', '8'
    '--action-repeat', '1'
    '--frame-stack', '4'
    '--frame-stack-interval', '1'
    '--level-files',
    'level_th06_stage3_spell1.json',
    'level_th06_stage3_spell2.json',
    'level_th06_stage3_spell3.json'
    '--level-spawn-time-jitter', '0'
    '--player-start-margin', '51.2'
    '--rollout-steps', '1024'
    '--minibatch-size', '256'
    '--update-epochs', '4'
    '--gamma', '0.99'
    '--gae-lambda', '0.95'
    '--learning-rate-final', '0.00004'
    '--entropy-coef-final', '0.0015'
    '--clip-range', '0.2'
    '--value-coef', '0.5'
    '--max-grad-norm', '0.5'
    '--target-kl', '0.03'
    '--hidden-dim', '128'
    '--architecture-version', '2'
    '--pccm-implementation', 'numba'
    '--pccm-observation-mode', $observationMode
    '--pccm-prediction-frames', '5'
    '--pccm-halo-width', '20'
    '--pccm-wall-margin', '0.12'
    '--pccm-upper-field-threshold', '0.7'
    '--pccm-upper-field-cost', '0.3'
    '--save-interval', '5'
    '--device', 'auto'
)

$runs = @(
    foreach ($seed in $Seeds) {
        foreach ($variant in $variants) {
            $source = "checkpoints/formal_1800_1m/seed_$seed/$($variant.Name).pt"
            foreach ($phase in 1, 2) {
                $model = "checkpoints/formal_1800_$($phase)m/seed_$seed/$($variant.Name).pt"
                $log = "training_logs/formal_1800_$($phase)m/seed_$seed/$($variant.Name).csv"
                $trainArgs = $common + @(
                    '--seed', "$seed"
                    '--observation-scales', $variant.Scales
                    '--pccm-reward-weight', $variant.PCC
                    '--max-total-frame-steps', "$($phase * 1000000)"
                    '--model-path', $model
                    '--log-path', $log
                )
                if ($phase -eq 1) {
                    $trainArgs += @('--learning-rate', '0.0003', '--entropy-coef', '0.004')
                } else {
                    $trainArgs += @(
                        '--load-path', $source
                        '--learning-rate', '0.00004'
                        '--entropy-coef', '0.0015'
                    )
                }
                [PSCustomObject]@{
                    Name = "seed_$seed / $($variant.Name) / $($phase)m"
                    Model = $model
                    Log = $log
                    Arguments = $trainArgs
                }
            }
        }
    }
)

# DryRun returns the exact argument arrays without loading Python or writing files.
if ($DryRun) {
    $runs
    return
}

Push-Location $projectRoot
try {
    Get-Command $Python -ErrorAction Stop | Out-Null
    foreach ($run in $runs) {
        foreach ($path in @($run.Model, $run.Log)) {
            if (Test-Path -LiteralPath $path) {
                throw "Output already exists: $path. This script starts fresh seeds; it will not overwrite or append existing runs."
            }
        }
    }
    & $Python -c 'import torch, numpy, pygame, dotenv, numba'
    if ($LASTEXITCODE -ne 0) {
        throw 'The selected Python environment is missing a training dependency.'
    }

    foreach ($run in $runs) {
        Write-Host "`nStarting $($run.Name)" -ForegroundColor Cyan
        $trainArgs = $run.Arguments
        & $Python @trainArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed: $($run.Name). Remaining runs were not started."
        }
        if (-not (Test-Path -LiteralPath $run.Model -PathType Leaf)) {
            throw "Training returned without its checkpoint: $($run.Model)"
        }
        Write-Host "Finished $($run.Name)" -ForegroundColor Green
    }
    Write-Host "`nCompleted $($Seeds.Count * $variants.Count) training runs, with both 1M and 2M checkpoints." -ForegroundColor Green
} finally {
    Pop-Location
}
