# Oneiros Phase 3 Review 1 - panel demonstration preflight.
# Run from the oneiros root. Runs all five live demo steps and times each one.
# Usage:  powershell -ExecutionPolicy Bypass -File .\scripts\demo_preflight.ps1

$ErrorActionPreference = 'Continue'
$PY = "C:\Users\saira\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

function Step($name, $block) {
    Write-Host ""
    Write-Host "=== $name ===" -ForegroundColor Cyan
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $block
    $sw.Stop()
    Write-Host ("[{0:N1}s]" -f $sw.Elapsed.TotalSeconds) -ForegroundColor DarkGray
}

Write-Host "Interpreter check" -ForegroundColor Cyan
Write-Host ("  ML interpreter present : " + (Test-Path $PY))
& $PY --version

Step "0. Required artifacts (expect five True)" {
    Test-Path .\data\corpus\v3_final_candidate\records.json
    Test-Path .\checkpoints\v3_full_sft_monitored_20260819_1\sft_adapter\adapter_model.safetensors
    Test-Path .\results\v3_full_sft_monitored_20260819_1\sft_validation_results.json
    Test-Path .\results\v3_dpo_smoke_20260820_1\dpo_validation_checkpoint_100.json
    Test-Path .\results\v3_repo1024_aligned_constant_lr_smoke_800_20260821_1\sft_monitor_checkpoint_143.json
}

Step "1. Arnav - canonical corpus verification" {
    & $PY -c "from pathlib import Path; from harness.corpus import verify_corpus; m=verify_corpus(Path('data/corpus/v3_final_candidate')); print('Corpus:',m['corpus_id']); print('Records:',m['training_records']); print('Splits:',{k:v['record_count'] for k,v in m['splits'].items()}); print('All quality gates:',all(m['quality_gate'].values()))"
}

Step "2. Saaj - execution oracle replay" {
    & $PY scripts\demo_oracle_replay.py
}

Step "3. Saaj - focused regression suite (expect 42 passed)" {
    & $PY -m pytest -q `
        tests\test_model_runtime_alignment.py `
        tests\test_sft_best_checkpoint.py `
        tests\test_sft_monitor_comparison.py `
        tests\test_preflight_sft_run.py `
        tests\test_training_smoke_selection.py
}

Step "4. Venkat - adapter and checkpoint metadata" {
    Get-Item -LiteralPath '.\checkpoints\v3_full_sft_monitored_20260819_1\sft_adapter\adapter_model.safetensors' |
        Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
    $m = Get-Content '.\checkpoints\v3_full_sft_monitored_20260819_1\sft_metadata.json' -Raw | ConvertFrom-Json
    [pscustomobject]@{
        Status         = $m.status
        CompletedSteps = $m.completed_optimizer_steps
        MonitorGate    = $m.monitor_gate_passed
        BestCheckpoint = $m.monitor_best_adapter
        BestPanel      = "$($m.monitor_best_metrics.function_validation_killed)/$($m.monitor_best_metrics.function_validation_records)"
        BestPanelRate  = ('{0:P2}' -f [double]$m.monitor_best_metrics.function_kill_rate)
    } | Format-List
}

Step "5. Anushka - result table from saved JSON" {
    $runs = @(
        @{Name = 'V2 SFT'; Path = 'results\v2_full_sft\sft_validation_results.json' },
        @{Name = 'V3 Full SFT'; Path = 'results\v3_full_sft_monitored_20260819_1\sft_validation_results.json' },
        @{Name = 'V3 SFT + DPO'; Path = 'results\v3_dpo_smoke_20260820_1\dpo_validation_checkpoint_100.json' },
        @{Name = 'Latest aligned SFT smoke'; Path = 'results\v3_repo1024_aligned_constant_lr_smoke_800_20260821_1\sft_monitor_checkpoint_143.json' }
    )
    $rows = @()
    foreach ($run in $runs) {
        $j = Get-Content -LiteralPath $run.Path -Raw | ConvertFrom-Json
        $rows += [pscustomobject]@{
            Experiment  = $run.Name
            Evaluated   = $j.function_validation_records
            Killed      = $j.function_validation_killed
            'Kill rate' = ('{0:P2}' -f [double]$j.function_kill_rate)
        }
    }
    $rows | Format-Table -AutoSize
}

Write-Host ""
Write-Host "Preflight complete. Compare every value above against the guide before presenting." -ForegroundColor Green
