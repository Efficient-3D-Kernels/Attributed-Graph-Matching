# Sequential execution of Scripts 4, 3, 5, 2 after Script 1 (no two benchmarks share the CPU).
param([int]$WaitPid = 0)
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..\..")
$logs = "benchmarks\live_runs\logs"
if ($WaitPid -gt 0) { while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 20 } }
python benchmarks\scripts\run_multi_scale_ablation_live.py --trials 50 --workers 5 *> "$logs\ablation_console.log"
python benchmarks\scripts\run_large_scale_grid_live.py --trials 10 --workers 5 *> "$logs\large_scale_grid_console.log"
python benchmarks\scripts\run_3d_superposition_live.py --trials 100 --workers 8 *> "$logs\superposition_3d_console.log"
python benchmarks\scripts\run_image_keypoints_live.py --samplings 4 --workers 8 *> "$logs\image_console.log"
"ALL DONE $(Get-Date -Format o)" | Out-File "$logs\pipeline_done.txt"
