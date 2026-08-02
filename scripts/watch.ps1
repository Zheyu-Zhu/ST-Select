# Live one-line GPU + CPU + RAM dashboard for a terminal.
#
#   powershell -NoProfile -File scripts\watch.ps1
#   powershell -NoProfile -File scripts\watch.ps1 -IntervalSec 5
#
# Ctrl-C to stop. Read-only: safe to run while jobs are training.

param([int]$IntervalSec = 2)

Write-Host "GPU/CPU watch - Ctrl-C to stop" -ForegroundColor Cyan
Write-Host ""

while ($true) {
    $g = (nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw `
                     --format=csv,noheader,nounits) -split ',' | ForEach-Object { $_.Trim() }
    $cpu = [math]::Round((Get-Counter '\Processor(_Total)\% Processor Time' `
                          -SampleInterval 1 -MaxSamples 1).CounterSamples[0].CookedValue, 0)
    $os  = Get-CimInstance Win32_OperatingSystem
    $ramFree = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
    $ramTot  = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    $vramPct = [math]::Round(100 * [double]$g[1] / [double]$g[2], 0)

    # Colour the GPU number so a stalled job is visible at a glance.
    $gpuUtil = [int]$g[0]
    $col = if ($gpuUtil -ge 80) { 'Green' } elseif ($gpuUtil -ge 20) { 'Yellow' } else { 'DarkGray' }

    Write-Host ("{0}  " -f (Get-Date -Format 'HH:mm:ss')) -NoNewline
    Write-Host ("GPU {0,3}%" -f $gpuUtil) -ForegroundColor $col -NoNewline
    Write-Host ("  VRAM {0,5}/{1} MiB ({2}%)  {3}C  {4}W  |  CPU {5,3}%  RAM {6}/{7} GB free" `
                -f $g[1], $g[2], $vramPct, $g[3], $g[4], $cpu, $ramFree, $ramTot)

    Start-Sleep -Seconds $IntervalSec
}
