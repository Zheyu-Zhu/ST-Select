"""Background resource recorder — samples GPU/CPU/RAM to a CSV during long runs.

Ad-hoc `nvidia-smi` calls only ever show the instant you happen to look. That is
useless after the fact: when a 10 h job finishes you cannot tell whether it was
saturated throughout, thermally throttled at hour 6, or stalled waiting on disk.
This records the whole run so the question is answerable later.

Started automatically by run_queue.py (one CSV per queue session) and safe to run
by hand alongside anything:

    python scripts/log_resources.py --out results_pg/resources.csv --interval 30

Columns: iso time, gpu_util%, vram_used_mb, vram_total_mb, gpu_temp_c, gpu_power_w,
sm_clock_mhz, cpu_pct, ram_free_gb, ram_total_gb, python_procs, note.

Read it back with scripts/log_resources.py --summarise <csv>, which reports the
duty cycle and flags throttling and idle gaps rather than making you eyeball
thousands of rows.
"""

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

FIELDS = ["time", "gpu_util", "vram_used_mb", "vram_total_mb", "gpu_temp_c",
          "gpu_power_w", "sm_clock_mhz", "cpu_pct", "ram_free_gb",
          "ram_total_gb", "python_procs", "note"]


def _ps(cmd: str) -> str:
    try:
        return subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                              capture_output=True, text=True, timeout=25).stdout.strip()
    except Exception:
        return ""


def sample(note: str = "") -> dict:
    row = {k: "" for k in FIELDS}
    row["time"] = datetime.now().isoformat(timespec="seconds")
    row["note"] = note
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,"
             "temperature.gpu,power.draw,clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=25).stdout.strip().splitlines()
        if out:
            g = [x.strip() for x in out[0].split(",")]
            (row["gpu_util"], row["vram_used_mb"], row["vram_total_mb"],
             row["gpu_temp_c"], row["gpu_power_w"], row["sm_clock_mhz"]) = g[:6]
    except Exception:
        pass
    cpu = _ps("[math]::Round((Get-Counter '\\Processor(_Total)\\% Processor Time' "
              "-SampleInterval 1 -MaxSamples 1).CounterSamples[0].CookedValue,1)")
    row["cpu_pct"] = cpu
    mem = _ps("$o=Get-CimInstance Win32_OperatingSystem; "
              "'{0:N2},{1:N2}' -f ($o.FreePhysicalMemory/1MB),($o.TotalVisibleMemorySize/1MB)")
    if "," in mem:
        row["ram_free_gb"], row["ram_total_gb"] = [x.strip() for x in mem.split(",")]
    row["python_procs"] = _ps("(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" "
                              "| Measure-Object).Count")
    return row


def record(out: Path, interval: int, note: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        while True:
            w.writerow(sample(note))
            fh.flush()          # flush every row: a killed logger must not lose history
            time.sleep(max(1, interval))


def summarise(path: Path) -> None:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        print("empty log")
        return

    def nums(k):
        out = []
        for r in rows:
            try:
                out.append(float(r[k]))
            except (ValueError, KeyError):
                pass
        return out

    gu, temp, pw, cpu, clk = (nums("gpu_util"), nums("gpu_temp_c"),
                              nums("gpu_power_w"), nums("cpu_pct"), nums("sm_clock_mhz"))
    ram = nums("ram_free_gb")
    span = (datetime.fromisoformat(rows[-1]["time"])
            - datetime.fromisoformat(rows[0]["time"]))
    print(f"{path}  —  {len(rows)} samples over {span}")
    if gu:
        busy = sum(1 for x in gu if x >= 50) / len(gu)
        idle = sum(1 for x in gu if x < 5) / len(gu)
        print(f"  GPU util   mean {sum(gu)/len(gu):5.1f}%   max {max(gu):3.0f}%   "
              f"busy(>=50%) {busy:.0%}   idle(<5%) {idle:.0%}")
        if idle > 0.10:
            print(f"    ! GPU idle for {idle:.0%} of the run — a job stalled, "
                  f"or the queue had a gap")
    if temp:
        print(f"  GPU temp   mean {sum(temp)/len(temp):5.1f}C  max {max(temp):3.0f}C")
        if max(temp) >= 84:
            print(f"    ! peaked at {max(temp):.0f}C — thermal throttling likely")
    if clk and max(clk) > 0:
        low = sum(1 for x in clk if x < 0.85 * max(clk)) / len(clk)
        if low > 0.15:
            print(f"    ! SM clock below 85% of peak for {low:.0%} of samples "
                  f"— throttling or power limit")
    if pw:
        print(f"  GPU power  mean {sum(pw)/len(pw):5.1f}W  max {max(pw):5.1f}W")
    if cpu:
        print(f"  CPU        mean {sum(cpu)/len(cpu):5.1f}%   max {max(cpu):5.1f}%")
    if ram:
        print(f"  RAM free   min  {min(ram):5.2f} GB  mean {sum(ram)/len(ram):5.2f} GB")
        if min(ram) < 1.0:
            print(f"    ! dropped to {min(ram):.2f} GB free — OOM risk")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_pg/resources.csv")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--note", default="")
    ap.add_argument("--summarise", metavar="CSV")
    a = ap.parse_args()
    if a.summarise:
        summarise(Path(a.summarise))
        return
    try:
        record(Path(a.out), a.interval, a.note)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
