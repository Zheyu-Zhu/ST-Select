"""Read exact learning-rate scalars and epoch batch counts for saved references.

Uses only the Python standard library to read TensorBoard TFRecord/protobuf
scalar events. No training or model checkpoint loading occurs.
"""
import csv
import hashlib
import json
import re
import struct
from pathlib import Path


def varint(data, offset):
    result, shift = 0, 0
    while True:
        byte = data[offset]
        offset += 1
        result |= (byte & 127) << shift
        if byte < 128:
            return result, offset
        shift += 7
        assert shift <= 63


def fields(data):
    offset = 0
    while offset < len(data):
        key, offset = varint(data, offset)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, offset = varint(data, offset)
        elif wire == 1:
            value, offset = data[offset:offset+8], offset+8
        elif wire == 2:
            n, offset = varint(data, offset)
            value, offset = data[offset:offset+n], offset+n
        elif wire == 5:
            value, offset = data[offset:offset+4], offset+4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield number, wire, value


def lr_scalars(path):
    """Event.step=2, Event.summary=5; Summary.value=1, Value.tag=1/simple_value=2.

    Standard TensorFlow event protobuf field numbers. TFRecord framing is
    uint64 length, uint32 length CRC, payload, uint32 payload CRC. These saved
    local files are not modified; this reader checks lengths but not CRCs.
    """
    result=[]
    with path.open("rb") as f:
        while header := f.read(12):
            assert len(header) == 12
            length = struct.unpack("<Q", header[:8])[0]
            data = f.read(length)
            assert len(data) == length and len(f.read(4)) == 4
            event = list(fields(data))
            step = next((v for n,w,v in event if n == 2), 0)
            for n,w,summary in event:
                if n != 5:
                    continue
                for sn,sw,value in fields(summary):
                    if sn != 1:
                        continue
                    values = list(fields(value))
                    tag=next((v.decode() for vn,vw,v in values if vn == 1), "")
                    if tag == "lr":
                        scalar=next((struct.unpack("<f",v)[0] for vn,vw,v in values if vn == 2), None)
                        assert scalar is not None
                        result.append((step,scalar))
    return result

ROOT = Path(r"C:\Users\zheyu\workspace\stnet_ref\ST-Net")
OUT = Path(__file__).resolve().parent
rows, sources = [], {}
for dataset in ("HER2", "cscc"):
    for arm, log_arm in (("random_0.5", "random_0.5"), ("random_1.0", "full")):
        for fold in range(4):
            log_path = ROOT / "logs" / f"{dataset}_{log_arm}_fold{fold}.log"
            text = log_path.read_text(encoding="utf-8", errors="replace")
            batches = {int(epoch): int(n) for epoch, n in re.findall(
                r"Epoch (\d+)/\d+:\s+[^\r\n]*?\|\s*\d+/(\d+)\s*\[", text)}
            assert set(range(1, 51)).issubset(batches), (log_path, batches)
            steps50 = sum(batches[e] for e in range(1,51))
            steps_final = sum(batches.values())
            directory = ROOT / "weights_st" / dataset / arm / f"fold{fold}" / "log"
            events = sorted(directory.glob("events.out.tfevents.*"), key=lambda p: int(p.name.split(".")[3]))
            event_path = events[-1]
            lr = lr_scalars(event_path)
            scalar_by_step = dict(lr)
            assert len(lr) == steps_final, (event_path, len(lr), steps_final)
            at50 = scalar_by_step[steps50-1]
            expected = 1e-4 * (1 - (steps50-1)/100000000)**0.9
            assert abs(at50-expected) < 1e-10
            rows.append(dict(dataset=dataset, arm=arm, fold=fold,
                total_epochs=max(batches), steps_epoch50=steps50, steps_final=steps_final,
                lr_first=lr[0][1], lr_end_epoch50=at50, lr_final=lr[-1][1],
                expected_lr_end_epoch50_iteration_polynomial=expected,
                event_source=str(event_path), log_source=str(log_path)))
            for path in (log_path,event_path):
                sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
with (OUT / "learning_rate_and_steps.csv").open("w", encoding="utf-8", newline="") as f:
    writer=csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(OUT / "learning_rate_source_hashes.json").write_text(json.dumps(sources, indent=2)+"\n", encoding="utf-8")
print(json.dumps(rows, indent=2))
