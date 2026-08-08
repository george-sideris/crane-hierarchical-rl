"""Full-page clearing grid: one panel per grasp cycle, empty cycles marked.

Each panel is the GoPro timelapse frame at that cycle's logged wall time, with an
inset of the cloud the policy scored at that moment (viridis = per-point grasp
score, red pin = issued target). Cycles 1-3 preceded the start of recording.
"""
import datetime
import json
import os
import subprocess

from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm

S = "/home/george/IsaacLab/crane_testbed/docs/figures/real_trials/_build/"
RUN = ("/home/george/IsaacLab/crane_testbed/logs/bc_pointcloud/"
       "scoring_margin05_2048_c/policy_debug/run_20260805_194008/")
VID = "/home/george/Downloads/GX010129.MP4"
OUT = "/home/george/IsaacLab/crane_testbed/docs/figures/real_trials/clearing_grid.jpg"
SERIF = ("/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/"
         "tex-gyre/texgyrepagella-regular.otf")

T0 = datetime.datetime(2026, 8, 5, 15, 45, 19).timestamp()
RATE = 30000 / 1001.0
CROP = (200, 950, 3300, 2160)
PW, PH = 820, 320          # photo
LH = 80                    # label strip
COLS, GUT = 4, 12
EMPTY_TINT = (252, 232, 232)


def font(sz):
    try:
        return ImageFont.truetype(SERIF, sz)
    except Exception:
        return ImageFont.load_default()


ODS = "/home/george/Documents/real_trials.ods"
N_SCORED = 30          # cycles scored for this trial in the spreadsheet
REASON = [("chok", "choke"), ("pole", "pole"), ("rack", "rack"),
          ("yaw", "yaw"), ("misalign", "yaw"), ("pinch", "pinch")]


def scored_trial():
    """(logs, note) per cycle for the BC double-mound trial, from the scored sheet."""
    import zipfile
    from xml.etree import ElementTree as ET
    ns = {"table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
          "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0"}
    root = ET.fromstring(zipfile.ZipFile(ODS).read("content.xml"))
    grid = []
    for r in root.iter("{%s}table-row" % ns["table"]):
        row = []
        for c in r.findall("table:table-cell", ns):
            rep = int(c.get("{%s}number-columns-repeated" % ns["table"], "1"))
            pel = c.find("text:p", ns)
            row += ["".join(pel.itertext()) if pel is not None else ""] * min(rep, 40)
        grid.append(row)
    grid = [r for r in grid if any(x.strip() for x in r)]
    hdr = next(i for i, r in enumerate(grid)
               if r and r[0].strip() == "BC" and "double" in " ".join(r[:3]).lower())
    out = {}
    for k in range(N_SCORED):
        r = grid[hdr + 1 + k]
        def g(i):
            return r[i].strip() if i < len(r) else ""
        out[k + 1] = (g(0), g(2))
    return out


def load_cycles():
    rows = [json.loads(l) for l in open(RUN + "decisions.jsonl")]
    gaze = {r["cycle"]: r for r in rows if r["kind"] == "gaze"}
    sc = scored_trial()
    out = []
    for c in sorted(gaze):
        if c not in sc or (gaze[c]["t_wall"] - T0) < 0:
            continue
        logs, note = sc[c]
        n = int(logs) if logs.isdigit() else 0
        tag = ""
        if n == 0:
            low = note.lower()
            tag = next((v for k, v in REASON if k in low), "empty")
        out.append((c, gaze[c]["t_wall"], n, tag))
    return out


def cell(cycle, twall, nlogs, tag):
    ok = nlogs > 0
    t = (twall - T0) / RATE
    src = S + f"grid_frames/p{cycle:03d}.png"
    if not os.path.exists(src):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", VID,
                        "-frames:v", "1", "-y", src], check=True)
    ph = Image.open(src).crop(CROP).resize((PW, PH), Image.LANCZOS)

    ins = Image.open(S + f"insets/inset_{cycle:03d}.png").convert("RGB")
    iw = int(PW * 0.36)
    ins = ins.resize((iw, int(iw * ins.height / ins.width)), Image.LANCZOS)
    bx, by = PW - iw - 8, 8
    d = ImageDraw.Draw(ph)
    d.rectangle([bx - 2, by - 2, bx + ins.width + 1, by + ins.height + 1],
                fill=(255, 255, 255), outline=(60, 60, 60), width=2)
    ph.paste(ins, (bx, by))

    c = Image.new("RGB", (PW, PH + LH), EMPTY_TINT if not ok else (255, 255, 255))
    c.paste(ph, (0, 0))
    d2 = ImageDraw.Draw(c)
    f = font(58)
    txt = f"{cycle}:  {nlogs} logs" if ok else f"{cycle}:  empty ({tag})"
    w = d2.textlength(txt, font=f)
    d2.text(((PW - w) / 2, PH + 10), txt,
            fill=(20, 20, 20) if ok else (150, 20, 20), font=f)
    d2.rectangle([0, 0, PW - 1, PH - 1], outline=(90, 90, 90) if ok else (185, 60, 60), width=3)
    return c


def key_cell():
    c = Image.new("RGB", (PW, PH + LH), (255, 255, 255))
    d = ImageDraw.Draw(c)
    f = font(56)
    bw, bh = int(PW * 0.72), 42
    x0, y0 = (PW - bw) // 2, PH // 2 - 40
    for x in range(bw):
        r, g, b = cm.get_cmap("viridis")(x / max(bw - 1, 1))[:3]
        for y in range(bh):
            c.putpixel((x0 + x, y0 + y), (int(r * 255), int(g * 255), int(b * 255)))
    d.rectangle([x0 - 2, y0 - 2, x0 + bw + 1, y0 + bh + 1], outline=(60, 60, 60), width=2)
    d.text((x0, y0 + bh + 10), "low", fill=(20, 20, 20), font=f)
    w = d.textlength("high", font=f)
    d.text((x0 + bw - w, y0 + bh + 10), "high", fill=(20, 20, 20), font=f)
    t = "per-point grasp score"
    d.text(((PW - d.textlength(t, font=f)) / 2, y0 - 74), t, fill=(20, 20, 20), font=f)
    t2 = "red pin: issued target"
    d.text(((PW - d.textlength(t2, font=f)) / 2, y0 + bh + 82), t2, fill=(20, 20, 20), font=f)
    return c


def main():
    os.makedirs(S + "grid_frames", exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cyc = load_cycles()
    cells = [cell(c, w, n, t) for c, w, n, t in cyc] + [key_cell()]
    rows = (len(cells) + COLS - 1) // COLS
    CW, CH = PW, PH + LH
    sheet = Image.new("RGB", (COLS * CW + (COLS - 1) * GUT, rows * CH + (rows - 1) * GUT), "white")
    for i, c in enumerate(cells):
        sheet.paste(c, ((i % COLS) * (CW + GUT), (i // COLS) * (CH + GUT)))
    sheet.save(OUT, quality=90)
    n_ok = sum(1 for _, _, n, _ in cyc if n > 0)
    print(f"{OUT}  {sheet.size}  cycles {cyc[0][0]}-{cyc[-1][0]} "
          f"({n_ok} successful, {len(cyc)-n_ok} empty), {rows} rows")
    print("MB %.1f" % (os.path.getsize(OUT) / 1e6))


if __name__ == "__main__":
    main()
