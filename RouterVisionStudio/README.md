# AVTR — Automatic Vision Tab Router

Current setup and behavior (2026-09-05): see [OPERATIONS.md](OPERATIONS.md).
The older engineering notes below are retained for reference; their dataset
measurements and timings are not production validation claims.

> Current development entry point: `production_app.py` (launched by `run.bat`).
> It is a Sobel-edge + DINOv2 GOOD/NG gate with a **simulation-only** conveyor
> interlock. Physical PLC output remains disabled until the real protocol,
> wiring map, polarity and acknowledgement sequence are approved.

## Current mass-production decision path

`production_app.py` no longer uses dimensional measurement or a learned-good
baseline for RELEASE. Its fail-safe path is:

```
complete panel -> crop -> grayscale/blur -> Sobel magnitude -> DINOv2
               -> every image GOOD above confidence threshold -> RELEASE
               -> any NG / unreadable / incomplete / uncertain -> HOLD
```

The model classes are `GOOD` and `NG`. Models saved before
`sobel-magnitude-v1` are rejected and must be retrained; silently applying an
old colour-image head to Sobel input is not allowed. Measurement, calibration
and `.rcp` views remain in `app.py` as engineering tools for a later metrology
phase and do not participate in the production verdict.

## Installation and launch

Requirements: Windows, Python 3.10+, PySide6, OpenCV, NumPy and PyTorch.

```powershell
py -m venv venv
.\venv\Scripts\python.exe -m pip install -r .\RouterVisionStudio\requirements.txt
Copy-Item .\RouterVisionStudio\config.example.json .\RouterVisionStudio\config.json
$env:AVTR_SETTINGS_PASSWORD = "choose-your-own-password"
[Environment]::SetEnvironmentVariable("AVTR_SETTINGS_PASSWORD", $env:AVTR_SETTINGS_PASSWORD, "User")
.\RouterVisionStudio\run.bat
```

Edit the copied `config.json` or select an AUO6000 export root from the
password-protected Setting page. Runtime configuration, labels, machine data,
Sobel outputs and trained model files are intentionally excluded from Git.

`production_app.py` is fail-safe: no compatible trained model means HOLD. A
new model must be validated against representative GOOD and real NG samples
before any physical conveyor/PLC integration is approved.

## Legacy Engineering Studio

`app.py` contains the earlier engineering workspace for measurement,
calibration, cut-path inspection, labeling and model experiments. Launch it
directly only for engineering work:

```bat
cd RouterVisionStudio
python app.py
```

---

## Legacy Engineering Studio: 6 tabs

| แท็บ | ทำอะไร |
|---|---|
| **Dashboard** | เฝ้าโฟลเดอร์ภาพ ขึ้นภาพใหม่ทันทีที่เครื่องเขียน วัดร่อง + จำแนกด้วยโมเดล |
| **Inspect** | ซ้าย = แผ่นที่ผลิตเรียงตาม SN, ขวา = ภาพทั้ง 21 ใบพร้อมค่าเบี่ยง mm |
| **Cut Path** | วาดเส้นทางการตัดที่อ่านจากไฟล์ recipe `.rcp` โดยตรง |
| **Label** | เลือกภาพหลายใบพร้อมกัน กดปุ่ม **GOOD** หรือ **NG** |
| **Train** | เทรน DINOv2 บน label ที่ทำไว้ แล้วเซฟโมเดล |
| **Config** | path, PixelSize, threshold, calibrate baseline |

---

## เส้นทางการตัด (Cut Path) — อ่านจาก recipe จริง

ไฟล์ `.rcp` เป็น .NET BinaryWriter เก็บจุดเป็น `[0x01][float X][float Y][float Z]` 13 ไบต์ต่อจุด

**จุดที่ต้องระวัง:** ชื่อ element (`Line`, `SubBoard_01`) เป็น UTF-16 แต่**บางตัวอยู่ที่ byte offset คี่** ถ้าค้นด้วยสตริงจะหาไม่เจอราวครึ่งหนึ่ง — โมดูลนี้จึงสแกนหา pattern ของจุดโดยตรง

ผลที่ได้จาก `160914002C01.rcp`:

```
43 segments, 18 slots, 122 points, span 362.05 x 253.65 mm

  #  Centre X    Centre Y   Width   Length
  1     70.88      -50.97   0.735    5.045   mm
  2    137.86      -51.02   0.740    4.960
  3    204.86      -50.98   0.750    4.670
  4    271.88      -51.03   0.740    4.820
```

ร่อง 0.74 × 5 mm เรียงห่างกัน ~67 mm คือ **breakaway tab** — ตรงกับความกว้างร่องที่วัดได้จากภาพ (2.4-2.5 mm คือร่องกัดหลัก ส่วน 0.74 mm คือ tab)

ทดสอบกับ recipe อื่นแล้วใช้ได้: `160914002D01` (4 slots), `204860000E-R01` (10 slots)

---

## การวัด — ใช้วิธีของแอปนี้

สแกนแต่ละแถวจากกลางภาพออกซ้าย/ขวา หาพิกเซล solder mask ตัวแรก ระยะห่าง = ร่องกัด แล้วเอา median ของ 60 แถว เพื่อกันเศษฝุ่นรบกวน

- `PixelSize` = **0.008692709 mm/px** อ่านจาก `Eqp.cfg` อัตโนมัติ (115.04 px/mm)
- ความคลาดเคลื่อน **~0.04 mm** วัดข้าม 40 แผ่น
- ตัดสิน NG เมื่อเกิน**อย่างใดอย่างหนึ่งที่เข้มกว่า** ระหว่างขีดจำกัด mm กับ `sigma_k × sd` ของตำแหน่งนั้น

---

## Label — มีแค่ 2 class

**`GOOD` / `NG`** เท่านั้น

ถ้าภาพไหนตัดสินไม่ได้ **ไม่ต้อง label** — ภาพที่ไม่มี label จะไม่ถูกนำไปเทรน ไม่ต้องมี class ที่สาม

เงื่อนไขก่อนเทรน: ต้องมีทั้ง 2 class และแต่ละ class อย่างน้อย 5 ภาพ ถ้ายังไม่ครบแอปจะบอกว่าขาดอะไร เช่น

```
Need labels for both classes; none yet for NG.
Need at least 5 images per class; short on NG=3.
20 images (GOOD 8, NG 12).          <- พร้อมเทรน
```

---

## เทรน Sobel + DINOv2

ภาพทุกใบถูกแปลงเป็น Sobel edge magnitude ก่อนเข้าโมเดล จากนั้น **Backbone แช่แข็ง เทรนแค่ linear head** บน embedding 384 มิติ — เป็นขนาดโมเดลที่เหมาะกับ label ไม่กี่สิบใบ ถ้า fine-tune ทั้งเครือข่ายมันจะแค่ท่องจำ

- ครั้งแรกโหลด weights ~85 MB จากอินเทอร์เน็ต หลังจากนั้นทำงาน offline ได้
- เทรน 36 ภาพ 300 epoch บน CUDA ใช้ **~6-10 วินาที**
- โมเดลที่เซฟมีแค่ ~5 KB (เก็บเฉพาะ head)
- แบ่ง validation แบบ stratified + ถ่วงน้ำหนัก class ที่มีน้อย

### ผลทดสอบที่บอกอะไรสำคัญ

ผมทดสอบ 2 แบบ:

| การทดลอง | ผล |
|---|---|
| Label ตามตำแหน่งที่**ต่างกันจริง** (position 0-2 vs 6-8) | train 100% / **val 100%** |
| Label แบบ**มั่ว** (18 ใบแรก=OK, 18 ใบถัดไป=OVERCUT ทั้งที่ไม่ต่างกันจริง) | train 82% / **val 25%** |

**การทดลองที่ 2 คือข่าวดี** — เมื่อ label ไม่มีความหมายจริง โมเดลรายงาน val 25% (แย่กว่าเดา 50%) แปลว่าระบบวัดผล**ไม่โกหก** ไม่มี data leak ระหว่าง train/val

⚠️ ดังนั้น **ค่า accuracy บอกแค่ว่าโมเดลเรียนสิ่งที่คุณสอนได้ ไม่ได้บอกว่า label ของคุณถูก** ต้องสุ่มดูผลทำนายด้วยตาก่อนเชื่อ — แอปเตือนข้อความนี้ในแท็บ Train ทุกครั้งที่เทรนเสร็จ

---

## อ่านอย่างเดียว

เขียนเฉพาะในโฟลเดอร์แอป: `config.json`, `baselines.json`, `positions.json`, `labels.json`, `models/` และไฟล์ CSV ที่ export

`guard.py` บังคับระดับโค้ด — ถ้า path ปลายทางอยู่ใน `Picture`/`Result`/`Recipe`/โฟลเดอร์ `Eqp.cfg` จะปฏิเสธการเขียน ตรวจแล้ว 285,664 ไฟล์ใน `E:\router` ไม่ถูกแตะเลย

---

## โครงสร้าง

```
app.py                  GUI 6 แท็บ (PySide6) - งานหนักรันบน QThread
router_vision/
  config.py             AppConfig, อ่าน PixelSize/flags จาก Eqp.cfg
  machine.py            อ่าน Result CSV, index ภาพ, จับคู่ภาพ<->รอบงาน
  vision.py             วัดร่องกัด (numpy) + วาด overlay
  analysis.py           calibrate() / inspect() + BaselineStore
  toolpath.py           แกะเส้นทางการตัดจาก .rcp
  labeling.py           LabelStore
  model.py              DINOv2 + linear head, train/predict/save/load
  positions.py          ตั้งชื่อตำแหน่งตรวจ
  guard.py              write guard
```

---

## ข้อจำกัด

1. **ยังไม่มีตัวอย่าง defect จริง** — งานที่ Fail ได้ภาพ 0 ใบ (เครื่องหยุดก่อนกล้องทำงาน) ภาพ 141,701 ใบที่มีเป็นของดีทั้งหมด ต้อง label ของเสียเองก่อนโมเดลถึงจะมีประโยชน์
2. **เครื่องปิดกล้องไปแล้ว** — `EnableAfterCuttingTakePicture = False` ตั้งแต่ 2026-08-11 แท็บ Dashboard จะไม่มีภาพใหม่จนกว่าจะเปิดกลับ (แอปเตือนให้ในแท็บ Config)
3. **ข้อมูลไม่ระบุ sub-board** — CSV มี SN ใบเดียวต่อแผ่นแม่ ต้องตั้งชื่อตำแหน่งเองเพื่อให้รายงานอ่านออก
4. **ยังไม่เชื่อม Cut Path เข้ากับภาพ** — รู้พิกัดร่องจาก recipe และรู้ว่าภาพลำดับ N คือจุดเดิมเสมอ แต่ยังไม่มีข้อมูลตำแหน่งกล้องต่อภาพ จึง map พิกัด mm เข้ากับ index ภาพแบบอัตโนมัติไม่ได้
