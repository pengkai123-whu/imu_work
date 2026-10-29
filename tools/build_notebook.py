"""Regenerate the Colab/local notebook with plain Python cells."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(text): cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(keepends=True)))
def code(text): cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=text.strip().splitlines(keepends=True)))

md('''# MEIO-YT：本地 spk / Colab GPU 共用笔记本

基于作者公开训练模块和 LLIO 残差模块补齐的 YT 适配版本，非论文数值的严格复现。

按顺序执行。Colab 选择“运行时 → 更改运行时类型 → GPU”，或 VS Code 选择本地 spk kernel。
本地 kernel 和 Colab hosted runtime 是二选一，VS Code 的本地文件不会自动传到 Colab。

Colab 准备：把随附 meio_yt_colab.zip 放到 Drive 的 MyDrive/MEIO/，数据可放同目录，或下面选择官方直下。
缓存放云端本地磁盘，checkpoint 每30秒及命令结束同步到 Drive。数据约15.14GB，缓存约2.47GB，建议空闲25GB以上。
''')
code(r'''
# 1. 运行位置和GPU：Colab用云GPU，本地用当前spk kernel的GPU。
import os, sys, platform, subprocess, importlib.util
from pathlib import Path
RUNTIME = "auto"  # auto / colab / local
IS_COLAB = (bool(os.environ.get("COLAB_RELEASE_TAG")) or Path("/content").exists()) if RUNTIME == "auto" else RUNTIME == "colab"
print("Python:", sys.executable, "Platform:", platform.platform(), "Colab:", IS_COLAB)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if importlib.util.find_spec("torch") is None:
    raise RuntimeError("未安装PyTorch：本地请选择spk，Colab请选择标准GPU runtime")
import torch
print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available(): raise RuntimeError("没有GPU：Colab启用GPU，本地选择spk kernel")
print("GPU:", torch.cuda.get_device_name(0), "VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2))
subprocess.run(["nvidia-smi"], check=False)
''')
code(r'''
# 2. 定位代码。Colab从Drive的小代码ZIP解压，不上传大数据到Git。
import json, zipfile, shutil
if IS_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_ROOT = Path("/content/drive/MyDrive/MEIO")  # 可修改
    ARCHIVE = DRIVE_ROOT / "meio_yt_colab.zip"
    CODE_DIR = Path("/content/meio_yt")
    if not ARCHIVE.exists(): raise FileNotFoundError(f"请把代码ZIP放到 {ARCHIVE}")
    with zipfile.ZipFile(ARCHIVE) as z:
        for member in z.infolist():
            resolved = (Path("/content") / member.filename).resolve()
            if Path("/content").resolve() not in resolved.parents: raise ValueError("不安全ZIP路径")
        z.extractall("/content")
else:
    candidates = [Path.cwd(), Path.cwd()/"meio_yt", Path(r"C:\Users\19510\Desktop\work_space\imu_work\meio_yt")]
    CODE_DIR = next((p.resolve() for p in candidates if (p/"run.py").exists()), None)
    if CODE_DIR is None: raise FileNotFoundError("把CODE_DIR设为meio_yt目录")
os.chdir(CODE_DIR)
subprocess.run([sys.executable, "-m", "pip", "install", "h5py>=3.9", "numpy>=1.24"], check=True)
print("Code:", CODE_DIR)
''')
code(r'''
# 3. 数据和持久化路径。Colab不能访问你电脑的C:路径。
DATA_SOURCE = "drive"  # Colab: drive / official；local模式忽略
RUN_NAME = "yt_full_v1"  # 新实验换名称；相同名称恢复last.pt
if IS_COLAB:
    H5_PATH = Path("/content/YT_server_200Hz.h5")
    CACHE_DIR = Path("/content/yt_cache")
    RUN_DIR = Path("/content/runs") / RUN_NAME
    PERSIST_DIR = DRIVE_ROOT / "runs" / RUN_NAME
    if not H5_PATH.exists():
        if shutil.disk_usage("/content").free < 25*1024**3: raise RuntimeError("云端空闲磁盘不足25GiB")
        temporary = H5_PATH.with_suffix(".download")
        if DATA_SOURCE == "drive":
            source = DRIVE_ROOT / H5_PATH.name
            if not source.exists(): raise FileNotFoundError(source)
            shutil.copyfile(source, temporary)
        elif DATA_SOURCE == "official":
            import urllib.request
            url = "https://zenodo.org/api/records/18149105/files/YT_server_200Hz.h5/content"
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                total = 0
                while True:
                    chunk = response.read(16*1024*1024)
                    if not chunk: break
                    output.write(chunk); total += len(chunk)
                    if total % (1024**3) < len(chunk): print("Downloaded GiB:", round(total/1024**3, 2))
        else: raise ValueError("DATA_SOURCE应为drive或official")
        if temporary.stat().st_size != 15138253808: raise RuntimeError("下载/复制不完整，请重新执行")
        temporary.replace(H5_PATH)
else:
    H5_PATH = CODE_DIR.parent / "YT_server_200Hz.h5"
    CACHE_DIR = CODE_DIR / "cache" / "yt"
    RUN_DIR = CODE_DIR / "runs" / RUN_NAME
    PERSIST_DIR = None
if not H5_PATH.exists(): raise FileNotFoundError(H5_PATH)
if H5_PATH.stat().st_size != 15138253808: raise RuntimeError("H5大小与已核对的官方版本不同，请检查文件")
# 同步启动前恢复整个运行目录，避免新日志覆盖Drive历史。
RUN_DIR.mkdir(parents=True, exist_ok=True)
if PERSIST_DIR is not None and PERSIST_DIR.exists():
    for source in PERSIST_DIR.iterdir():
        if source.is_file() and source.suffix in {".pt", ".json", ".jsonl"}:
            destination = RUN_DIR/source.name
            if not destination.exists(): shutil.copy2(source, destination)
print("Data:", H5_PATH, "Cache:", CACHE_DIR, "Run:", RUN_DIR)
''')
code(r'''
# 4. 16GB Colab从2048起步，试探至8192，按实测吞吐和显存留量选择。
# 大batch不保证更快；不自动放大学习率。probe不包含H5读取吞吐。
cfg = json.loads((CODE_DIR/"configs/yt.json").read_text(encoding="utf-8"))
cfg["data"].update(h5=str(H5_PATH), cache=str(CACHE_DIR))
cfg["train"].update(out=str(RUN_DIR), device="cuda", batch=2048, auto_batch=True, max_batch=8192, workers=0)
CONFIG_PATH = CODE_DIR / "configs" / "active.json"
CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(json.dumps(cfg, indent=2))
''')
code(r'''
# 5. 命令运行器：Colab自动保存到Drive，本地不需要同步。
import threading, time
SYNC_LOCK = threading.Lock()
RUN_DIR.mkdir(parents=True, exist_ok=True)
def sync_outputs():
    if PERSIST_DIR is None: return
    with SYNC_LOCK:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        for source in RUN_DIR.iterdir():
            if source.is_file() and source.suffix in {".pt", ".json", ".jsonl"}:
                target = PERSIST_DIR/source.name
                if target.exists() and target.stat().st_size == source.stat().st_size and target.stat().st_mtime >= source.stat().st_mtime: continue
                temp = target.with_name(target.name+".sync")
                shutil.copy2(source, temp); temp.replace(target)

def run(*arguments):
    stop = threading.Event(); failures = []
    def worker():
        while not stop.wait(30):
            try: sync_outputs()
            except Exception as exc: failures.append(str(exc)); print("Drive sync warning:", exc)
    thread = threading.Thread(target=worker, daemon=True); thread.start()
    process = subprocess.Popen([sys.executable, "-u", *map(str, arguments)], cwd=CODE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    try:
        for line in process.stdout: print(line, end="", flush=True)
        returncode = process.wait()
    except KeyboardInterrupt:
        process.terminate(); process.wait(); raise
    finally:
        stop.set(); thread.join(timeout=35); sync_outputs()
    if returncode: raise RuntimeError(f"命令失败，返回码{returncode}")
    if failures: print("出现过同步警告；最终同步已完成，请检查Drive文件。")
''')
md('''## 先检查，再训练

下一单元格只对真实数据小切片跑4轮、每轮3个batch，验证GPU、恢复、校准和评估；不是完整精度实验。
首次可运行，之后可跳过。smoke输出使用时间戳，避免覆盖。
''')
code(r'''
run("-m", "unittest", "discover", "-s", "tests", "-v")
SMOKE_NAME = "notebook_smoke_" + time.strftime("%Y%m%d_%H%M%S")
run("tools/smoke_check.py", "--device", "cuda", "--h5", H5_PATH, "--out", CODE_DIR/"runs"/SMOKE_NAME)
''')
code(r'''
# 6. 完整缓存。仅从train估计统计量，H5的para不参与。此步CPU处理。
run("run.py", "prepare", "--config", CONFIG_PATH)
''')
code(r'''
# 7. 恢复或开始训练。云端断线后重跑路径/缓存单元格，再运行这里。
resume = RUN_DIR/"last.pt"
args = ["run.py", "train", "--config", CONFIG_PATH]
if resume.exists(): args += ["--resume", resume]
run(*args)
''')
code(r'''
# 8. 验证集选择阈值。best.pt只从已进入NLL阶段的模型中选择。
CHECKPOINT = RUN_DIR/"best.pt"
POLICY = RUN_DIR/"policy.json"
run("run.py", "calibrate", "--checkpoint", CHECKPOINT, "--cache", CACHE_DIR, "--out", POLICY, "--batch", 512)
''')
code(r'''
# 9. 测试集只用于冻结后的评估。
for name, extra in [("final", []), ("early", ["--policy", POLICY])]:
    run("run.py", "evaluate", "--checkpoint", CHECKPOINT, "--cache", CACHE_DIR, "--out", RUN_DIR/(name+"_test.json"), "--batch", 512, *extra)
''')
code(r'''
# 10. 1秒不重叠位移累加，XYZ/XY ATE、30秒RTE；非论文未公开的20Hz拼接。
for name in ["final", "early"]:
    report = json.loads((RUN_DIR/(name+"_test.json")).read_text())
    print(name, "mode", report["mode"], "displacement RMSE", report["displacement_rmse"], "macro", report["macro"], "exits", report["exit_histogram"])
sync_outputs()
''')
md('''## 本地与云端协同

- 本地spk调试/分析，Colab跑完整训练；同一代码、配置和checkpoint通用。
- 接续：拷贝整个运行目录到另一端，重建相同全量缓存，用last.pt恢复。smoke子集checkpoint不能接续全量实验。
- 缓存标识绑定所用记录、统计量、采样率和数据限制；模型和关键训练设置不匹配会拒绝恢复。可修改路径、batch，或增加纯NLL轮数。
- 两端可同时跑不同RUN_NAME的实验，不可同时写同一运行目录。跨公网GPU不做梯度同步。
- Drive挂载需要你在Colab交互授权。交付未替你登录或连接Colab。
- 硬断线会丢失最近完整epoch checkpoint之后的进度；每30秒同步不能保存未结束的epoch。
''')
notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python'),colab=dict(name='MEIO_YT_Colab.ipynb',provenance=[]),accelerator='GPU'),nbformat=4,nbformat_minor=5)
for i,cell in enumerate(cells):cell['id']=f'cell-{i:02d}'
(ROOT/'MEIO_YT_Colab.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
print('Wrote',ROOT/'MEIO_YT_Colab.ipynb')
