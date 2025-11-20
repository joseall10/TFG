import pandas as pd
from pathlib import Path
import re
import zipfile
import tempfile
import shutil

# ========= CONFIG =========
ROOT_DIR = Path("Test")  # <-- cámbiala
ZIP_PATH = None  # o Path("/mnt/data/Test.zip")
MERGE_COLS = ["dataset", "prop", "method"]

# ========= AUX =========
def try_parse_float(name: str):
    try:
        return float(name.replace(",", "."))
    except Exception:
        return None

def parse_metadata(xlsx_path: Path):
    dataset_dir = xlsx_path.parent.name
    method_dir  = xlsx_path.parent.parent.name
    lvl3 = xlsx_path.parent.parent.parent.name
    lvl4 = xlsx_path.parent.parent.parent.parent.name if xlsx_path.parent.parent.parent.parent else None

    prop = try_parse_float(lvl3)
    if prop is not None:
        experiment = lvl4
    else:
        if lvl4 and lvl4.startswith("experimento"):
            experiment = f"{lvl4}/{lvl3}"
        else:
            experiment = lvl3

    dataset_name = re.sub(r"\.csv$", "", dataset_dir)
    return {"experiment": experiment, "prop": prop, "method": method_dir, "dataset": dataset_name}

def sanitize_sheet(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))[:31] or "sheet"

def merge_equal_cells(workbook, worksheet, df: pd.DataFrame, header_row=0, start_col=0, merge_cols=None):
    """Une celdas contiguas iguales en las columnas especificadas."""
    from xlsxwriter.utility import xl_rowcol_to_cell

    if not merge_cols:
        return

    fmt = workbook.add_format({"valign": "vcenter", "align": "left"})
    n = len(df)
    data_start_row = header_row + 1
    col_index = {col: idx for idx, col in enumerate(df.columns)}

    for col_name in merge_cols:
        if col_name not in col_index:
            continue
        c = start_col + col_index[col_name]
        r0 = 0
        while r0 < n:
            val = df.iloc[r0, col_index[col_name]]
            r1 = r0 + 1
            while r1 < n and df.iloc[r1, col_index[col_name]] == val:
                r1 += 1
            excel_r0 = data_start_row + r0
            excel_r1 = data_start_row + r1 - 1
            if excel_r1 > excel_r0:
                top_left = xl_rowcol_to_cell(excel_r0, c)
                bottom_right = xl_rowcol_to_cell(excel_r1, c)
                worksheet.merge_range(f"{top_left}:{bottom_right}", "" if pd.isna(val) else val, fmt)
            r0 = r1

# ========= CARGA =========
tmp_dir = None
if ZIP_PATH:
    tmp_dir = Path(tempfile.mkdtemp(prefix="exp_metrics_"))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(tmp_dir)
    entries = [p for p in tmp_dir.iterdir() if p.is_dir()]
    ROOT_DIR = entries[0] if len(entries) == 1 else tmp_dir

ROOT_DIR = Path(ROOT_DIR).resolve()
xlsx_files = list(ROOT_DIR.rglob("metricas_por_k.xlsx"))
if not xlsx_files:
    raise FileNotFoundError(f"No se encontraron 'metricas_por_k.xlsx' en {ROOT_DIR}")

# ========= LECTURA Y TABLA MÉTRICAS =========
rows = []
for x in xlsx_files:
    meta = parse_metadata(x)
    try:
        df = pd.read_excel(x)
    except Exception as e:
        print(f"[WARN] No se pudo leer {x}: {e}")
        continue
    for _, r in df.iterrows():
        row = {"experiment": meta["experiment"], "prop": meta["prop"], "method": meta["method"], "dataset": meta["dataset"]}
        for col in df.columns:
            row[col] = r[col]
        rows.append(row)

master = pd.DataFrame(rows)
metric_keep = [c for c in ["accuracy", "f1_score", "roc_auc"] if c in master.columns]
key_cols = [c for c in ["dataset", "prop", "method", "k"] if c in master.columns]
cols_metrics_only = ["experiment"] + key_cols + metric_keep
metrics_only = master[cols_metrics_only].copy()

# ========= EXPORTACIÓN CON MERGES =========
global_path = ROOT_DIR / "metrics_only.xlsx"
with pd.ExcelWriter(global_path, engine="xlsxwriter") as writer:
    workbook = writer.book
    num_fmt = workbook.add_format({"num_format": "0.000"})

    for exp_name, df_exp in metrics_only.groupby("experiment", dropna=False):
        sheet_name = sanitize_sheet(exp_name)
        sort_cols = [c for c in MERGE_COLS if c in df_exp.columns]
        df_exp = df_exp.drop(columns=["experiment"])
        if sort_cols:
            df_exp = df_exp.sort_values(sort_cols, kind="mergesort")

        df_exp.to_excel(writer, sheet_name=sheet_name, index=False)
        ws = writer.sheets[sheet_name]

        # formato columnas numéricas
        for col_name in ["accuracy", "f1_score", "roc_auc"]:
            if col_name in df_exp.columns:
                col_idx = df_exp.columns.get_loc(col_name)
                ws.set_column(col_idx, col_idx, 12, num_fmt)

        # ancho para texto
        for col_name in df_exp.columns:
            col_idx = df_exp.columns.get_loc(col_name)
            if df_exp[col_name].dtype == object:
                ws.set_column(col_idx, col_idx, 18)

        # merge de celdas iguales
        merge_equal_cells(workbook, ws, df_exp, header_row=0, start_col=0, merge_cols=MERGE_COLS)

print(f"✅ Hecho: {global_path}")

if tmp_dir and tmp_dir.exists():
    shutil.rmtree(tmp_dir)

