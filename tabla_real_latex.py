import pandas as pd

# ======== CONFIGURACIÓN ========
excel_path = "Test/metrics_only.xlsx"   # <-- tu ruta
latex_file_path = "tabla_real.tex"   # <-- salida .tex

# columnas que siempre van delante (sin 'k')
ID_COLS = ["dataset", "prop", "method"]

# columnas reales del Excel
METRICS = ["accuracy", "f1_score", "roc_auc"]

# nombres que quieres en la tabla
METRIC_LABELS = {
    "accuracy": "accuracy",
    "f1_score": "F1",
    "roc_auc": "ROC",
}

OBJETOS_POR_TABLA = 5

# mapeo de nombres de método
METHOD_RENAME = {
    "embeddings": "MP",
    "tus_embeddings": "FLA",
    "dataset_embeddings_encoder": "ENS",
    "dataset_embeddings_encoder_resnet": "RNE",
}


# ======== FUNCIONES ========

def formatear_numero(x, decimales=3):
    if pd.isna(x) or x == "":
        return ""
    if isinstance(x, (int, float)):
        fmt = f"{{:.{decimales}f}}".format(x)
        if float(fmt) == 0 and x != 0:
            fmt = f"{x:.{decimales+4}f}"
        fmt = fmt.rstrip("0").rstrip(".")
        return fmt
    return x


def aplicar_formato(df: pd.DataFrame):
    df = df.fillna("")
    for col in df.columns:
        for m in METRICS:
            if col.endswith(m):
                df[col] = df[col].apply(lambda x: formatear_numero(x, 3))
    return df


def nombre_objeto_pretty(obj_key: str) -> str:
    return obj_key.replace("_", " ").title()


# ======== LECTURA DE HOJAS REALES ========

xls = pd.ExcelFile(excel_path)
real_sheets = [s for s in xls.sheet_names if s.startswith("experimento_real_")]

wide_df = None

for sheet in real_sheets:
    df = pd.read_excel(xls, sheet_name=sheet)
    obj_key = sheet.replace("experimento_real_", "")

    # renombrar métodos
    if "method" in df.columns:
        df["method"] = df["method"].replace(METHOD_RENAME)

    if wide_df is None:
        wide_df = df[ID_COLS].copy()

    # añadir métricas del objeto
    for m in METRICS:
        wide_df[f"{obj_key}_{m}"] = df[m]

# aplicar formato
wide_df = aplicar_formato(wide_df)


# ======== CONSTRUCCIÓN DE TABLAS ========

def construir_tablas(df: pd.DataFrame):
    tablas = []

    # detectar objetos reales
    objetos = []
    for col in df.columns[len(ID_COLS):]:
        for m in METRICS:
            suf = f"_{m}"
            if col.endswith(suf):
                obj_key = col[:-len(suf)]
                if obj_key not in objetos:
                    objetos.append(obj_key)

    # agrupar de 5 en 5
    for i in range(0, len(objetos), OBJETOS_POR_TABLA):
        chunk = objetos[i:i + OBJETOS_POR_TABLA]

        # columnas de esta tabla
        cols_tabla = ID_COLS.copy()
        for obj in chunk:
            for m in METRICS:
                cols_tabla.append(f"{obj}_{m}")

        subdf = df[cols_tabla]

        num_id = len(ID_COLS)
        num_obj_cols = len(chunk) * len(METRICS)
        col_format = "l" * num_id + "c" * num_obj_cols

        # cabecera 1 (objetos)
        header1 = " & " * (num_id - 1)
        header1 += " & ".join(
            f"\\multicolumn{{{len(METRICS)}}}{{c}}{{{nombre_objeto_pretty(obj)}}}"
            for obj in chunk
        )
        header1 += " \\\\ \\hline\n"

        # cabecera 2 (métricas con etiquetas)
        id_headers = " & ".join(c.replace("_", "\\_") for c in ID_COLS)
        metric_headers = " & ".join(
            METRIC_LABELS[m] for _ in chunk for m in METRICS
        )
        header2 = f"{id_headers} & {metric_headers} \\\\ \\hline\n"

        # filas
        filas = ""
        for _, row in subdf.iterrows():
            parte_id = " & ".join(str(row[c]) for c in ID_COLS)
            valores = []
            for obj in chunk:
                for m in METRICS:
                    col_name = f"{obj}_{m}"
                    valores.append(str(row[col_name]))
            filas += f"{parte_id} & " + " & ".join(valores) + " \\\\\n"

        tabla_latex = (
            "\\begin{table}[h!]\n"
            "\\centering\n"
            f"\\caption{{Resultados objetos reales ({', '.join(nombre_objeto_pretty(o) for o in chunk)})}}\n"
            f"\\begin{{tabular}}{{{col_format}}}\n"
            "\\hline\n"
            f"{header1}"
            f"{header2}"
            f"{filas}"
            "\\hline\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )
        tablas.append(tabla_latex)

    return tablas


# ======== GUARDAR ========

with open(latex_file_path, 'w', encoding='utf-8') as f:
    f.write("\\documentclass{article}\n")
    f.write("\\usepackage{graphicx}\n")
    f.write("\\usepackage{longtable}\n")
    f.write("\\usepackage[margin=2cm]{geometry}\n")
    f.write("\\begin{document}\n\n")

    tablas = construir_tablas(wide_df)
    for t in tablas:
        f.write(t + "\n\n")

    f.write("\\end{document}\n")

print(f"✅ Archivo LaTeX guardado en: {latex_file_path}")
