import os
import pandas as pd

# ======== CONFIGURACIÓN ========
BASE_DIR = "experimento_real"       # carpeta raíz
OUTPUT_TEX = "metricas_reales.tex"  # archivo de salida

OBJETOS_POR_TABLA = 5
METRICS = ["accuracy", "f1_score", "roc_auc"]
METRIC_LABELS = {"accuracy": "accuracy", "f1_score": "F1", "roc_auc": "ROC"}

METHOD_FOLDERS = {
    "reconstrucción": "Reconstrucción",
    "semilla_aleatoria": "Semilla aleatoria",
    "ZeroR": "ZeroR",
}


# ======== FUNCIONES ========

def read_metrics(path_xlsx):
    """Lee el Excel y devuelve el promedio de métricas como fila única."""
    if not os.path.exists(path_xlsx):
        return None
    df = pd.read_excel(path_xlsx)
    cols = [c for c in METRICS if c in df.columns]
    if not cols:
        return None
    # promedio de cada columna
    df_mean = df[cols].mean()
    return df_mean.to_dict()


def recoger_metricas():
    """
    Devuelve un DataFrame con:
    objeto | método | accuracy | f1_score | roc_auc
    """
    registros = []

    for objeto in sorted(os.listdir(BASE_DIR)):
        obj_path = os.path.join(BASE_DIR, objeto)
        if not os.path.isdir(obj_path):
            continue

        for folder, metodo_pretty in METHOD_FOLDERS.items():
            xlsx_path = os.path.join(obj_path, folder, "metricas_por_k.xlsx")
            metrics = read_metrics(xlsx_path)
            if metrics is None:
                continue
            row = {"objeto": objeto, "método": metodo_pretty}
            row.update(metrics)
            registros.append(row)

    df = pd.DataFrame(registros)
    return df


def aplicar_formato(df):
    """Formatea todos los números con tres decimales fijos (ej: 0.930)."""
    for m in METRICS:
        if m in df.columns:
            df[m] = df[m].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else ""
            )
    return df


def construir_tablas(df):
    """Genera las tablas en grupos de 5 objetos por tabla."""
    tablas = []
    objetos = sorted(df["objeto"].unique())

    for i in range(0, len(objetos), OBJETOS_POR_TABLA):
        chunk = objetos[i:i + OBJETOS_POR_TABLA]

        # Cabecera 1: nombres de objetos
        header1 = " & " + " & ".join(
            [f"\\multicolumn{{{len(METRICS)}}}{{c}}{{{obj.title()}}}" for obj in chunk]
        ) + " \\\\ \\hline\n"

        # Cabecera 2: nombres de métricas
        header2 = "Método & " + " & ".join(
            [f"{METRIC_LABELS[m]}" for obj in chunk for m in METRICS]
        ) + " \\\\ \\hline\n"

        # Filas
        filas = ""
        for metodo in METHOD_FOLDERS.values():
            parte = [metodo]
            for obj in chunk:
                row = df[(df["objeto"] == obj) & (df["método"] == metodo)]
                if row.empty:
                    parte += ["", "", ""]
                else:
                    parte += [row.iloc[0][m] if m in row.columns else "" for m in METRICS]
            filas += " & ".join(parte) + " \\\\\n"

        num_cols = 1 + len(chunk) * len(METRICS)
        col_format = "l" + "c" * (num_cols - 1)

        tabla = (
            "\\begin{table}[h!]\n"
            "\\centering\n"
            f"\\caption{{Métricas Reconstrucción, Semilla aleatoria y ZeroR ({', '.join(chunk)})}}\n"
            f"\\begin{{tabular}}{{{col_format}}}\n"
            "\\hline\n"
            f"{header1}"
            f"{header2}"
            f"{filas}"
            "\\hline\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )
        tablas.append(tabla)

    return tablas


# ======== EJECUCIÓN PRINCIPAL ========

df = recoger_metricas()
df = aplicar_formato(df)

with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
    f.write("\\documentclass{article}\n")
    f.write("\\usepackage[margin=2cm]{geometry}\n")
    f.write("\\usepackage{graphicx}\n")
    f.write("\\usepackage{booktabs}\n")
    f.write("\\begin{document}\n\n")

    tablas = construir_tablas(df)
    for t in tablas:
        f.write(t + "\n\n")

    f.write("\\end{document}\n")

print(f"✅ Archivo LaTeX generado en: {OUTPUT_TEX}")
