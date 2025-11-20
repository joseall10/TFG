import pandas as pd

# ======== CONFIGURACIÓN ========
excel_path = "Test/metrics_only.xlsx"  # <-- cambia esta ruta
latex_file_path = "output_tables.tex"  # <-- salida .tex

# Diccionario con número de decimales por columna
DECIMALES_POR_COLUMNA = {
    "accuracy": 3,
    "f1_score": 3,
    "roc_auc": 3,
    "prop": 2,   # por ejemplo, proporciones con 2 decimales
    "k": 1       # nº de decimales para 'k'
}

# ======== FUNCIONES ========

def formatear_numero(x, decimales):
    """Formatea número sin ceros de más, sin redondear a 0 los valores pequeños."""
    if pd.isna(x) or x == "":
        return ""
    if isinstance(x, (int, float)):
        # Mostrar con el número de decimales deseado
        fmt = f"{{:.{decimales}f}}".format(x)
        # Si es muy pequeño pero no cero (ej: 0.004), no truncarlo
        if float(fmt) == 0 and x != 0:
            fmt = f"{x:.{decimales+4}f}"
        # Eliminar ceros y punto final si sobran
        fmt = fmt.rstrip("0").rstrip(".")
        return fmt
    return x

def aplicar_formato(df: pd.DataFrame, decimales_dict: dict):
    """Aplica formateo por columna (decimales sin ceros de más) y reemplaza NaN por vacío."""
    df = df.fillna("")
    for col, n_dec in decimales_dict.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: formatear_numero(x, n_dec))
    return df

def poner_en_negrita_filas_max_f1(df: pd.DataFrame, col_f1: str = "f1_score") -> pd.DataFrame:
    """
    Envuelve en \\textbf{...} todas las celdas de las filas cuyo f1_score sea máximo.
    Si no existe la columna, no hace nada.
    """
    if col_f1 not in df.columns:
        return df

    # Convertir a numérico (por si ya está formateado como string)
    f1_numeric = pd.to_numeric(df[col_f1].replace("", pd.NA), errors="coerce")
    if f1_numeric.notna().any():
        max_f1 = f1_numeric.max()
        mask = f1_numeric == max_f1
        if mask.any():
            # Envolver en \textbf{...} cada celda de las filas marcadas
            cols = df.columns.tolist()
            for idx in df.index[mask]:
                for c in cols:
                    val = df.at[idx, c]
                    if isinstance(val, str) and val != "":
                        df.at[idx, c] = f"\\textbf{{{val}}}"
                    elif pd.notna(val) and val != "":  # por si quedara algo no-string
                        df.at[idx, c] = f"\\textbf{{{val}}}"
    return df

def df_to_latex(df: pd.DataFrame, sheet_name: str, decimales_dict: dict):
    # 1) Formateo (NaN -> "", decimales por columna, sin ceros de más)
    df = aplicar_formato(df.copy(), decimales_dict)
    # 2) Negrita en filas con f1 máximo
    df = poner_en_negrita_filas_max_f1(df, "f1_score")

    # 3) Generar tabla sin líneas verticales (estilo limpio)
    latex_code = df.to_latex(
        index=False,
        header=True,
        longtable=False,
        escape=False,  # necesario para respetar \textbf{}
        column_format="c" * len(df.columns),
        na_rep=""
    )

    # Reglas horizontales con \hline
    latex_code = (
        latex_code.replace("\\toprule", "\\hline")
                  .replace("\\midrule", "\\hline")
                  .replace("\\bottomrule", "\\hline")
    )

    return f"\\begin{{table}}[h!]\n\\centering\n\\caption{{{sheet_name}}}\n{latex_code}\n\\end{{table}}"

# ======== GENERACIÓN DEL ARCHIVO ========

xls = pd.ExcelFile(excel_path)

with open(latex_file_path, 'w', encoding='utf-8') as f:
    f.write("\\documentclass{article}\n")
    f.write("\\usepackage{graphicx}\n")
    f.write("\\usepackage{longtable}\n")
    f.write("\\usepackage[margin=2cm]{geometry}\n")
    f.write("\\begin{document}\n\n")

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        latex_code = df_to_latex(df, sheet_name, DECIMALES_POR_COLUMNA)
        f.write(latex_code + "\n\n")
    
    f.write("\\end{document}\n")

print(f"✅ Archivo LaTeX guardado en: {latex_file_path}")
