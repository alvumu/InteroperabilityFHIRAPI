import pandas as pd


df = pd.read_csv("cluster_results.csv")

# OPCIÓN B: si lo tienes en un fichero CSV, comenta la línea anterior y descomenta esta:
# df = pd.read_csv("tu_fichero.csv")

# ----------------------------------------------------
# 2. PREPROCESADO: convertir listas de resources a sets
# ----------------------------------------------------
def parse_list(s):
    """
    Convierte 'Observation;Condition;Patient' -> {'Observation','Condition','Patient'}
    Maneja nulos y espacios.
    """
    if pd.isna(s):
        return set()
    return {x.strip() for x in str(s).split(";") if x.strip()}

df["pred_set"] = df["fhir_resources"].apply(parse_list)
df["gold_set"] = df["correct_fhir_resource"].apply(parse_list)

# ----------------------------------------------------
# 3. MÉTRICAS POR FILA
# ----------------------------------------------------
# Acierto estricto: todas las gold dentro de las predicciones
df["strict_correct"] = df.apply(
    lambda r: r["gold_set"].issubset(r["pred_set"]) and len(r["gold_set"]) > 0,
    axis=1
)

# Acierto top-k / parcial: hay al menos una resource correcta en la lista
df["any_overlap"] = df.apply(
    lambda r: len(r["gold_set"].intersection(r["pred_set"])) > 0,
    axis=1
)

# Casos "parcial solo": acierta alguna, pero no todas
df["partial_only"] = df["any_overlap"] & ~df["strict_correct"]

# ----------------------------------------------------
# 4. MÉTRICAS GLOBALES
# ----------------------------------------------------
total = len(df)
strict_n = df["strict_correct"].sum()
topk_n = df["any_overlap"].sum()
partial_only_n = df["partial_only"].sum()
fail_n = total - topk_n

print("=== MÉTRICAS GLOBALES ===")
print(f"Total filas: {total}")
print(f"Aciertos estrictos: {strict_n} ({strict_n/total*100:.1f} %)")
print(f"Aciertos top-k (al menos una correcta): {topk_n} ({topk_n/total*100:.1f} %)")
print(f"  de los cuales 'parciales solo': {partial_only_n} ({partial_only_n/total*100:.1f} %)")
print(f"Fallos totales (sin ninguna correcta): {fail_n} ({fail_n/total*100:.1f} %)")
print()

# ----------------------------------------------------
# 5. MÉTRICAS POR CLUSTER
# ----------------------------------------------------
cluster_stats = (
    df.groupby("cluster")
      .agg(
          total=("attribute_name", "count"),
          strict_correct=("strict_correct", "sum"),
          topk_correct=("any_overlap", "sum")
      )
)

cluster_stats["strict_pct"] = cluster_stats["strict_correct"] / cluster_stats["total"] * 100
cluster_stats["topk_pct"] = cluster_stats["topk_correct"] / cluster_stats["total"] * 100

print("=== MÉTRICAS POR CLUSTER ===")
print(cluster_stats.to_string(float_format=lambda x: f"{x:.1f}"))
print()

# ----------------------------------------------------
# 6. OPCIONAL: listar errores para depuración
# ----------------------------------------------------
# Filas sin ninguna resource correcta
errors = df[~df["any_overlap"]][
    ["attribute_name", "cluster", "fhir_resources", "correct_fhir_resource"]
]
print("=== ERRORES (sin ninguna resource correcta) ===")
print(errors.to_string(index=False))
print()

# Filas con acierto parcial (alguna correcta pero no todas)
partial = df[df["partial_only"]][
    ["attribute_name", "cluster", "fhir_resources", "correct_fhir_resource"]
]
print("=== ACIERTOS PARCIALES (top-k pero no estrictos) ===")
print(partial.to_string(index=False))