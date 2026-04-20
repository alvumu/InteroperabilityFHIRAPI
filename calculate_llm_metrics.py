#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Set


# =========================
# CONFIG (EDITA AQUÍ)
# =========================
GROUND_TRUTH_CSV_PATH = "correct_attributes_resources.csv"
LLM_JSON_PATH = "llm_output/GPT/GPT_llm_results_iter_10_temp_1_clusters_rag_mimic_prueba_config_desc.json"

TOPK = 3  # k = 3

WRITE_DETAILS_CSV = True
DETAILS_OUT_CSV_PATH = "llm_output/GPT/results/results_iter_10_temp_1_clusters.csv"

# Si True: atributos sin GT cuentan como fallo.
# Si False: se excluyen del denominador (recomendado).
COUNT_MISSING_GT_AS_MISS = False

# Tiers (para breakdown/figuras) derivados de hierarchical similarity (schema-level)
TIER_NEAR = 0.75
TIER_MODERATE = 0.50
# =========================


# -------------------------
# Normalización / paths
# -------------------------
def norm_path(p: str) -> str:
    """Lowercase + trim + remove all whitespace."""
    p = (p or "").strip()
    p = re.sub(r"\s+", "", p)
    return p.lower()


def split_multi(s: str) -> List[str]:
    """Split by ';' and strip."""
    if s is None:
        return []
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return []
    return [x.strip() for x in s.split(";") if x.strip()]


def split_segments(path: str) -> List[str]:
    return [seg for seg in path.split(".") if seg]


def seg_base(seg: str) -> str:
    # for "extension:ethnicity" treat base as "extension"
    return seg.split(":", 1)[0]


def canon_path(path: str) -> str:
    """
    Canonicaliza por segmentos base (para tolerar qualifiers tipo extension:ethnicity).
    Ej: Patient.extension:ethnicity -> patient.extension
    """
    path = norm_path(path)
    if not path:
        return ""
    segs = [seg_base(s) for s in split_segments(path)]
    return ".".join(segs)


def resource_of(path: str) -> str:
    parts = split_segments(path)
    return parts[0] if parts else ""


def lcp_len(a: str, b: str) -> int:
    """Longest Common Prefix length por segmentos (ya canonicalizados)."""
    sa = split_segments(a)
    sb = split_segments(b)
    n = min(len(sa), len(sb))
    i = 0
    while i < n and sa[i] == sb[i]:
        i += 1
    return i


def lcp_similarity(a: str, b: str) -> float:
    """
    Hierarchical similarity schema-level basada en LCP, normalizada por la profundidad máxima.
    sim = lcp / max(depth(a), depth(b))
    """
    if not a or not b:
        return 0.0
    sa = split_segments(a)
    sb = split_segments(b)
    if not sa or not sb:
        return 0.0
    lcp = lcp_len(a, b)
    denom = max(len(sa), len(sb))
    return (lcp / denom) if denom else 0.0


def dedupe_preserve_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


# -------------------------
# Ground truth / predictions
# -------------------------
def load_ground_truth(csv_path: str) -> Dict[str, List[str]]:
    """
    Se asume que el GT YA está a nivel schema (como indicas).
    """
    gt: Dict[str, List[str]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"attribute_name", "correct_fhir_attribute"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        for row in reader:
            attr = (row.get("attribute_name") or "").strip()
            correct_attr = row.get("correct_fhir_attribute")
            gt[attr] = [canon_path(x) for x in split_multi(correct_attr)]
    return gt


def _extract_mappings_container(cluster_obj: Any) -> List[Dict[str, Any]]:
    """
    Soporta dos variantes comunes:
      A) {"LLM_Mappings": {"mappings": [ ... ]}}
      B) {"LLM_Mappings": [ ... ]}  (lista directa)
    """
    llm_mappings = cluster_obj.get("LLM_Mappings", None)
    if isinstance(llm_mappings, dict):
        mappings = llm_mappings.get("mappings", [])
        return mappings if isinstance(mappings, list) else []
    if isinstance(llm_mappings, list):
        return llm_mappings
    return []


def load_llm_json(json_path: str) -> Dict[str, List[str]]:
    pred: Dict[str, List[str]] = {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("LLM JSON must be a list of cluster objects.")

    for cluster_obj in data:
        if not isinstance(cluster_obj, dict):
            continue

        mappings = _extract_mappings_container(cluster_obj)

        for m in mappings:
            if not isinstance(m, dict):
                continue

            a = (m.get("table_attribute_name") or "").strip()
            lst = m.get("fhir_attribute_name") or []

            cleaned: List[str] = []
            for x in lst:
                if not isinstance(x, str):
                    continue
                if "no additional attribute found" in x.lower():
                    continue
                cleaned.append(canon_path(x))

            if a:
                pred[a] = cleaned

    return pred


# -------------------------
# Schema-level projection
# -------------------------
def build_schema_vocab_from_gt(gt: Dict[str, List[str]]) -> Set[str]:
    """
    Como indicas que ya truncaste GT al nivel schema, usamos el vocabulario schema
    como la unión de todos los paths presentes en GT.
    """
    vocab: Set[str] = set()
    for paths in gt.values():
        for p in paths:
            if p:
                vocab.add(p)
    return vocab


def ancestors(path: str) -> List[str]:
    """
    Devuelve lista de ancestros (de más específico a menos).
    Ej: encounter.actualperiod.start ->
        ['encounter.actualperiod.start', 'encounter.actualperiod', 'encounter']
    """
    segs = split_segments(path)
    out = []
    for i in range(len(segs), 0, -1):
        out.append(".".join(segs[:i]))
    return out


def project_to_schema(path: str, schema_vocab: Set[str]) -> str:
    """
    Proyecta una predicción (posiblemente más específica) al nodo más específico disponible
    en el vocabulario schema.
    - Si path ya está en schema_vocab -> se queda igual.
    - Si no, busca el ancestro más profundo que esté en schema_vocab.
    - Si no encuentra ninguno -> deja el path tal cual (no habrá match exacto).
    """
    if not path:
        return ""
    path = canon_path(path)
    if not path:
        return ""
    if path in schema_vocab:
        return path
    for anc in ancestors(path)[1:]:  # excluye el propio path (ya comprobado)
        if anc in schema_vocab:
            return anc
    return path


# -------------------------
# Ranking metrics
# -------------------------
def reciprocal_rank_at_k(pr_paths: List[str], gt_paths: List[str], k: int) -> float:
    """
    Reciprocal Rank@k para exact-match (schema-level).
    1/rank si alguno de los gold aparece en top-k, 0 si no.
    """
    if not gt_paths:
        return 0.0
    topk = pr_paths[:max(0, k)]
    gt_set = set(gt_paths)
    for i, p in enumerate(topk, start=1):
        if p in gt_set:
            return 1.0 / i
    return 0.0


def best_similarity_over_preds(
    pred_paths: List[str],
    gt_paths: List[str]
) -> Tuple[float, str, str]:
    """
    Devuelve (best_sim, best_pred, best_gt) maximizando LCP similarity.
    """
    best_sim = 0.0
    best_pred = ""
    best_gt = ""
    for p in pred_paths:
        for g in gt_paths:
            s = lcp_similarity(p, g)
            if s > best_sim:
                best_sim, best_pred, best_gt = s, p, g
    return best_sim, best_pred, best_gt


def tier_from_similarity(sim: float, same_resource: bool) -> str:
    """
    Tiers para breakdown/figura (derivados de sim).
    """
    if sim >= 1.0:
        return "exact"
    if sim >= TIER_NEAR:
        return "near"
    if sim >= TIER_MODERATE:
        return "moderate"
    if sim > 0.0 and same_resource:
        return "far_same_resource"
    if sim > 0.0 and not same_resource:
        # Esto es raro con LCP porque si no coincide recurso, sim suele ser 0;
        # se deja por completitud.
        return "far_diff_resource"
    return "wrong_resource"


def pct(x: int, d: int) -> float:
    return round((x / d) * 100.0, 2) if d else 0.0


def main():
    gt = load_ground_truth(GROUND_TRUTH_CSV_PATH)
    pred = load_llm_json(LLM_JSON_PATH)

    schema_vocab = build_schema_vocab_from_gt(gt)

    all_attrs = sorted(set(gt.keys()) | set(pred.keys()))

    rows_out: List[Dict[str, Any]] = []

    total = 0
    evaluable_n = 0

    exact_top1_n = 0
    exact_any_n = 0

    rr_sum = 0.0  # MRR@k
    sim_top1_sum = 0.0
    sim_bestk_sum = 0.0

    tier_counts: Dict[str, int] = {}

    for a in all_attrs:
        total += 1

        gt_paths = gt.get(a, [])
        pr_paths_full = pred.get(a, [])
        pr_paths_topk = pr_paths_full[:max(0, TOPK)]

        # Proyección schema-level para permitir "over-specification" como acierto
        pr_proj_topk = [project_to_schema(p, schema_vocab) for p in pr_paths_topk]
        pr_proj_topk = dedupe_preserve_order(pr_proj_topk)

        evaluable = len(gt_paths) > 0
        if (not evaluable) and COUNT_MISSING_GT_AS_MISS:
            evaluable = True

        if evaluable:
            evaluable_n += 1

        # -------------------------
        # Exact@1 / Exact@k (schema-level)
        # -------------------------
        top1 = pr_proj_topk[0] if pr_proj_topk else ""
        exact_top1 = evaluable and (top1 in gt_paths)
        exact_any = evaluable and any(p in set(gt_paths) for p in pr_proj_topk)

        if exact_top1:
            exact_top1_n += 1
        if exact_any:
            exact_any_n += 1

        # -------------------------
        # MRR@k (schema-level)
        # -------------------------
        rr = reciprocal_rank_at_k(pr_proj_topk, gt_paths, TOPK) if evaluable else 0.0
        rr_sum += rr

        # -------------------------
        # Hierarchical similarity (schema-level)
        # - sim_top1: similitud del top1 vs best gold
        # - sim_bestk: mejor similitud dentro del top-k vs best gold
        # -------------------------
        if evaluable and gt_paths and top1:
            sim_top1, best_pred_top1, best_gt_top1 = best_similarity_over_preds([top1], gt_paths)
        else:
            sim_top1, best_pred_top1, best_gt_top1 = 0.0, "", ""

        if evaluable and gt_paths and pr_proj_topk:
            sim_bestk, best_pred_bestk, best_gt_bestk = best_similarity_over_preds(pr_proj_topk, gt_paths)
        else:
            sim_bestk, best_pred_bestk, best_gt_bestk = 0.0, "", ""

        sim_top1_sum += sim_top1
        sim_bestk_sum += sim_bestk

        # Tiers: usa best-of-k (representa mejor "casi aciertos" con top-k)
        gt_resources = {resource_of(g) for g in gt_paths}
        same_resource = bool(best_pred_bestk) and (resource_of(best_pred_bestk) in gt_resources)
        tier = "no_gt" if not evaluable else tier_from_similarity(sim_bestk, same_resource)
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        rows_out.append({
            "attribute": a,
            "gt_schema": ";".join(gt_paths),
            "pred_raw_topk": ";".join(pr_paths_topk),
            "pred_schema_topk": ";".join(pr_proj_topk),

            "exact@1_schema": int(exact_top1),
            f"exact@{TOPK}_schema": int(exact_any),
            f"rr@{TOPK}_schema": rr,

            "sim_top1_schema": round(sim_top1, 6),
            f"sim_best@{TOPK}_schema": round(sim_bestk, 6),

            "best_gt_for_top1": best_gt_top1,
            "best_gt_for_bestk": best_gt_bestk,
            "best_pred_bestk": best_pred_bestk,

            "tier_bestk": tier,
        })

    # =========================
    # SUMMARY
    # =========================
    print("=== Evaluation Summary (Schema-level) ===")
    print(f"Ground truth CSV: {GROUND_TRUTH_CSV_PATH}")
    print(f"LLM JSON:         {LLM_JSON_PATH}")
    print(f"TOPK:             {TOPK}")
    print(f"COUNT_MISSING_GT_AS_MISS: {COUNT_MISSING_GT_AS_MISS}")
    print("Evaluation is schema-level (GT assumed already truncated to schema granularity).")
    print("Predictions are projected to the nearest schema ancestor (over-specification counts as correct).")
    print()

    print(f"Total attributes seen (GT ∪ LLM): {total}")
    print(f"Evaluated attributes: {evaluable_n}")

    print(f"Exact@1 (schema):     {exact_top1_n}/{evaluable_n} = {pct(exact_top1_n, evaluable_n)}%")
    print(f"Exact@{TOPK} (schema): {exact_any_n}/{evaluable_n} = {pct(exact_any_n, evaluable_n)}%")

    mrr = (rr_sum / evaluable_n) if evaluable_n else 0.0
    print(f"MRR@{TOPK} (schema):    {round(mrr, 6)}")

    mean_sim_top1 = (sim_top1_sum / evaluable_n) if evaluable_n else 0.0
    mean_sim_bestk = (sim_bestk_sum / evaluable_n) if evaluable_n else 0.0
    print(f"Mean HierSim (top1, schema):        {round(mean_sim_top1, 6)}")
    print(f"Mean HierSim (best-of-{TOPK}, schema): {round(mean_sim_bestk, 6)}")

    print("\nTier breakdown (best-of-k, schema-level similarity):")
    for k in ["exact", "near", "moderate", "far_same_resource", "wrong_resource", "no_gt"]:
        if k in tier_counts:
            denom = evaluable_n if evaluable_n else 1
            # no_gt no debería dividirse por evaluable, pero lo mantenemos consistente:
            # si COUNT_MISSING_GT_AS_MISS=False, no_gt será parte del total no evaluable.
            if k == "no_gt":
                denom = total if total else 1
            print(f"  - {k}: {tier_counts[k]} ({pct(tier_counts[k], denom)}%)")

    if WRITE_DETAILS_CSV and rows_out:
        with open(DETAILS_OUT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
            fieldnames = list(rows_out[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nWrote detailed results to: {DETAILS_OUT_CSV_PATH}")


if __name__ == "__main__":
    main()
