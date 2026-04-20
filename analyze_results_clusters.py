#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Valida clusters de atributos y recursos FHIR frente a un mapping base.

Uso:
    python validate_clusters.py \
        --mapping-csv mapping.csv \
        --clusters-json clusters.json \
        --output-csv resultados_validacion.csv

- mapping.csv: archivo con columnas "attribute_name,correct_fhir_resource"
- clusters.json: JSON con la estructura de clusters que has mostrado
- output-csv (opcional): guarda el detalle atributo a atributo
"""

import argparse
import csv
import json
from collections import defaultdict
from typing import Dict, Set, List, Any


def load_mapping(csv_path: str) -> Dict[str, Set[str]]:
    """
    Carga el fichero CSV base:
    attribute_name,correct_fhir_resource

    Si hay varios recursos, se asume que vienen separados por ';'
    """
    mapping: Dict[str, Set[str]] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            attr = row["attribute"].strip()
            # Divide por ';' y limpia espacios
            resources = {
                r.strip()
                for r in row["correct_fhir_resource"].split(";")
                if r.strip()
            }
            mapping[attr] = resources

    return mapping


def load_clusters(json_path: str) -> Dict[str, Any]:
    """Carga el JSON de clusters."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def validate_clusters(
    mapping: Dict[str, Set[str]],
    clusters: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Compara, para cada atributo de cada clúster, si alguno de los recursos
    correctos (según el CSV) aparece en los Top Resources del clúster.

    Devuelve una lista de dicts con el detalle de cada atributo.
    """
    results = []

    for cluster_name, cluster_data in clusters.items():
        # Ignoramos entradas que no son clústeres (Winner Embedding Config, etc.)
        if not cluster_name.startswith("Cluster"):
            continue

        attrs = cluster_data.get("Attributes", {})
        top_resources_raw = cluster_data.get("Top Resources", [])

        # Conjunto de recursos FHIR propuestos por similitud semántica para el clúster
        top_resources = {
            r.get("resource")
            for r in top_resources_raw
            if isinstance(r, dict) and r.get("resource")
        }

        for attr_name, _description in attrs.items():
            correct_resources = mapping.get(attr_name)

            if correct_resources is None:
                # Atributo no está en el CSV base
                results.append(
                    {
                        "cluster": cluster_name,
                        "attribute": attr_name,
                        #"in_base_mapping": False,
                        "correct_resources": "",
                        "cluster_top_resources": ";".join(sorted(top_resources)),
                        "hit": False,
                        "intersection": "",
                    }
                )
                continue

            # Intersección entre recursos correctos y top_resources del clúster
            intersection = correct_resources & top_resources
            hit = len(intersection) > 0

            results.append(
                {
                    "cluster": cluster_name,
                    "attribute": attr_name,
                    #"in_base_mapping": True,
                    "correct_resources": ";".join(sorted(correct_resources)),
                    "cluster_top_resources": ";".join(sorted(top_resources)),
                    "hit": hit,
                    "intersection": ";".join(sorted(intersection)),
                }
            )

    return results

def validate_clusters_attrs(
    mapping: Dict[str, Set[str]],
    clusters: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Compara, para cada atributo de cada clúster, si alguno de los recursos
    correctos (según el CSV) aparece en los Top Resources del clúster.

    Devuelve una lista de dicts con el detalle de cada atributo.
    """
    results = []

    for cluster_name in clusters:
        # Ignoramos entradas que no son clústeres (Winner Embedding Config, etc.)

        attr = clusters[cluster_name].get("Attributes")
        top_resources_raw = clusters[cluster_name].get("Top Resources", [])
        # Conjunto de recursos FHIR propuestos por similitud semántica para el clúster
        top_resources = {
            r.get("resource")
            for r in top_resources_raw
            if isinstance(r, dict) and r.get("resource")
        }
        print(attr)
        print(top_resources)
        correct_resources = mapping.get(attr)
        if correct_resources is None:
                # Atributo no está en el CSV base
                results.append(
                    {
                        "cluster": cluster_name,
                        "attribute": attr,
                        #"in_base_mapping": False,
                        "correct_resources": "",
                        "cluster_top_resources": ";".join(sorted(top_resources)),
                        "hit": False,
                        "intersection": "",
                    }
                )
                continue

            # Intersección entre recursos correctos y top_resources del clúster
        intersection = correct_resources & top_resources
        hit = len(intersection) > 0

        results.append(
                {
                    "cluster": cluster_name,
                    "attribute": attr,
                    #"in_base_mapping": True,
                    "correct_resources": ";".join(sorted(correct_resources)),
                    "cluster_top_resources": ";".join(sorted(top_resources)),
                    "hit": hit,
                    "intersection": ";".join(sorted(intersection)),
                }
            )

    return results

def print_summary(results: List[Dict[str, Any]]) -> None:
    """Imprime un resumen por clúster y un resumen global."""
    # Estadísticas por clúster
    stats_by_cluster = defaultdict(lambda: {"known": 0, "hits": 0, "unknown": 0})

    for r in results:
        cluster = r["cluster"]
        
        if r["hit"]:
            stats_by_cluster[cluster]["hits"] += 1
        else:
            stats_by_cluster[cluster]["unknown"] += 1
        
        
        #if r["in_base_mapping"]:
            #stats_by_cluster[cluster]["known"] += 1

        #else:
            #stats_by_cluster[cluster]["unknown"] += 1

    total_known = 0
    total_hits = 0
    total_unknown = 0
    total_acc = 0
    print("\n===== RESUMEN POR CLÚSTER =====")

    for cluster, s in sorted(stats_by_cluster.items()):
        known = s["known"]
        hits = s["hits"]
        unknown = s["unknown"]

        total = unknown + hits
        acc = hits * 100 / total 

        total_known += known
        total_hits += hits
        total_unknown += unknown
        total_acc = total_unknown + total_hits

        print(f"\n{cluster}:")
        #print(f"  Atributos en mapping base (known): {known}")
        print(f"  Aciertos (hit): {hits}")
        print(f"  Desconocidos (no están en CSV base): {unknown}")
        print(f"  Accuracy (hits / known): {acc:.3f}")

    global_acc = total_hits * 100 / total_acc 
    print("\n===== RESUMEN GLOBAL =====")
    #print(f"Atributos en mapping base (known): {total_known}")
    print(f"Aciertos totales (hit): {total_hits}")
    print(f"Atributos desconocidos (no están en CSV base): {total_unknown}")
    print(f"Accuracy global: {global_acc:.3f}\n")


def save_results_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    """Guarda el detalle de la validación en un CSV."""
    if not results:
        return

    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def main():
    # parser = argparse.ArgumentParser(
    #     description="Valida clusters de atributos y recursos FHIR frente a un mapping base."
    # )
    # parser.add_argument(
    #     "--mapping-csv",
    #     "-m",
    #     required=True,
    #     help="Ruta al CSV base con columnas attribute_name,correct_fhir_resource",
    # )
    # parser.add_argument(
    #     "--clusters-json",
    #     "-c",
    #     required=True,
    #     help="Ruta al JSON con la definición de clústeres",
    # )
    # parser.add_argument(
    #     "--output-csv",
    #     "-o",
    #     required=False,
    #     help="Ruta para guardar resultados detallados en CSV",
    # )

    # args = parser.parse_args()

    rag_filename = "clusters_rag_mimic_prueba_config_desc_v2" 
    mapping = load_mapping("correct_attributes_resources_v2.csv")
    clusters = load_clusters(f"output_rag/{rag_filename}.json")
    results = validate_clusters(mapping, clusters)

    print_summary(results)

    save_results_csv(results, f"results/results_{rag_filename}.csv")
    print(f"Resultados detallados guardados en: results/results_{rag_filename}.csv")


if __name__ == "__main__":
    main()
