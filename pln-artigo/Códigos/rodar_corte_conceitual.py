


from __future__ import annotations

import argparse
import ast
import os
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd





class UnionFind:
    def __init__(self, items: Iterable[int]):
        self.parent: Dict[int, int] = {i: i for i in items}
        self.rank: Dict[int, int] = {i: 0 for i in items}

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1





def safe_int_list(x) -> List[int]:
    if isinstance(x, list):
        out = x
    elif pd.isna(x):
        return []
    else:
        s = str(x).strip()
        try:
            out = ast.literal_eval(s)
        except Exception:
            out = [int(v) for v in re.findall(r"-?\d+", s)]

    if isinstance(out, (tuple, set)):
        out = list(out)
    if not isinstance(out, list):
        out = [out]

    res = []
    for v in out:
        try:
            res.append(int(v))
        except Exception:
            pass
    return res


def safe_str_list(x) -> List[str]:
    if isinstance(x, list):
        out = x
    elif pd.isna(x):
        return []
    else:
        s = str(x).strip()
        try:
            out = ast.literal_eval(s)
        except Exception:
            s2 = s.strip("[]")
            parts = [p.strip() for p in s2.split(",") if p.strip()]
            out = [p.strip("'").strip('"') for p in parts]

    if isinstance(out, (tuple, set)):
        out = list(out)
    if not isinstance(out, list):
        out = [out]

    return [str(v).strip().strip("'").strip('"') for v in out if str(v).strip()]


def parse_representation_cell(cell) -> List[str]:
    return safe_str_list(cell)





def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-z0-9\s_-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


CONCEPT_MAP = {
    "paixao": {
        "amor", "amar", "amo", "apaixonado", "apaixonada", "apaixonar",
        "apaixonei", "apaixonou", "paixao", "paixonite", "apaixonadamente"
    },
    "saudade": {
        "saudade", "saudades", "falta", "lembranca", "lembrancas", "recordacao", "recordacoes"
    },
    "sofrimento_amoroso": {
        "sofri", "sofrer", "sofrimento", "chorei", "choro", "dor",
        "magoa", "magoa", "machucou", "machuca", "ferida", "tristeza",
        "sofrido", "doeu", "sofrendo"
    },
    "termino_rejeicao": {
        "termino", "terminou", "adeus", "abandono", "largou", "rejeicao",
        "rejeitado", "acabou", "fim", "separacao", "separou", "foi embora"
    },
    "reconciliacao_volta": {
        "volta", "voltar", "voltou", "reatar", "reconciliar", "reconciliacao", "perdao"
    },
    "beijo_atracao": {
        "beijo", "beijar", "beijou", "boca", "desejo", "atracao", "quimica", "tesao"
    },
    "ciume_posse": {
        "ciume", "ciumes", "posse", "rival", "traicao", "traiu", "desconfianca", "inseguranca"
    },
    "festa_bebida": {
        "bar", "cerveja", "cachaca", "buteco", "balada", "festa", "bebida", "bebendo", "barzinho"
    },
    "vida_roca_sertao": {
        "fazenda", "roca", "boi", "cavalo", "terra", "sertao", "interior", "peao", "viola"
    },
    "coracao_sentimento": {
        "coracao", "sentimento", "emocao", "alma", "sentir", "senti"
    },
}


def keyword_to_concept(word: str) -> str:
    w = normalize_text(word)

    for concept, vocab in CONCEPT_MAP.items():
        if w in vocab:
            return concept


    if w.startswith("apaixon"):
        return "paixao"
    if w.startswith("am"):
        if w in {"amor", "amar", "amo"}:
            return "paixao"
    if w.startswith("saudad"):
        return "saudade"
    if w.startswith("sofr"):
        return "sofrimento_amoroso"
    if w.startswith("chor"):
        return "sofrimento_amoroso"
    if w.startswith("beij"):
        return "beijo_atracao"
    if w.startswith("volt"):
        return "reconciliacao_volta"
    if w.startswith("cium"):
        return "ciume_posse"
    if w.startswith("term") or w.startswith("acab"):
        return "termino_rejeicao"

    return w





def auto_paths(base_dir: str, stamp: str) -> Dict[str, str]:
    base_dir = base_dir or "."
    return {
        "hier": os.path.join(base_dir, f"hierarchical_topics_{stamp}.csv"),
        "topics": os.path.join(base_dir, f"resultados_bertopic_LIMPO_{stamp}.csv"),
        "map": os.path.join(base_dir, f"mapa_docs_bertopic_normal_{stamp}.csv"),
    }


def pick_existing(paths: Dict[str, str]) -> Dict[str, Optional[str]]:
    return {k: (p if p and os.path.exists(p) else None) for k, p in paths.items()}





def merge_by_dist(hier_df: pd.DataFrame, leaf_topics: List[int], dist_cut: float) -> Dict[int, List[int]]:
    uf = UnionFind(leaf_topics)
    h = hier_df.sort_values("Distance", ascending=True)

    for _, row in h.iterrows():
        d = float(row["Distance"])
        if d > dist_cut:
            break

        topics = safe_int_list(row.get("Topics"))
        topics = [t for t in topics if t in uf.parent]

        if len(topics) <= 1:
            continue

        base = topics[0]
        for t in topics[1:]:
            uf.union(base, t)

    groups = defaultdict(list)
    for t in leaf_topics:
        groups[uf.find(t)].append(t)

    return {root: sorted(members) for root, members in groups.items()}





def build_labels(groups: Dict[int, List[int]], topics_df: Optional[pd.DataFrame], top_k_words: int = 6):
    group_items = sorted(groups.items(), key=lambda kv: (-len(kv[1]), min(kv[1])))
    group_id_map = {root: i for i, (root, _) in enumerate(group_items)}

    topic_count: Dict[int, int] = {}
    topic_words: Dict[int, List[str]] = {}

    if topics_df is not None and "Topic" in topics_df.columns:
        df = topics_df.copy()
        df["Topic"] = df["Topic"].astype(int)

        if "Count" in df.columns:
            topic_count = dict(zip(df["Topic"], df["Count"].astype(int)))

        source_col = None
        if "Representation_Clean" in df.columns:
            source_col = "Representation_Clean"
        elif "Representation" in df.columns:
            source_col = "Representation"

        if source_col is not None:
            for _, r in df.iterrows():
                tid = int(r["Topic"])
                topic_words[tid] = parse_representation_cell(r.get(source_col))

    rows_map = []
    rows_merged = []

    for root, members in group_items:
        gid = group_id_map[root]

        total = 0
        concept_counter = Counter()
        evidence_counter = Counter()

        for t in members:
            c = int(topic_count.get(t, 0))
            total += c

            for w in topic_words.get(t, []):
                if not w:
                    continue

                concept = keyword_to_concept(w)
                weight = max(c, 1)

                concept_counter[concept] += weight
                evidence_counter[w] += weight

        if concept_counter:
            main_concepts = [c for c, _ in concept_counter.most_common(3)]
            name = "_".join(main_concepts)
        else:
            name = f"grupo_{gid}"

        evidence = [w for w, _ in evidence_counter.most_common(top_k_words)]

        rows_merged.append({
            "MergedTopic": gid,
            "MergedName": name,
            "MergedKeywordsEvidence": evidence,
            "MergedCount": total if total > 0 else None,
            "n_topics": len(members),
            "topics": members,
        })

        for t in members:
            rows_map.append({
                "Topic": t,
                "MergedTopic": gid,
                "MergedName": name,
                "MergedKeywordsEvidence": evidence,
            })

    mapping_df = pd.DataFrame(rows_map).sort_values(["MergedTopic", "Topic"]).reset_index(drop=True)
    merged_df = pd.DataFrame(rows_merged).sort_values(["MergedTopic"]).reset_index(drop=True)
    return mapping_df, merged_df


def scan_table(hier_df: pd.DataFrame, topics_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if topics_df is not None and "Topic" in topics_df.columns:
        leaf = sorted([int(t) for t in topics_df["Topic"].unique() if int(t) != -1])
    else:
        leaf = sorted(set(sum((safe_int_list(x) for x in hier_df["Topics"].tolist()), [])))
        leaf = [t for t in leaf if t != -1]

    dist = hier_df["Distance"].astype(float).to_numpy()
    qs = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    rows = []
    for q in qs:
        dcut = float(np.quantile(dist, q))
        groups = merge_by_dist(hier_df, leaf, dcut)
        rows.append({
            "quantil": q,
            "dist_cut": dcut,
            "n_grupos": len(groups),
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Diretório onde estão os CSVs.")
    ap.add_argument("--stamp", default=None, help="Timestamp (ex: 20260324_115255) para auto-detectar arquivos.")
    ap.add_argument("--hier", default=None, help="Caminho do hierarchical_topics_*.csv (override).")
    ap.add_argument("--topics", default=None, help="Caminho do resultados_*.csv (opcional).")
    ap.add_argument("--map", dest="map_csv", default=None, help="Caminho do mapa_docs_*.csv (opcional).")

    ap.add_argument("--dist-cut", type=float, default=None, help="Corte por DISTÂNCIA.")
    ap.add_argument("--merge-level", type=float, default=None, help="Atalho 0..1: define dist_cut via quantil.")
    ap.add_argument("--scan", action="store_true", help="Mostra tabela (quantil -> dist_cut -> nº grupos) e sai.")
    ap.add_argument("--out-prefix", default=None, help="Prefixo dos arquivos de saída.")

    args = ap.parse_args()

    if args.stamp:
        auto = pick_existing(auto_paths(args.dir, args.stamp))
        hier_path = args.hier or auto["hier"]
        topics_path = args.topics or auto["topics"]
        map_path = args.map_csv or auto["map"]
    else:
        hier_path = args.hier
        topics_path = args.topics
        map_path = args.map_csv

    if not hier_path or not os.path.exists(hier_path):
        raise SystemExit(f"❌ hierarchical_topics não encontrado. Passe --hier ou --stamp. (tentativa: {hier_path})")

    hier_df = pd.read_csv(hier_path)
    if "Distance" not in hier_df.columns or "Topics" not in hier_df.columns:
        raise SystemExit("❌ CSV de hierarquia inesperado: preciso de colunas 'Distance' e 'Topics'.")

    topics_df = pd.read_csv(topics_path) if (topics_path and os.path.exists(topics_path)) else None
    map_df = pd.read_csv(map_path) if (map_path and os.path.exists(map_path)) else None

    if args.scan:
        tbl = scan_table(hier_df, topics_df)
        print("\n📌 Tabela rápida (quanto MAIOR o dist_cut, MAIS fusões / MENOS grupos):")
        print(tbl.to_string(index=False))
        return


    if topics_df is not None and "Topic" in topics_df.columns:
        leaf_topics = sorted([int(t) for t in topics_df["Topic"].unique() if int(t) != -1])
    else:
        leaf_topics = sorted(set(sum((safe_int_list(x) for x in hier_df["Topics"].tolist()), [])))
        leaf_topics = [t for t in leaf_topics if t != -1]

    dist_arr = hier_df["Distance"].astype(float).to_numpy()

    dist_cut = args.dist_cut
    if dist_cut is None:
        lvl = 0.80 if args.merge_level is None else float(args.merge_level)
        if not (0.0 < lvl < 1.0):
            raise SystemExit("❌ --merge-level precisa estar entre 0 e 1 (ex: 0.80).")
        dist_cut = float(np.quantile(dist_arr, lvl))

    groups = merge_by_dist(hier_df, leaf_topics, dist_cut)
    mapping_df, merged_df = build_labels(groups, topics_df)

    if args.out_prefix:
        prefix = args.out_prefix
    else:
        base = args.stamp or "manual"
        prefix = f"corte_conceitual_{base}_dist{dist_cut:.3f}".replace(".", "p")

    out_dir = args.dir or "."
    os.makedirs(out_dir, exist_ok=True)

    out_map = os.path.join(out_dir, f"topic_merge_map_{prefix}.csv")
    out_merged = os.path.join(out_dir, f"merged_topics_{prefix}.csv")
    mapping_df.to_csv(out_map, index=False, encoding="utf-8")
    merged_df.to_csv(out_merged, index=False, encoding="utf-8")

    print("\n✅ Corte conceitual concluído!")
    print(f"   hier:   {hier_path}")
    if topics_path and os.path.exists(topics_path):
        print(f"   topics: {topics_path}")
    if map_path and os.path.exists(map_path):
        print(f"   map:    {map_path}")

    print(f"\n📏 dist_cut = {dist_cut:.6f}")
    print(f"🧩 tópicos folhas = {len(leaf_topics)} | grupos = {len(merged_df)}")

    print(f"\n📝 Saídas:")
    print(f"   {out_map}")
    print(f"   {out_merged}")

    if map_df is not None:
        topic_col = None
        for cand in ("topic", "Topic", "topic_original", "Topic_original"):
            if cand in map_df.columns:
                topic_col = cand
                break

        if topic_col is None:
            raise SystemExit(f"❌ mapa_docs não tem coluna de tópico. Colunas: {map_df.columns.tolist()}")

        tmp = mapping_df.copy().rename(columns={"Topic": "__topic__"})
        map_out = map_df.copy()
        map_out["__topic__"] = map_out[topic_col].astype(int)
        map_out = map_out.merge(tmp, on="__topic__", how="left").drop(columns=["__topic__"])

        out_map_docs = os.path.join(out_dir, f"mapa_docs_merged_{prefix}.csv")
        map_out.to_csv(out_map_docs, index=False, encoding="utf-8")
        print(f"   {out_map_docs}")

        print("\n🔎 Prévia dos grupos (top 10):")
        print(merged_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()