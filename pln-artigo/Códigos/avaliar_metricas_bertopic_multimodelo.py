


from __future__ import annotations

import argparse
import ast
import glob
import math
import os
import re
import unicodedata
from itertools import combinations
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


TEXT_COL_CANDIDATES = [
    "ESTROFE", "estrofe", "trecho", "TRECHO", "letra", "Letra", "LETRA",
    "text", "Texto", "Document", "document", "doc"
]

TOPIC_COL_CANDIDATES = ["Topic", "topic", "topic_original", "Topic_original"]


def normalize_text(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-z0-9_\-\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    toks = re.findall(r"[a-zA-Z0-9_\-]{2,}", s)
    return toks


def parse_list_cell(cell) -> List[str]:
    if isinstance(cell, list):
        out = cell
    elif pd.isna(cell):
        return []
    else:
        s = str(cell).strip()
        try:
            out = ast.literal_eval(s)
        except Exception:
            s2 = s.strip("[]")
            out = [p.strip().strip("'").strip('"') for p in s2.split(",") if p.strip()]

    if isinstance(out, (tuple, set)):
        out = list(out)
    if not isinstance(out, list):
        out = [out]

    words = []
    for x in out:
        x = normalize_text(str(x).strip())
        if x:
            words.append(x)
    return words


def find_first_existing(patterns: Iterable[str]) -> Optional[str]:
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    candidates = sorted(set(candidates))
    if not candidates:
        return None


    def score(path: str) -> Tuple[int, str]:
        name = os.path.basename(path)
        s = 0
        if "LIMPO" in name:
            s -= 30
        if name.startswith("resultados_bertopic"):
            s -= 20
        if name.startswith("mapa_docs_bertopic"):
            s -= 20
        if name.startswith("docs_com_topicos"):
            s -= 10
        if "outliers" in name.lower():
            s += 100
        return (s, name)

    return sorted(candidates, key=score)[0]


def auto_topic_file(base_dir: str, stamp: str) -> Optional[str]:
    return find_first_existing([
        os.path.join(base_dir, f"resultados_bertopic_LIMPO_{stamp}.csv"),
        os.path.join(base_dir, f"resultados_bertopic_*_LIMPO_{stamp}.csv"),
        os.path.join(base_dir, f"resultados_bertopic_normal_{stamp}.csv"),
        os.path.join(base_dir, f"resultados_bertopic_*_{stamp}.csv"),
    ])


def auto_map_file(base_dir: str, stamp: str) -> Optional[str]:
    return find_first_existing([
        os.path.join(base_dir, f"mapa_docs_bertopic_normal_{stamp}.csv"),
        os.path.join(base_dir, f"mapa_docs_bertopic_*_{stamp}.csv"),
        os.path.join(base_dir, f"docs_com_topicos_{stamp}.csv"),
        os.path.join(base_dir, f"docs_com_topicos_*_{stamp}.csv"),
    ])


def get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def extract_topic_words(topic_df: pd.DataFrame, topk: int) -> List[List[str]]:
    if "Topic" not in topic_df.columns:
        raise ValueError(f"Arquivo de tópicos não tem coluna Topic. Colunas: {topic_df.columns.tolist()}")

    df = topic_df.copy()
    df["Topic"] = df["Topic"].astype(int)
    df = df[df["Topic"] != -1]

    source_col = None
    for c in ["Representation_Clean", "Representation", "Name_Clean", "Name"]:
        if c in df.columns:
            source_col = c
            break
    if source_col is None:
        raise ValueError(
            "Arquivo de tópicos não tem Representation_Clean, Representation, Name_Clean ou Name."
        )

    topics = []
    for _, row in df.sort_values("Topic").iterrows():
        words = parse_list_cell(row[source_col])

        if len(words) <= 1 and source_col.lower().startswith("name"):
            raw = normalize_text(str(row[source_col]))
            raw = re.sub(r"^-?\d+_?", "", raw)
            words = [w for w in raw.split("_") if w]
        words = [w for w in words if w]
        if words:
            topics.append(words[:topk])
    return topics


def proportion_unique_words(topics: List[List[str]], topk: int = 10) -> float:
    if not topics:
        return float("nan")
    trimmed = [t[:topk] for t in topics if len(t) > 0]
    denom = topk * len(trimmed)
    if denom == 0:
        return float("nan")
    unique_words = set()
    for t in trimmed:
        unique_words.update(t)
    return len(unique_words) / denom


def pairwise_jaccard_diversity(topics: List[List[str]], topk: int = 10) -> float:
    trimmed = [t[:topk] for t in topics if len(t) > 0]
    if len(trimmed) < 2:
        return float("nan")
    vals = []
    for a, b in combinations(trimmed, 2):
        sa, sb = set(a), set(b)
        union = sa | sb
        if not union:
            continue
        vals.append(1.0 - (len(sa & sb) / len(union)))
    return float(np.mean(vals)) if vals else float("nan")


def rbo_score(a: List[str], b: List[str], p: float = 0.9) -> float:
    if not a or not b:
        return 0.0
    depth = max(len(a), len(b))
    seen_a, seen_b = set(), set()
    score = 0.0
    for d in range(1, depth + 1):
        if d <= len(a):
            seen_a.add(a[d - 1])
        if d <= len(b):
            seen_b.add(b[d - 1])
        overlap = len(seen_a & seen_b)
        agreement = overlap / d
        score += agreement * (p ** (d - 1))
    return (1 - p) * score


def inverted_rbo(topics: List[List[str]], topk: int = 10, p: float = 0.9) -> float:
    trimmed = [t[:topk] for t in topics if len(t) > 0]
    if len(trimmed) < 2:
        return float("nan")
    vals = [rbo_score(a, b, p=p) for a, b in combinations(trimmed, 2)]
    return 1.0 - float(np.mean(vals))


def coherence_scores(texts_tokenized: List[List[str]], topics: List[List[str]]) -> Tuple[float, float]:
    try:
        import gensim.corpora as corpora
        from gensim.models.coherencemodel import CoherenceModel
    except Exception as e:
        raise RuntimeError(
            "Não consegui importar gensim. Instale com: pip install gensim"
        ) from e

    texts_tokenized = [t for t in texts_tokenized if t]
    dictionary = corpora.Dictionary(texts_tokenized)
    if len(dictionary) == 0:
        return float("nan"), float("nan")

    dictionary.filter_extremes(no_below=2, no_above=0.85)
    if len(dictionary) == 0:
        dictionary = corpora.Dictionary(texts_tokenized)

    topic_words = []
    for topic in topics:
        words = [w for w in topic if w in dictionary.token2id]
        if len(words) >= 2:
            topic_words.append(words)

    if not topic_words:
        return float("nan"), float("nan")

    cm_cv = CoherenceModel(
        topics=topic_words,
        texts=texts_tokenized,
        dictionary=dictionary,
        coherence="c_v",
        processes=1,
    )
    cm_npmi = CoherenceModel(
        topics=topic_words,
        texts=texts_tokenized,
        dictionary=dictionary,
        coherence="c_npmi",
        processes=1,
    )
    return float(cm_cv.get_coherence()), float(cm_npmi.get_coherence())


def evaluate_run(label: str, stamp: str, base_dir: str, topk: int, p_rbo: float) -> dict:
    topic_path = auto_topic_file(base_dir, stamp)
    map_path = auto_map_file(base_dir, stamp)

    if topic_path is None:
        raise FileNotFoundError(f"Não encontrei arquivo de tópicos para stamp {stamp}")
    if map_path is None:
        raise FileNotFoundError(f"Não encontrei arquivo mapa/docs para stamp {stamp}")

    topic_df = pd.read_csv(topic_path)
    map_df = pd.read_csv(map_path)

    topic_col = get_col(map_df, TOPIC_COL_CANDIDATES)
    text_col = get_col(map_df, TEXT_COL_CANDIDATES)
    if topic_col is None:
        raise ValueError(f"Mapa {map_path} não tem coluna de tópico. Colunas: {map_df.columns.tolist()}")
    if text_col is None:
        raise ValueError(f"Mapa {map_path} não tem coluna de texto. Colunas: {map_df.columns.tolist()}")

    topics_assigned = map_df[topic_col].astype(int).to_numpy()
    n_docs = int(len(map_df))
    outliers = int(np.sum(topics_assigned == -1))
    valid_topics = sorted(set(int(t) for t in topics_assigned if int(t) != -1))
    n_topics = int(len(valid_topics))
    grouped = int(n_docs - outliers)
    out_pct = (outliers / n_docs * 100.0) if n_docs else float("nan")
    avg_per_topic = (grouped / n_topics) if n_topics else float("nan")

    topic_words = extract_topic_words(topic_df, topk=topk)
    docs_tokens = [tokenize(x) for x in map_df[text_col].tolist()]

    cv, npmi = coherence_scores(docs_tokens, topic_words)
    mean_coh = np.nanmean([cv, npmi])
    div = proportion_unique_words(topic_words, topk=topk)
    jac = pairwise_jaccard_diversity(topic_words, topk=topk)
    rbo_inv = inverted_rbo(topic_words, topk=topk, p=p_rbo)
    mean_div = np.nanmean([div, rbo_inv])

    return {
        "Modelo": label,
        "Timestamp": stamp,
        "Arquivo_topicos": os.path.basename(topic_path),
        "Arquivo_mapa": os.path.basename(map_path),
        "Textos_validos": n_docs,
        "Topicos": n_topics,
        "Outliers": outliers,
        "Outliers_%": out_pct,
        "Textos_agrupados": grouped,
        "Media_por_topico": avg_per_topic,
        "Coerencia_CV": cv,
        "NPMI": npmi,
        "Media_coerencias": mean_coh,
        "Diversidade": div,
        "Jaccard_diversidade": jac,
        "RBO_invertido": rbo_inv,
        "Media_diversidades": mean_div,
    }


def format_pt_float(x, nd=3):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "--"
    return f"{x:.{nd}f}".replace(".", ",")


def latex_escape(s: str) -> str:
    s = str(s)
    return (s.replace("\\", r"\textbackslash{}")
             .replace("_", r"\_")
             .replace("%", r"\%")
             .replace("&", r"\&")
             .replace("#", r"\#")
             .replace("{", r"\{")
             .replace("}", r"\}"))


def write_latex_table(df: pd.DataFrame, out_path: str):
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comparação quantitativa dos modelos de \textit{embeddings} aplicados ao mesmo corpus.}")
    lines.append(r"\label{tab:comparacao-modelos-embeddings}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{adjustbox}{max width=\linewidth}")
    lines.append(r"\begin{tabular}{p{7.2cm}rrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Modelo de embeddings} & \textbf{Tópicos} & \textbf{Outliers} & \textbf{Outliers (\%)} & \textbf{Textos agrupados} & \textbf{Média por tópico} \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        model = latex_escape(str(r["Modelo"]))
        line = (r"\texttt{" + model + r"}"
                + f" & {int(r['Topicos'])} & {int(r['Outliers'])} & "
                + f"{format_pt_float(float(r['Outliers_%']), 2)} & {int(r['Textos_agrupados'])} & "
                + f"{format_pt_float(float(r['Media_por_topico']), 1)} "
                + r"\\")
        lines.append(line)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{table}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Diretório dos CSVs gerados pelo BERTopic.")
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        help="Modelo e timestamp no formato 'nome_do_modelo|YYYYMMDD_HHMMSS'. Pode repetir.",
    )
    ap.add_argument("--topk", type=int, default=10, help="Número de palavras por tópico para métricas.")
    ap.add_argument("--p-rbo", type=float, default=0.9, help="Parâmetro p do RBO.")
    ap.add_argument("--out", default="metricas_bertopic_modelos.csv", help="CSV de saída.")
    ap.add_argument("--latex", default="tabela_modelos_bertopic_overleaf.tex", help="Tabela LaTeX de saída.")
    args = ap.parse_args()

    rows = []
    for item in args.run:
        if "|" not in item:
            raise SystemExit("Cada --run precisa estar no formato 'modelo|timestamp'.")
        label, stamp = item.split("|", 1)
        label = label.strip()
        stamp = stamp.strip()
        print(f"\n🔎 Avaliando: {label} | {stamp}")
        rows.append(evaluate_run(label, stamp, args.dir, args.topk, args.p_rbo))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    write_latex_table(df, args.latex)

    cols = [
        "Modelo", "Topicos", "Outliers", "Outliers_%", "Textos_agrupados",
        "Media_por_topico", "Coerencia_CV", "NPMI", "Diversidade", "RBO_invertido"
    ]
    print("\n✅ Métricas calculadas:")
    print(df[cols].to_string(index=False))
    print(f"\n📄 CSV: {args.out}")
    print(f"📄 Tabela Overleaf: {args.latex}")


if __name__ == "__main__":
    main()
