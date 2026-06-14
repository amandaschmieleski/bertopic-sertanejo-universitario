import sys
import re
import ast
import os
from datetime import datetime
import argparse

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

import umap
import hdbscan
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic





RANDOM_STATE = 42

MIN_CLUSTER_SIZE = 25
MIN_SAMPLES = 1

N_NEIGHBORS = 25
N_COMPONENTS = 5
MIN_DIST = 0.0

EMBEDDING_MODEL_NAME = (
    "/home/amanda/.cache/huggingface/hub/"
    "models--intfloat_multilingual_e5_base"
    "snapshots/4328cf26390c98c5e3c738b4460a05b95f4911f5"
)

TEXT_COL_CANDIDATAS = ["ESTROFE", "trecho", "TRECHO", "letra"]


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"




stopwords_pt = [
    "dizer", "disse", "fala", "falar", "falou", "diz", "ai", "aí", "ah", "oh", "ô", "ei", "hey", "hei", "ieh", "ué", "uai", "hum", "hã", "eita", 
    "pra", "pro", "pras", "pros", "tá", "ta", "tô", "to", "cê", "ce", "né", "num", "lalaiá", "lalala", "laiá", "laiaiá", "lajala", "chorus", "miau", 
    "fiu", "tic", "tom", "tum", "tan", "tunt", "bum", "let", "go", "top", "look", "baby"

    "quer", "querer", "quero", "pode", "poder", "podia", "deve", "dever",
    "tem", "têm", "tenho", "tinha", "ter", "ficar", "fica", "ficou", "ficando",
    "deixar", "deixa", "deixou", "fazer",

    "nóis", "eu", "demais", "alguém", "fico", "nós", "voltar", "assim",
    "me", "te", "lhe", "ela", "ele", "elas", "eles", "cê", "você", "vocês", "tb",

    "pra", "pro", "pros", "q", "pq", "porque", "que", "se", "Iê", "fo",

    "oi", "iê", "ê", "ô", "ah", "oh", "ei", "uai", "oxe", "porra", "pá", "yeah", "Uôô", "Hô", "berê", "tcha", "iê", "Uô", "Lê Lê Lê",
    "opa", "oba", "aê", "ae", "yeah", "la", "lá", "aí", "ai", "Ooh", "Uô", "ih", "uh", "Bará", "Ôo", "Cha-la-la-la-la ", "Uôô", "Ai, ai",

    "a", "o", "as", "os", "tô", "nada", "vez", "dessas", "thiaguinho", "andrezinho",
    "de", "do", "da", "dos", "das", "aah", "ôô", "olha", "sério", "ni", "style", "ton ton", "zeca", "tú",
    "em", "no", "na", "nos", "nas", "aa", "fez", "fui", "vou", "essa", "juliana", "raimunda", "42",
    "por", "para", "com", "sem", "sobre", "entre", "vá", "daqui", "israel", "c7", "b7", "intro",
    "até", "um", "uns", "uma", "umas", "foi", "muitas", "ema", "israel novaes", "bit", "bm", "34",
    "mas", "mais", "ou", "e", "tá", "uê", "volto", "viu", "ôuô", "d7", "g7", "ão", "Iê", "dana", "4m",

    "seu", "sua", "seus", "suas", "sete", "nada", "kall", "daddy kall", "uo", "crew", "laia laia",
    "meu", "minha", "meus", "minhas", "está", "duzentos", "ram", "novaes", "tey", "cês", "dig dig", "ra",
    "teu", "tua", "teus", "tuas", "sou", "está", "uou", "dodge", "dj", "feat", "part", "ton", "ha",
    "nosso", "nossa", "nossos", "nossas", "nesta", "dodge ram", "tu", "se", "lá", "vai", "money", "nois",
    "dele", "dela", "deles", "delas", "ia", "valeu", "cm", "am", "ui", "nóis", "brown", "wool", "ig",

    "isso", "isto", "aquilo", "lauê", "lauê", "am7", "zum", "diguidim", "dm", "lê", "you", "love", 
    "aquele", "aquela", "aqueles", "aquelas", "180", "360", "212", "iô", "hardy", "tam", "thug", "didididiê",
    "quem", "onde", "quando", "como", "quanto", "qual", "quais", "algum", "ôôôôô", "boy", "lady",

    "agora", "sempre", "nunca", "tão", "muito", "pouco", "bem", "mal", "já", "ainda", "só", "também",

    "vi", "não", "novo", "esse", "pa", "ia", "era", "uuuu", "pam", "f7", "gm7", "és", "mc", "gm", "eb", "12", "32", "eb7",

    "isabela", "teodoro", "gugu", "salim", "ângela", "paula", "zé", "juca", "zezinho", "maria", "joãozinho", 
    "julieta", "vanda", "tereza",  "victor", "oswaldo", "lucimar", "robinho", "rodriguinho", "gabriela", "cláudio", "cátia", 
    "ana", "laura", "mariá", "felipinho", "lari", "lili", "pow", "ed", "ferrari",
]

STOPWORDS_FINAL = sorted(set([str(w).strip().lower() for w in stopwords_pt if str(w).strip()]))





def _strip_outer_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and ((text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'")):
        text = text[1:-1]
    return text


def ler_csv_robusto(path: str) -> pd.DataFrame:
    ultimo_erro = None


    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            ultimo_erro = e
            continue
        except pd.errors.ParserError as e:
            ultimo_erro = e
            break


    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                linhas = [linha.rstrip("\r\n") for linha in f]

            if not linhas:
                return pd.DataFrame({"trecho": []})

            cabecalho = linhas[0].lstrip("\ufeff").strip()
            if not cabecalho:
                cabecalho = "trecho"

            dados = [_strip_outer_quotes(linha) for linha in linhas[1:]]
            return pd.DataFrame({cabecalho: dados})

        except Exception as e:
            ultimo_erro = e
            continue

    raise ultimo_erro


def escolher_coluna_texto(df: pd.DataFrame) -> str:
    for c in TEXT_COL_CANDIDATAS:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in TEXT_COL_CANDIDATAS:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(
        f"Não encontrei coluna de texto. Procurei: {TEXT_COL_CANDIDATAS}. Colunas: {df.columns.tolist()}"
    )


def limpar_texto(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_representation(rep):
    if isinstance(rep, list):
        return rep
    if isinstance(rep, str):
        rep = rep.strip()
        if rep.startswith("[") and rep.endswith("]"):
            try:
                return ast.literal_eval(rep)
            except Exception:
                return []
    return []


def clean_representation_list(lst, stopwords_set):
    cleaned = []
    for t in lst:
        if not isinstance(t, str):
            continue
        tt = t.strip()
        if not tt:
            continue
        if tt.lower() in stopwords_set:
            continue
        cleaned.append(tt)
    return cleaned



class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _col(df, *cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def merge_by_similarity(hier_df: pd.DataFrame, sim_threshold: float = 0.8, dist_threshold: float | None = None):
    dist_col = _col(hier_df, "Distance", "distance", "DISTANCE")
    left_col = _col(hier_df, "Child_Left_ID", "Child_Left", "Left", "left", "child_left_id")
    right_col = _col(hier_df, "Child_Right_ID", "Child_Right", "Right", "right", "child_right_id")

    if not dist_col or not left_col or not right_col:
        raise ValueError(f"Não reconheci colunas esperadas no hierarchical_topics. Colunas: {hier_df.columns.tolist()}")

    df = hier_df.copy()
    df = df.dropna(subset=[dist_col, left_col, right_col])

    df = df.sort_values(dist_col, ascending=True).reset_index(drop=True)

    max_dist = float(df[dist_col].max()) if len(df) else 1.0
    if max_dist <= 0:
        max_dist = 1.0

    if dist_threshold is None:
        dist_threshold = (1.0 - float(sim_threshold)) * max_dist

    uf = UnionFind()

    for _, row in df.iterrows():
        dist = float(row[dist_col])
        if dist <= dist_threshold:
            uf.union(int(row[left_col]), int(row[right_col]))
        else:
            break

    groups = {}
    for _, row in df.iterrows():
        a = int(row[left_col])
        b = int(row[right_col])
        groups.setdefault(uf.find(a), set()).add(a)
        groups.setdefault(uf.find(b), set()).add(b)

    groups = {root: sorted(list(members)) for root, members in groups.items() if len(members) > 1}
    return groups, dist_threshold, max_dist


def apply_groups_to_docs(topics: np.ndarray, groups: dict):
    mapping = {}
    next_id = 0

    for _, members in groups.items():
        for t in members:
            mapping[t] = next_id
        next_id += 1

    unique_topics = sorted(set(int(t) for t in topics if int(t) != -1))
    for t in unique_topics:
        if t not in mapping:
            mapping[t] = next_id
            next_id += 1

    merged = np.array([(-1 if int(t) == -1 else mapping[int(t)]) for t in topics], dtype=int)
    return merged, mapping





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="CSV com coluna de texto (ex.: ESTROFE)")
    ap.add_argument(
        "--nr_topics",
        type=int,
        default=None,
        help="Número de tópicos para reduce_topics. Se não informar, não reduz.",
    )
    ap.add_argument("--sim", type=float, default=None, help="(Opção 3) Similaridade mínima para mesclar (0-1). Ex: 0.8")
    ap.add_argument("--dist", type=float, default=None, help="(Opção 3) Distância máxima para mesclar. Se setar, ignora --sim.")
    args = ap.parse_args()

    df = ler_csv_robusto(args.csv_path)
    col_texto = escolher_coluna_texto(df)
    print(f"✅ Coluna de texto: {col_texto}")

    docs = [limpar_texto(x) for x in df[col_texto].tolist()]
    mask_ok = [len(d) >= 20 for d in docs]
    df = df.loc[mask_ok].reset_index(drop=True)
    docs = [d for d, ok in zip(docs, mask_ok) if ok]

    print(f"✅ Linhas lidas: {len(mask_ok)} | Textos válidos: {len(docs)}")
    print(f"✅ Stopwords (normalizadas): {len(STOPWORDS_FINAL)}")

    vectorizer_model = CountVectorizer(
        stop_words=STOPWORDS_FINAL,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
    )

    umap_model = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        n_components=N_COMPONENTS,
        min_dist=MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE,
        low_memory=True,
    )

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    if not os.path.isdir(EMBEDDING_MODEL_NAME):
        raise FileNotFoundError(
            f"Modelo local não encontrado em: {EMBEDDING_MODEL_NAME}"
        )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        local_files_only=True
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        verbose=True,
        calculate_probabilities=True,
    )

    topics, probs = topic_model.fit_transform(docs)

    if args.nr_topics is not None:
        topic_model.reduce_topics(docs, nr_topics=int(args.nr_topics))
        doc_info_reduce = topic_model.get_document_info(docs)
        topics_final = doc_info_reduce["Topic"].to_numpy()
        print(f"✅ reduce_topics aplicado com nr_topics={int(args.nr_topics)}")
    else:
        topics_final = np.array(topics)
        print("✅ reduce_topics não aplicado; usando tópicos originais do BERTopic")

    outliers = int(np.sum(topics_final == -1))
    n_topics = len(set([t for t in topics_final if t != -1]))

    print("\n=========================")
    print(f"📌 Tópicos (sem -1): {n_topics}")
    print(f"📌 Docs outliers (Topic=-1): {outliers}")
    print("=========================\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    hierarchical_topics = topic_model.hierarchical_topics(docs)
    out_hier_csv = f"hierarchical_topics_{ts}.csv"
    out_hier_html = f"hierarquia_{ts}.html"

    hierarchical_topics.to_csv(out_hier_csv, index=False, encoding="utf-8-sig")

    fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
    fig.write_html(out_hier_html)

    print(f"✅ Hierarquia CSV: {out_hier_csv}")
    print(f"✅ Hierarquia HTML: {out_hier_html}")

    out_topics = f"resultados_bertopic_normal_{ts}.csv"
    out_map = f"mapa_docs_bertopic_normal_{ts}.csv"
    out_outliers = f"outliers_bertopic_normal_{ts}.csv"

    topic_model.get_topic_info().to_csv(out_topics, index=False, encoding="utf-8-sig")

    df_out = df.copy()
    df_out["topic"] = topics_final
    df_out.to_csv(out_map, index=False, encoding="utf-8-sig")
    df_out[df_out["topic"] == -1].to_csv(out_outliers, index=False, encoding="utf-8-sig")

    print(f"✅ CSV (tópicos): {out_topics}")
    print(f"✅ CSV (docs->tópico): {out_map}")
    print(f"✅ CSV (outliers): {out_outliers}")


    topic_info_limpo = topic_model.get_topic_info().copy()
    if "Representation" in topic_info_limpo.columns:
        parsed = topic_info_limpo["Representation"].map(parse_representation)
        cleaned = parsed.map(lambda lst: clean_representation_list(lst, STOPWORDS_FINAL))
        topic_info_limpo["Representation_Clean"] = cleaned
        topic_info_limpo["Representation"] = topic_info_limpo["Representation_Clean"].map(lambda lst: lst)

        def make_name(row):
            words = row["Representation_Clean"][:4]
            if not words:
                return f"{row['Topic']}_sem_palavras"
            return f"{row['Topic']}_" + "_".join(w.replace(" ", "-") for w in words)

        topic_info_limpo["Name_Clean"] = topic_info_limpo.apply(make_name, axis=1)

    out_topics_limpo = f"resultados_bertopic_LIMPO_{ts}.csv"
    topic_info_limpo.to_csv(out_topics_limpo, index=False, encoding="utf-8-sig")
    print(f"✅ CSV (tópicos limpos): {out_topics_limpo}")

    doc_info_full = topic_model.get_document_info(docs)

    df_docs_topicos = df.copy()
    df_docs_topicos["Topic"] = topics_final

    if "Probability" in doc_info_full.columns:
        df_docs_topicos["Probability"] = doc_info_full["Probability"].values
    else:
        probs_arr = np.asarray(probs) if probs is not None else None
        df_docs_topicos["Probability"] = probs_arr.max(axis=1) if (probs_arr is not None and probs_arr.ndim == 2) else None

    out_docs_topicos = f"docs_com_topicos_{ts}.csv"
    df_docs_topicos.to_csv(out_docs_topicos, index=False, encoding="utf-8-sig")
    print(f"✅ CSV (docs + tópicos + prob): {out_docs_topicos}")

    out_outliers_extra = f"outliers_bertopic_{ts}.csv"
    df_docs_topicos[df_docs_topicos["Topic"] == -1].to_csv(out_outliers_extra, index=False, encoding="utf-8-sig")
    print(f"✅ CSV (outliers extra): {out_outliers_extra}")

    if args.sim is not None or args.dist is not None:
        sim = 0.8 if args.sim is None else float(args.sim)
        dist = None if args.dist is None else float(args.dist)

        groups, used_dist_threshold, max_dist = merge_by_similarity(
            hierarchical_topics, sim_threshold=sim, dist_threshold=dist
        )

        merged_topics, mapping = apply_groups_to_docs(topics_final, groups)

        out_map_merged = f"mapa_docs_bertopic_merged_sim_{ts}.csv"
        out_summary = f"merged_topics_summary_{ts}.csv"

        df_m = df.copy()
        df_m["topic_original"] = topics_final
        df_m["topic_merged"] = merged_topics
        df_m.to_csv(out_map_merged, index=False, encoding="utf-8-sig")

        info = topic_model.get_topic_info()
        info = info[info["Topic"] != -1].copy()
        info["Topic"] = info["Topic"].astype(int)

        inv = {}
        for k, v in mapping.items():
            inv.setdefault(v, []).append(k)

        rows = []
        for merged_id, members in sorted(inv.items(), key=lambda x: x[0]):
            sub = info[info["Topic"].isin(members)]
            if len(sub) > 0:
                rep = sub.sort_values("Count", ascending=False).iloc[0]
                rep_name = rep.get("Name", "")
                total = int(sub["Count"].sum())
            else:
                rep_name = ""
                total = 0

            rows.append({
                "topic_merged": int(merged_id),
                "total_docs": total,
                "members_original_topics": ",".join(map(str, sorted(members))),
                "representative_name": rep_name,
            })

        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(out_summary, index=False, encoding="utf-8-sig")

        print("\n=========================")
        print("🧠 Opção 3 (conceitual) aplicada")
        print(f"   sim_threshold: {sim}  |  maxDist: {max_dist:.4f}  |  dist_threshold_usado: {used_dist_threshold:.4f}")
        print(f"✅ CSV (docs->tópico merged): {out_map_merged}")
        print(f"✅ CSV (resumo merges): {out_summary}")
        print("=========================\n")


if __name__ == "__main__":
    main()