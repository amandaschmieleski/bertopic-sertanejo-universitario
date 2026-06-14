

import os
import re
import io
import csv
import json
import time
import gzip
import random
import sqlite3
import requests
from datetime import datetime
from typing import Optional, Iterable, List, Tuple

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag





BASE_URL = "https://www.letras.mus.br"
SITEMAP_INDEX_URL = "https://www.letras.mus.br/sitemap_index.xml.gz"


EXIGIR_CONFIRMACAO_SERTANEJO = os.getenv("EXIGIR_CONFIRMACAO_SERTANEJO", "1").strip().lower() not in ("0", "false")

ANO_MINIMO = int(os.getenv("ANO_MINIMO", "1950").strip())
INCLUIR_SEM_ANO = os.getenv("INCLUIR_SEM_ANO", "1").strip().lower() not in ("0", "false")


MAX_CANDIDATAS_ENV = os.getenv("MAX_CANDIDATAS", "").strip()
MAX_CANDIDATAS = int(MAX_CANDIDATAS_ENV) if MAX_CANDIDATAS_ENV else None


PROGRESS_EVERY = int(os.getenv("PROGRESS_EVERY", "50").strip())


SITEMAP_HEARTBEAT_EVERY = int(os.getenv("SITEMAP_HEARTBEAT_EVERY", "300000").strip())

OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "letras_sertanejo")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", r"D:\Downloads")

DELAY_MIN = float(os.getenv("DELAY_MIN", "0.8"))
DELAY_MAX = float(os.getenv("DELAY_MAX", "1.6"))

TIMEOUT = int(os.getenv("TIMEOUT", "30").strip())
RETRIES = int(os.getenv("RETRIES", "3").strip())



SITEMAP_INCLUDE = os.getenv("SITEMAP_INCLUDE", r"(letras|musicas|lyrics|songs)")

SITEMAP_EXCLUDE = os.getenv("SITEMAP_EXCLUDE", r"(artistas|artists|albums|discografia|bio|noticias|fotos|videos|playlists)")

INCLUDE_RE = re.compile(SITEMAP_INCLUDE, re.IGNORECASE)
EXCLUDE_RE = re.compile(SITEMAP_EXCLUDE, re.IGNORECASE)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


MUSICA_URL_RE = re.compile(
    r"^https?://(www\.)?letras\.mus\.br/[a-z0-9-]+/[a-z0-9-]+/?$",
    re.IGNORECASE
)





class SeenCache:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("CREATE TABLE IF NOT EXISTS seen_urls (url TEXT PRIMARY KEY)")
        self._pending = 0

    def add_if_new(self, url: str) -> bool:
        cur = self.conn.execute("INSERT OR IGNORE INTO seen_urls(url) VALUES (?)", (url,))
        self._pending += 1
        if self._pending >= 5000:
            self.conn.commit()
            self._pending = 0
        return cur.rowcount == 1

    def close(self):
        try:
            self.conn.commit()
        except Exception:
            pass
        self.conn.close()





def http_get(url: str) -> Optional[requests.Response]:
    for i in range(RETRIES):
        try:
            r = SESSION.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except Exception:
            time.sleep(1.0 + (i * 0.8) + random.uniform(0.2, 0.8))
    return None


def get_soup(url: str) -> Optional[BeautifulSoup]:
    r = http_get(url)
    if not r:
        return None
    return BeautifulSoup(r.text, "html.parser")





def iterar_locs_de_sitemap_bytes(content: bytes, heartbeat_every: int = 0) -> Iterable[str]:
    if content[:2] == b"\x1f\x8b":
        stream = gzip.GzipFile(fileobj=io.BytesIO(content))
        reader = stream
    else:
        reader = io.BytesIO(content)

    buf = ""
    count = 0

    for raw_line in reader:
        try:
            line = raw_line.decode("utf-8", errors="ignore")
        except Exception:
            continue

        buf += line


        while True:
            a = buf.find("<loc>")
            if a < 0:

                if len(buf) > 8192:
                    buf = buf[-2048:]
                break
            b = buf.find("</loc>", a + 5)
            if b < 0:

                if len(buf) > 65536:
                    buf = buf[a:]
                break

            loc = buf[a + 5 : b].strip()
            buf = buf[b + 6 :]

            if loc:
                count += 1
                if heartbeat_every and (count % heartbeat_every == 0):
                    print(f"   💓 sitemap heartbeat: {count} <loc> lidos...", flush=True)
                yield loc


def baixar_locs(url: str) -> List[str]:
    r = http_get(url)
    if not r:
        return []
    return list(iterar_locs_de_sitemap_bytes(r.content, heartbeat_every=0))


def filtrar_e_ordenar_sitemaps(sitemap_urls: List[str]) -> Tuple[List[str], List[str]]:
    selected = []
    skipped = []

    for u in sitemap_urls:
        if EXCLUDE_RE.search(u):
            skipped.append(u)
            continue
        if INCLUDE_RE.search(u):
            selected.append(u)
        else:
            skipped.append(u)

    if not selected:

        selected = [u for u in sitemap_urls if not EXCLUDE_RE.search(u)]
        skipped = [u for u in sitemap_urls if u not in selected]


    selected.sort(key=lambda x: (0 if INCLUDE_RE.search(x) else 1, x))
    return selected, skipped


def iterar_urls_candidatas_sitemap(seen: SeenCache) -> Iterable[str]:
    print("🌐 Baixando sitemap index...", flush=True)
    sitemap_urls = baixar_locs(SITEMAP_INDEX_URL)
    if not sitemap_urls:
        print("❌ Falha ao baixar/parsear sitemap index.", flush=True)
        return

    selected, skipped = filtrar_e_ordenar_sitemaps(sitemap_urls)

    print(f"🗂️ Sitemaps no index: {len(sitemap_urls)}", flush=True)
    print(f"✅ Sitemaps selecionados p/ músicas: {len(selected)} | pulados: {len(skipped)}", flush=True)

    candidatas_count = 0

    for idx, sm_url in enumerate(selected, start=1):
        print(f"📥 Baixando sitemap {idx}/{len(selected)}: {sm_url}", flush=True)
        rr = http_get(sm_url)
        if not rr:
            print("   ⚠️ Falha no download (timeout/retry). Pulando.", flush=True)
            continue

        print("   🧩 Parseando <loc> (streaming)...", flush=True)
        novos_neste_sitemap = 0
        total_locs_neste_sitemap = 0

        try:
            for loc in iterar_locs_de_sitemap_bytes(rr.content, heartbeat_every=SITEMAP_HEARTBEAT_EVERY):
                total_locs_neste_sitemap += 1

                if not loc:
                    continue
                u = loc.strip()


                if not MUSICA_URL_RE.match(u):
                    continue

                if not u.endswith("/"):
                    u += "/"

                if not seen.add_if_new(u):
                    continue

                candidatas_count += 1
                novos_neste_sitemap += 1

                if MAX_CANDIDATAS is not None and candidatas_count >= MAX_CANDIDATAS:
                    print(f"🧪 MAX_CANDIDATAS atingido ({MAX_CANDIDATAS}). Parando varredura.", flush=True)
                    return

                yield u

        except Exception as e:
            print(f"   ⚠️ Erro parseando sitemap: {e}. Pulando.", flush=True)
            continue

        print(
            f"   ✅ Concluído {idx}/{len(selected)} | locs lidos: {total_locs_neste_sitemap} | "
            f"novas candidatas (músicas) aqui: {novos_neste_sitemap} | total candidatas: {candidatas_count}",
            flush=True
        )





def _lower(s: str) -> str:
    return (s or "").strip().lower()


def detectar_generos_jsonld(soup: BeautifulSoup) -> List[str]:
    out: List[str] = []
    scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
    for sc in scripts:
        if not sc.string:
            continue
        try:
            data = json.loads(sc.string)
        except Exception:
            continue

        objs: List[dict] = []
        if isinstance(data, dict):
            objs = [data]
        elif isinstance(data, list):
            objs = [d for d in data if isinstance(d, dict)]

        for d in objs:
            g = d.get("genre")
            if isinstance(g, str):
                out.append(g)
            elif isinstance(g, list):
                out.extend([x for x in g if isinstance(x, str)])
    return [x for x in out if x.strip()]


def pagina_eh_sertanejo(soup: BeautifulSoup) -> bool:

    generos = [_lower(g) for g in detectar_generos_jsonld(soup)]
    if generos:
        return any("sertanejo" in g for g in generos)


    if soup.find("a", href=re.compile(r"/estilos/sertanejo/?", re.IGNORECASE)):
        return True


    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        if "sertanejo" in meta.get("content", "").lower():
            return True

    return False





def extrair_ano(soup: BeautifulSoup) -> Optional[int]:
    scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
    for sc in scripts:
        if not sc.string:
            continue
        try:
            data = json.loads(sc.string)
        except Exception:
            continue

        objs: List[dict] = []
        if isinstance(data, dict):
            objs = [data]
        elif isinstance(data, list):
            objs = [d for d in data if isinstance(d, dict)]

        for d in objs:
            for campo in ("datePublished", "releaseDate", "dateCreated", "uploadDate"):
                if campo in d:
                    m = re.search(r"\b(19|20)\d{2}\b", str(d[campo]))
                    if m:
                        return int(m.group())
    return None


def extrair_titulo_artista_pagina(soup: BeautifulSoup, url: str) -> Tuple[str, str]:
    parts = [p for p in url.strip("/").split("/") if p]
    artista_prev = parts[-2].replace("-", " ").title() if len(parts) >= 2 else ""
    titulo_prev = parts[-1].replace("-", " ").title() if len(parts) >= 1 else ""

    titulo = titulo_prev
    h1 = soup.find("h1", class_="textStyle-primary") or soup.find("h1")
    if h1:
        t = h1.get_text(strip=True)
        if t:
            titulo = t

    artista = artista_prev
    try:
        if h1:
            a = h1.find_next("a", href=True)
            if a and a.get_text(strip=True):
                artista = a.get_text(strip=True)
    except Exception:
        pass

    return titulo, artista


def achar_bloco_letra(soup: BeautifulSoup) -> Optional[Tag]:
    seletores = [
        ".lyric-original",
        "div.lyric-original",
        "[class*='lyric']",
        "div.lyric",
        ".letra",
        "article"
    ]
    for sel in seletores:
        e = soup.select_one(sel)
        if e:
            txt = e.get_text(" ", strip=True)
            if txt and len(txt) > 200:
                return e
    return None


def html_para_texto_com_blocos(letra_elem: Tag) -> str:
    ps = letra_elem.find_all("p")
    ps_textos = []
    for p in ps:
        t = p.get_text("\n", strip=True)
        t = t.replace("\r\n", "\n").replace("\r", "\n").strip()
        if t:
            ps_textos.append(t)
    if ps_textos:
        return "\n\n".join(ps_textos)

    chunks = []
    prev_was_br = False

    for node in letra_elem.descendants:
        if isinstance(node, NavigableString):
            s = str(node).replace("\r\n", "\n").replace("\r", "\n")
            if s.strip():
                chunks.append(s)
                prev_was_br = False
        elif isinstance(node, Tag) and node.name == "br":
            if prev_was_br:
                chunks.append("\n\n")
                prev_was_br = False
            else:
                chunks.append("\n")
                prev_was_br = True

    return "".join(chunks).replace("\r\n", "\n").replace("\r", "\n")


def limpar_texto_preservando_linhas_em_branco(texto: str) -> str:
    if not texto:
        return ""
    t = texto.replace("\r\n", "\n").replace("\r", "\n")

    linhas = []
    for line in t.split("\n"):
        if line.strip() == "":
            linhas.append("")
        else:
            line = re.sub(r"[ \t]+", " ", line).strip()
            linhas.append(line)

    t = "\n".join(linhas)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def quebrar_em_estrofes(texto_limpo: str) -> List[str]:
    if not texto_limpo:
        return []
    blocos = [b.strip() for b in re.split(r"\n\s*\n+", texto_limpo) if b.strip()]
    estrofes = []
    for b in blocos:
        linhas = [l.strip() for l in b.split("\n") if l.strip()]
        est = " ".join(linhas)
        est = re.sub(r"\s+", " ", est).strip()
        if est:
            estrofes.append(est)
    return estrofes


def extrair_estrofes_da_pagina(soup: BeautifulSoup) -> Optional[List[str]]:
    letra_elem = achar_bloco_letra(soup)
    if not letra_elem:
        return None
    texto_raw = html_para_texto_com_blocos(letra_elem)
    texto_limpo = limpar_texto_preservando_linhas_em_branco(texto_raw)
    estrofes = quebrar_em_estrofes(texto_limpo)
    return estrofes if estrofes else None





def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_sitemap_{ts}.csv")
    db_path = os.path.join(OUTPUT_DIR, "seen_urls_cache.sqlite")

    cols = ["ranking_posicao", "musica", "tag_musica", "tag_trecho", "ESTROFE", "artista", "ano", "ntokens", "url"]

    print("=" * 90, flush=True)
    print("🎵 Letras.mus.br | SOMENTE SERTANEJO | sitemap streaming + filtro de sitemaps | 1 linha por ESTROFE", flush=True)
    print("=" * 90, flush=True)
    print(f"✅ Salva em: {OUTPUT_DIR}", flush=True)
    print(f"🧠 Cache SQLite: {db_path}", flush=True)
    print(f"🧪 MAX_CANDIDATAS={MAX_CANDIDATAS} | PROGRESS_EVERY={PROGRESS_EVERY}", flush=True)
    print(f"💓 SITEMAP_HEARTBEAT_EVERY={SITEMAP_HEARTBEAT_EVERY}", flush=True)
    print(f"✅ EXIGIR_CONFIRMACAO_SERTANEJO={int(EXIGIR_CONFIRMACAO_SERTANEJO)}", flush=True)
    print(f"📅 ANO_MINIMO={ANO_MINIMO} | INCLUIR_SEM_ANO={int(INCLUIR_SEM_ANO)}", flush=True)
    print(f"🔎 SITEMAP_INCLUDE={SITEMAP_INCLUDE}", flush=True)
    print(f"🚫 SITEMAP_EXCLUDE={SITEMAP_EXCLUDE}", flush=True)
    print("", flush=True)

    seen = SeenCache(db_path)

    pages_baixadas = 0
    ok_musicas = 0
    ok_estrofes = 0
    puladas_nao_sertanejo = 0
    puladas_ano = 0
    falhas = 0

    start = time.time()

    try:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()

            for url in iterar_urls_candidatas_sitemap(seen):
                pages_baixadas += 1

                soup = get_soup(url)
                if not soup:
                    falhas += 1
                    continue

                if not pagina_eh_sertanejo(soup):
                    if EXIGIR_CONFIRMACAO_SERTANEJO:
                        puladas_nao_sertanejo += 1
                        continue

                titulo, artista = extrair_titulo_artista_pagina(soup, url)
                estrofes = extrair_estrofes_da_pagina(soup)
                if not estrofes:
                    falhas += 1
                    continue

                ano = extrair_ano(soup)
                if ano is None and not INCLUIR_SEM_ANO:
                    puladas_ano += 1
                    continue
                if ano is not None and ano < ANO_MINIMO:
                    puladas_ano += 1
                    continue

                ok_musicas += 1
                for est in estrofes:
                    writer.writerow({
                        "ranking_posicao": pages_baixadas,
                        "musica": titulo,
                        "tag_musica": f"{titulo} {artista}",
                        "tag_trecho": f"{titulo} {artista}",
                        "ESTROFE": est,
                        "artista": artista,
                        "ano": ano if ano is not None else "",
                        "ntokens": len(est.split()),
                        "url": url
                    })
                    ok_estrofes += 1

                if pages_baixadas % PROGRESS_EVERY == 0:
                    elapsed = time.time() - start
                    print(
                        f"📌 pages baixadas: {pages_baixadas} | ok músicas: {ok_musicas} | ok estrofes: {ok_estrofes} | "
                        f"puladas não-sertanejo: {puladas_nao_sertanejo} | puladas ano: {puladas_ano} | falhas: {falhas} | {elapsed:.1f}s",
                        flush=True
                    )

                time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    except KeyboardInterrupt:
        print("\n🛑 Interrompido pelo usuário (CTRL+C). Cache mantém progresso.", flush=True)
    finally:
        seen.close()

    elapsed = time.time() - start
    print("\n" + "=" * 90, flush=True)
    print("✅ FINALIZADO", flush=True)
    print(f"📄 CSV: {out_csv}", flush=True)
    print(f"📥 Páginas baixadas: {pages_baixadas}", flush=True)
    print(f"🎵 Músicas Sertanejas OK: {ok_musicas}", flush=True)
    print(f"🧩 Estrofes gravadas: {ok_estrofes}", flush=True)
    print(f"⏭️ Puladas não-sertanejo: {puladas_nao_sertanejo}", flush=True)
    print(f"⏭️ Puladas por ano: {puladas_ano}", flush=True)
    print(f"❌ Falhas: {falhas}", flush=True)
    print(f"⏱️ Tempo: {elapsed:.1f}s", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
