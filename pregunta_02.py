#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parte 2: Web Scraping (Portal Inmobiliario / Huechuraba)

Requisitos del enunciado (Parte 2):
- Implementar un scraper para obtener datos de casas y departamentos en Huechuraba desde
  Portal Inmobiliario (precios y metros cuadrados).

Este script está diseñado para ser:
1) Ético: respeta robots.txt y aplica throttling; no usa técnicas de evasión.
2) Reproducible: genera logs, dumps HTML ante bloqueo, y outputs CSV.
3) Fail-fast: si detecta 401/403/429 o señales de captcha, se detiene y deja evidencia.

Uso (simple):
    python pregunta_02.py

Uso (configurable):
    python pregunta_02.py --max-pages 3 --max-items 120 --throttle-min 2 --throttle-max 4 --headless 1

Outputs:
- data/out/portal_huechuraba_*.csv      (filas extraídas, aunque sean parciales)
- data/out/metrics_*.csv               (métricas del cuadro 2.1)
- data/logs/scrape_*.log               (bitácora)
- data/raw/blocked_*.html              (dumps ante bloqueo/captcha)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Response


# -----------------------------
# Constantes del sitio objetivo
# -----------------------------
BASE = "https://www.portalinmobiliario.com"
ROBOTS_URL = f"{BASE}/robots.txt"

LISTING_TEMPLATES = [
    f"{BASE}/venta/{{tipo}}/{{comuna}}-{{region}}",
]

# Regex robustos para UF y m2 (best-effort)
UF_RE = re.compile(r"\bUF\s*([\d\.\,]+)", re.IGNORECASE)
M2_RE = re.compile(r"(\d+(?:[\,\.]\d+)?)\s*m2\b", re.IGNORECASE)

USER_AGENT_GROUP = "*"  # para evaluación de robots

# -----------------------------
# Configuración de ejecución
# -----------------------------
@dataclass
class ScrapeConfig:
    comuna: str
    region: str
    max_pages: int
    max_items_per_type: int
    throttle_min_s: float
    throttle_max_s: float
    headless: bool
    timeout_ms: int
    debug_dump: bool

# -----------------------------
# Robots.txt (parser simple)
# -----------------------------
class RobotsRules:
    """
    Parser simple de robots.txt:
    - Soporta grupos User-agent
    - Soporta Allow / Disallow
    - Aplica "longest match" (aprox estándar de facto)
    """

    def __init__(self, raw: str):
        self.raw = raw
        # rules[ua] = list of (action, path_prefix)
        self.rules: Dict[str, List[Tuple[str, str]]] = {}
        self._parse()

    def _parse(self) -> None:
        current_agents: List[str] = []
        for line in self.raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # elimina comentarios inline
            if "#" in line:
                line = line.split("#", 1)[0].strip()

            if ":" not in line:
                continue

            key, val = [x.strip() for x in line.split(":", 1)]
            key_lower = key.lower()

            if key_lower == "user-agent":
                ua = val
                current_agents = [ua]
                self.rules.setdefault(ua, [])
            elif key_lower in ("allow", "disallow"):
                path = val or "/"
                for ua in current_agents or [USER_AGENT_GROUP]:
                    self.rules.setdefault(ua, []).append((key_lower, path))

    def can_fetch(self, path: str, user_agent: str = USER_AGENT_GROUP) -> bool:
        """
        Retorna True si path es permitido según robots.txt (mejor-esfuerzo).
        """
        if not path.startswith("/"):
            path = "/" + path

        candidates: List[Tuple[str, str]] = []
        if user_agent in self.rules:
            candidates.extend(self.rules[user_agent])
        if USER_AGENT_GROUP in self.rules and user_agent != USER_AGENT_GROUP:
            candidates.extend(self.rules[USER_AGENT_GROUP])

        # Sin reglas -> permitido
        if not candidates:
            return True

        # Longest-prefix match
        best_len = -1
        best_action = "allow"
        for action, rule_path in candidates:
            if rule_path and path.startswith(rule_path):
                if len(rule_path) > best_len:
                    best_len = len(rule_path)
                    best_action = action

        return best_action != "disallow"

# -----------------------------
# Utilidades de I/O y logging
# -----------------------------
def ensure_dirs(root: Path) -> Dict[str, Path]:
    data = root / "data"
    raw = data / "raw"
    out = data / "out"
    logs = data / "logs"
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return {"data": data, "raw": raw, "out": out, "logs": logs}

def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def log_line(log_path: Path, msg: str) -> None:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def polite_sleep(cfg: ScrapeConfig) -> None:
    # Throttling (mínimo impacto). No concurrencia.
    time.sleep(random.uniform(cfg.throttle_min_s, cfg.throttle_max_s))

def fetch_robots(log_path: Path, timeout_s: int = 20) -> RobotsRules:
    """
    Descarga robots.txt (usando urllib estándar) y retorna reglas parseadas.
    """
    import urllib.request

    log_line(log_path, f"Descargando robots.txt: {ROBOTS_URL}")
    req = urllib.request.Request(ROBOTS_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    log_line(log_path, "robots.txt descargado OK.")
    return RobotsRules(raw)

def build_listing_urls(tipo: str, comuna: str, region: str, page: int) -> List[str]:
    """
    Construye URLs de listado:
    - Page 1: base
    - Page > 1: intenta paginación tipo "_Desde_" (frecuentemente permitida en robots.txt)
    Nota: el offset real puede variar; sirve como intento y se documenta en logs.
    """
    base_url = LISTING_TEMPLATES[0].format(tipo=tipo, comuna=comuna, region=region)
    if page <= 1:
        return [base_url]

    # Offset aproximado, típico de 48 items por página (best-effort)
    offset = 48 * (page - 1) + 1
    return [f"{base_url}_Desde_{offset}"]

def save_dump(raw_dir: Path, prefix: str, url: str, status: Optional[int], html: str) -> Path:
    stamp = now_stamp()
    safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", prefix)[:80]
    fname = f"{safe}_{stamp}_status{status or 'NA'}.html"
    p = raw_dir / fname
    with p.open("w", encoding="utf-8") as f:
        f.write(f"<!-- URL: {url} -->\n")
        f.write(html)
    return p

def response_status(resp: Optional[Response]) -> Optional[int]:
    try:
        return resp.status if resp is not None else None
    except Exception:
        return None

def fetch_page_html(page, url: str, cfg: ScrapeConfig, log_path: Path) -> Tuple[Optional[int], str]:
    """
    Navega a la URL con Playwright y devuelve (status_code, html).
    """
    log_line(log_path, f"GET {url}")
    resp = page.goto(url, wait_until="domcontentloaded", timeout=cfg.timeout_ms)
    status = response_status(resp)

    # Espera breve adicional para contenido base (sin insistencia)
    page.wait_for_timeout(800)
    html = page.content()
    log_line(log_path, f"Status={status} len(html)={len(html)}")
    return status, html

# -----------------------------
# Parsing (best-effort)
# -----------------------------
def parse_items_from_jsonld(soup: BeautifulSoup) -> List[dict]:
    """
    Intenta extraer estructuras JSON-LD (si existen).
    Si el sitio entrega ItemList, puede haber itemListElement.
    """
    items: List[dict] = []
    for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
        text = (s.string or "").strip()
        if not text:
            continue
        try:
            data = json.loads(text)
        except Exception:
            continue

        candidates = data if isinstance(data, list) else [data]
        for obj in candidates:
            if not isinstance(obj, dict):
                continue
            if obj.get("@type") == "ItemList" and isinstance(obj.get("itemListElement"), list):
                for el in obj["itemListElement"]:
                    if isinstance(el, dict):
                        items.append(el)
    return items

def extract_price_uf_and_m2_from_text(text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extrae UF y m2 desde texto libre (JSON o DOM).
    Normaliza formatos 1.234,56 o 1,234.56 a float (best-effort).
    """
    uf_val = None
    m = UF_RE.search(text)
    if m:
        num = m.group(1).replace(".", "").replace(",", ".")
        try:
            uf_val = float(num)
        except Exception:
            uf_val = None

    m2_val = None
    mm = M2_RE.search(text)
    if mm:
        num = mm.group(1).replace(",", ".")
        try:
            m2_val = float(num)
        except Exception:
            m2_val = None

    return uf_val, m2_val

def parse_listings(html: str, tipo: str, url: str) -> List[dict]:
    """
    Parser robusto (best-effort):
    1) Prefiere JSON-LD.
    2) Fallback: inspecciona texto del DOM en bloques grandes buscando UF y m2.

    Retorna filas con columnas:
    - tipo: casa|departamento
    - source_url: listado fuente
    - price_uf: float o NaN
    - m2: float o NaN
    - raw_hint: snippet textual (solo para debug)
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: List[dict] = []

    # 1) JSON-LD
    jsonld_items = parse_items_from_jsonld(soup)
    if jsonld_items:
        for it in jsonld_items:
            blob = json.dumps(it, ensure_ascii=False)
            uf, m2 = extract_price_uf_and_m2_from_text(blob)
            rows.append(
                {
                    "tipo": tipo,
                    "source_url": url,
                    "price_uf": uf,
                    "m2": m2,
                    "raw_hint": None,
                }
            )

    # 2) Fallback DOM (sin depender de clases CSS específicas)
    if not rows:
        candidates = soup.find_all(["article", "li", "div"], limit=700)
        for c in candidates:
            text = c.get_text(" ", strip=True)
            if "UF" not in text:
                continue
            uf, m2 = extract_price_uf_and_m2_from_text(text)
            if uf is None and m2 is None:
                continue
            rows.append(
                {
                    "tipo": tipo,
                    "source_url": url,
                    "price_uf": uf,
                    "m2": m2,
                    "raw_hint": text[:250],
                }
            )

    # Deduplicación simple
    uniq: List[dict] = []
    seen = set()
    for r in rows:
        key = (r["tipo"], r.get("price_uf"), r.get("m2"), r["source_url"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    return uniq

# -----------------------------
# Métricas (cuadro 2.1)
# -----------------------------
def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna un DataFrame con el cuadro del enunciado (best-effort):
    - Conteo por tipo
    - Mediana UF por tipo
    - Promedio UF por tipo
    - Promedio UF/m2 por tipo (con filas válidas UF y m2)
    """
    def safe_median(s: pd.Series) -> float:
        s2 = s.dropna()
        return float(s2.median()) if len(s2) else float("nan")

    def safe_mean(s: pd.Series) -> float:
        s2 = s.dropna()
        return float(s2.mean()) if len(s2) else float("nan")

    metrics = []

    for tipo in ["casa", "departamento"]:
        sub = df[df["tipo"] == tipo].copy()
        n = int(len(sub))
        med = safe_median(sub["price_uf"])
        avg = safe_mean(sub["price_uf"])

        sub_valid = sub.dropna(subset=["price_uf", "m2"])
        uf_m2 = safe_mean(sub_valid["price_uf"] / sub_valid["m2"]) if len(sub_valid) else float("nan")

        if tipo == "casa":
            metrics.append(("Número de casas scrapeadas (#)", n))
            metrics.append(("Mediana de precio de las casas (UF)", med))
            metrics.append(("Promedio de precio de las casas (UF)", avg))
            metrics.append(("Precio por m2 de casas (UF/m2)", uf_m2))
        else:
            metrics.append(("Número de departamentos scrapeados (#)", n))
            metrics.append(("Mediana de precio de los departamentos (UF)", med))
            metrics.append(("Promedio de precio de los departamentos (UF)", avg))
            metrics.append(("Precio por m2 de departamento (UF/m2)", uf_m2))

    return pd.DataFrame(metrics, columns=["Métrica", "Valor"])

# -----------------------------
# Main (CLI)
# -----------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parte 2 Web Scraping (ético, fail-fast) — Portal Inmobiliario Huechuraba"
    )
    # Defaults:"python pregunta_02.py"
    parser.add_argument("--comuna", default="huechuraba", help="slug comuna (default: huechuraba)")
    parser.add_argument("--region", default="metropolitana", help="slug región (default: metropolitana)")
    parser.add_argument("--max-pages", type=int, default=3, help="máximo páginas por tipo (default: 3)")
    parser.add_argument("--max-items", type=int, default=150, help="máximo items por tipo (default: 150)")
    parser.add_argument("--throttle-min", type=float, default=2.0, help="sleep mínimo entre requests (s)")
    parser.add_argument("--throttle-max", type=float, default=4.0, help="sleep máximo entre requests (s)")
    parser.add_argument("--headless", type=int, default=1, help="1=headless, 0=con ventana (debug)")
    parser.add_argument("--timeout-ms", type=int, default=25000, help="timeout navegación playwright (ms)")
    parser.add_argument("--debug-dump", type=int, default=1, help="1=guardar dumps HTML ante bloqueo/falla")
    args = parser.parse_args()

    cfg = ScrapeConfig(
        comuna=args.comuna,
        region=args.region,
        max_pages=args.max_pages,
        max_items_per_type=args.max_items,
        throttle_min_s=args.throttle_min,
        throttle_max_s=args.throttle_max,
        headless=bool(args.headless),
        timeout_ms=args.timeout_ms,
        debug_dump=bool(args.debug_dump),
    )

    root = Path.cwd()
    paths = ensure_dirs(root)
    stamp = now_stamp()
    log_path = paths["logs"] / f"scrape_{stamp}.log"

    log_line(log_path, "Iniciando scraping (ético, fail-fast).")
    log_line(log_path, f"Config: {cfg}")
    log_line(log_path, "Nota: Si hay 403/captcha, el script se detiene y guarda evidencia en data/raw/.")

    # 1) robots.txt (se usa para decidir si visitar o no ciertas rutas)
    robots = fetch_robots(log_path)

    all_rows: List[dict] = []

    # 2) Playwright: un solo navegador, una sola pestaña, sin paralelismo
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=cfg.headless)
        context = browser.new_context()
        page = context.new_page()

        # Orden solicitado por el usuario: primero deptos, luego casas
        for tipo in ["departamento", "casa"]:
            log_line(log_path, f"[{tipo}] Iniciando...")

            blocked = False
            collected_for_type = 0

            for page_i in range(1, cfg.max_pages + 1):
                candidate_urls = build_listing_urls(tipo, cfg.comuna, cfg.region, page_i)

                for url in candidate_urls:
                    # Respeto robots.txt (si el path está desautorizado, se salta)
                    path = url.replace(BASE, "", 1)
                    if not robots.can_fetch(path, USER_AGENT_GROUP):
                        log_line(log_path, f"[{tipo}] SKIP por robots.txt: {path}")
                        continue

                    polite_sleep(cfg)

                    status, html = fetch_page_html(page, url, cfg, log_path)

                    # Detección de bloqueo (WAF/anti-bot)
                    is_captcha = "captcha" in (html or "").lower()
                    is_denied = (html and ("Access Denied" in html or "access denied" in html.lower()))
                    if status in (401, 403, 429) or is_captcha or is_denied:
                        log_line(log_path, f"[{tipo}] BLOQUEO detectado (status={status}, captcha={is_captcha}). Deteniendo tipo.")
                        if cfg.debug_dump:
                            dump_path = save_dump(paths["raw"], f"blocked_{tipo}_p{page_i}", url, status, html)
                            log_line(log_path, f"[{tipo}] Dump guardado: {dump_path}")
                        blocked = True
                        break

                    # Parseo de listados
                    rows = parse_listings(html, tipo, url)
                    log_line(log_path, f"[{tipo}] items parseados en página: {len(rows)}")
                    all_rows.extend(rows)

                    # Límite por tipo
                    collected_for_type = sum(1 for r in all_rows if r["tipo"] == tipo)
                    if collected_for_type >= cfg.max_items_per_type:
                        log_line(log_path, f"[{tipo}] alcanzado max_items_per_type={cfg.max_items_per_type}.")
                        break

                if blocked or collected_for_type >= cfg.max_items_per_type:
                    break

            log_line(log_path, f"[{tipo}] Finalizado. Total tipo={collected_for_type} blocked={blocked}")

        context.close()
        browser.close()

    # 3) Guardar dataset extraído (aunque sea parcial)
    df = pd.DataFrame(all_rows)
    out_csv = paths["out"] / f"portal_huechuraba_{stamp}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    log_line(log_path, f"CSV guardado: {out_csv} (rows={len(df)})")

    # 4) Si no hubo filas, igual dejamos evidencia en logs para el informe
    if len(df) == 0:
        log_line(log_path, "No se obtuvieron filas. Revisa logs/dumps para justificar bloqueo en el informe.")
        log_line(log_path, f"LOG: {log_path}")
        return 2

    # 5) Limpieza/normalización numérica
    df["price_uf"] = pd.to_numeric(df.get("price_uf"), errors="coerce")
    df["m2"] = pd.to_numeric(df.get("m2"), errors="coerce")

    # 6) Métricas 2.1
    metrics = compute_metrics(df)
    metrics_csv = paths["out"] / f"metrics_{stamp}.csv"
    metrics.to_csv(metrics_csv, index=False, encoding="utf-8")

    log_line(log_path, f"Métricas guardadas: {metrics_csv}")
    log_line(log_path, "Resumen métricas (tabla):")
    for _, row in metrics.iterrows():
        log_line(log_path, f" - {row['Métrica']}: {row['Valor']}")
    log_line(log_path, f"LOG: {log_path}")
    log_line(log_path, "Proceso finalizado.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
