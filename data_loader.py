# -*- coding: utf-8 -*-
"""
data_loader.py
---------------
Загрузчик Excel-данных для каталога TPMS.

Файлы каталога (по умолчанию ищутся в ./data и DATA_DIR):

- brands.xlsx
    Бренды.
    Ожидаемые поля (любые из перечисленных, имена не чувствительны к регистру):
        id / brand_id / bid
        pagetitle / name / brand / title

- brandmodels.xlsx
    Модели.
    Ожидаемые поля:
        id              – идентификатор модели (model_id)
        parent / brand  – id бренда
        pagetitle / ... – название модели

- applicability.xlsx
    Применяемость (связка Бренд + Модель + Годы + Артикул).
    Ожидаемые поля:
        brand / brand_id / bid / id   – id бренда
        model                         – id модели
        year OR (year_start/year_end) – год или диапазон
        article / sku / productid     – артикул (SKU) товара (опционально)

- products.xlsx
    Карточки товаров.
    Ожидаемые поля:
        id               – RowID карточки
        article / sku    – артикул
        price_*          – цена
        made_in / ...    – страна
        images / pics    – изображения (через "||")
        market_image     – основное изображение
        market_title     – маркетинговый заголовок
        pole1 / message  – текст описания

NEW: brand_aliases.xlsx (опционально)
    Алиасы брендов (для поддержки "BMW / бмв / бэмв" и т.д.)
    Поля:
        brand_id / id / bid
        alias / name / title
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ===== Импорты для Excel =====
try:
    import openpyxl  # для .xlsx
except Exception:  # pragma: no cover - мягкое падение
    openpyxl = None  # type: ignore

try:
    import xlrd  # для .xls (нужна версия <2)
except Exception:  # pragma: no cover
    xlrd = None  # type: ignore


# ===== Конфигурация поиска файлов =====

EXTS = [".xlsx", ".xls"]

BASENAMES: Dict[str, List[str]] = {
    "brands": ["brands (6)", "brands"],
    "brandmodels": ["brandmodels (2)", "brandmodels"],
    "products": ["products (9)", "products"],
    "applicability": ["applicability (4)", "applicability"],
    # NEW: файл с алиасами брендов
    "brand_aliases": ["brand_aliases", "brandaliases", "brand_alias"],
}


def _default_search_dirs(base_dir: Optional[str | Path]) -> List[Path]:
    """Формирует список директорий для поиска в порядке приоритета."""
    dirs: List[Path] = []
    if base_dir:
        dirs.append(Path(base_dir).expanduser())

    env_dir = os.getenv("DATA_DIR")
    if env_dir:
        dirs.append(Path(env_dir).expanduser())

    try:
        here = Path(__file__).resolve().parent
        dirs.append(here / "data")
        dirs.append(here)
    except Exception:
        pass

    dirs.append(Path.cwd())

    # Уникализируем порядок
    seen: set[str] = set()
    out: List[Path] = []
    for d in dirs:
        p = Path(d)
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


# ===== Утилиты нормализации =====

def _norm_header(s: Any) -> str:
    """Нормализует заголовок колонки к нижнему регистру a-z0-9_."""
    s = "" if s is None else str(s)
    s = s.strip().lower()
    # пробелы / тире / слэши -> _
    s = re.sub(r"[\s\-\/]+", "_", s)
    # оставляем только латиницу + цифры + _
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def _as_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _as_int(v: Any) -> Optional[int]:
    if v in (None, ""):
        return None
    try:
        if isinstance(v, float):
            return int(v)
        return int(float(str(v).strip()))
    except Exception:
        return None


def _get(row: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Достаёт первое непустое значение по списку возможных имён поля
    (с нормализацией заголовка).
    """
    for k in keys:
        nk = _norm_header(k)
        if nk in row and row[nk] not in (None, ""):
            return row[nk]
    return default


# ===== Чтение Excel =====

def _read_excel(path: Optional[Path]) -> List[Dict[str, Any]]:
    """
    Читает .xlsx/.xls и возвращает список dict с НОРМАЛИЗОВАННЫМИ заголовками.
    Если файла нет – возвращает [].
    """
    if not path or not path.exists():
        return []
    suf = path.suffix.lower()

    if suf == ".xlsx":
        if openpyxl is None:
            raise RuntimeError(
                "Нужен openpyxl для .xlsx (pip install openpyxl)")
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        ws = wb.worksheets[0]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return []
        headers = [_norm_header(h) for h in rows[0]]
        out: List[Dict[str, Any]] = []
        for row in rows[1:]:
            rec = {
                headers[i]: (row[i] if i < len(row) else None)
                for i in range(len(headers))
            }
            out.append(rec)
        return out

    if suf == ".xls":
        if xlrd is None:
            raise RuntimeError("Нужен xlrd<2 для .xls (pip install 'xlrd<2')")
        book = xlrd.open_workbook(str(path))
        sh = book.sheet_by_index(0)
        if sh.nrows == 0:
            return []
        headers = [_norm_header(sh.cell_value(0, c)) for c in range(sh.ncols)]
        out = []
        for r in range(1, sh.nrows):
            rec = {headers[c]: sh.cell_value(r, c) for c in range(sh.ncols)}
            out.append(rec)
        return out

    raise ValueError(f"Неизвестное расширение: {path}")


def _resolve_paths(search_dirs: Iterable[Path]) -> Dict[str, Optional[Path]]:
    """Находит пути к excel-файлам по известным базовым именам и расширениям."""
    out: Dict[str, Optional[Path]] = {k: None for k in BASENAMES}
    for key, variants in BASENAMES.items():
        found: Optional[Path] = None
        for folder in search_dirs:
            if not folder:
                continue
            for base in variants:
                for ext in EXTS:
                    cand = Path(folder) / f"{base}{ext}"
                    if cand.exists():
                        found = cand
                        break
                if found:
                    break
            if found:
                break
        out[key] = found
    return out


# ===== Основной класс каталога =====


class Catalog:
    """
    Инкапсулирует данные каталога и индексы для быстрого подбора.

    Данные:
        - _brands: List[{id:int, name:str}]
        - _brand_by_name: dict lower(имя/алиас) -> id
        - _brand_aliases: dict brand_id -> [alias1, alias2, ...]
        - _product_by_rowid: dict RowID(int) -> карточка
        - _rowids_by_sku: dict SKU(str) -> [RowID,...]
        - _rows_by_bym: dict (brand_id, year, model_title) -> [RowID,...]
        - _years_by_brand: dict brand_id -> set(years)
        - _models_by_brand_year: dict (brand_id, year) -> set(model_titles)
    """

    def __init__(self, base_dir: Optional[str | Path] = None) -> None:
        # Где искать файлы
        self._search_dirs = _default_search_dirs(base_dir)
        self._paths = _resolve_paths(self._search_dirs)

        # Основные структуры
        self._brands: List[Dict[str, Any]] = []
        self._brand_by_name: Dict[str, int] = {}
        self._brand_aliases: Dict[int, List[str]] = {}
        self._brand_id_by_alias: Dict[str, int] = {}

        self._product_by_rowid: Dict[int, Dict[str, Any]] = {}
        self._rowids_by_sku: Dict[str, List[int]] = {}

        self._rows_by_bym: Dict[Tuple[int, int, str], List[int]] = {}
        self._years_by_brand: Dict[int, set[int]] = {}
        self._models_by_brand_year: Dict[Tuple[int, int], set[str]] = {}

        # Для виртуальных товаров (когда артикул есть в применяемости, но нет в products)
        self._virtual_auto_dec: int = -1

        # Загружаем всё
        self._load_all()

    # ---------- Публичный API ----------

    def list_brands(self) -> List[str]:
        """Возвращает список имён брендов (официальные названия)."""
        return [b.get("name") or b.get("brand") or b.get("title") for b in self._brands]

    def list_years(self, brand_name: str) -> List[int]:
        """Список годов по имени бренда или его алиасу."""
        bid = self._brand_id_for_name(brand_name)
        if bid is None:
            return []
        return sorted(self._years_by_brand.get(bid, set()))

    def list_models(self, brand_name: str, year: int) -> List[str]:
        """Список моделей для бренда/года."""
        bid = self._brand_id_for_name(brand_name)
        if bid is None:
            return []
        key = (bid, int(year))
        models = sorted(self._models_by_brand_year.get(
            key, set()), key=lambda s: (len(s), s))
        return list(models)

    def products_for(self, brand_name: str, year: int, model_name: str) -> List[Dict[str, Any]]:
        """Карточки товаров для (бренд, год, модель)."""
        bid = self._brand_id_for_name(brand_name)
        if bid is None:
            return []
        key = (bid, int(year), _as_str(model_name))
        rowids = self._rows_by_bym.get(key, [])
        out: List[Dict[str, Any]] = []
        for rid in rowids:
            p = self._product_by_rowid.get(rid)
            if p is not None:
                out.append(p)
        return out

    def get_by_rowid(self, row_id: int) -> Optional[Dict[str, Any]]:
        """Возвращает карточку товара по RowID."""
        try:
            rid = int(row_id)
        except Exception:
            return None
        return self._product_by_rowid.get(rid)

    # NEW: удобный метод для ручного поиска бренда по подстроке (имя + алиасы)
    def search_brands(self, query: str) -> List[str]:
        """
        Возвращает список официальных имён брендов,
        которые матчатся по подстроке query в имени ИЛИ в алиасах.
        """
        q = _as_str(query).strip().lower()
        if not q:
            return []

        result_ids: List[int] = []

        # cache: id -> name
        name_by_id: Dict[int, str] = {
            b["id"]: (b.get("name") or "") for b in self._brands}

        for b in self._brands:
            bid = b["id"]
            name = (b.get("name") or "").lower()

            if q in name:
                result_ids.append(bid)
                continue

            aliases = self._brand_aliases.get(bid, [])
            if any(q in a for a in aliases):
                result_ids.append(bid)

        # уберём дубли и вернём имена
        seen: set[int] = set()
        out: List[str] = []
        for bid in result_ids:
            if bid in seen:
                continue
            seen.add(bid)
            nm = name_by_id.get(bid)
            if nm:
                out.append(nm)
        return out

    # ---------- Служебные методы ----------

    def reload(self) -> None:
        """Полная перезагрузка каталога с пересканированием директорий."""
        self._search_dirs = _default_search_dirs(None)
        self._paths = _resolve_paths(self._search_dirs)
        self._load_all()

    def _brand_id_for_name(self, name: str) -> Optional[int]:
        """По имени бренда или алиасу возвращает brand_id."""
        key = _as_str(name).lower()
        if not key:
            return None
        # сначала официальные имена
        bid = self._brand_by_name.get(key)
        if bid is not None:
            return bid
        # затем алиасы
        return self._brand_id_by_alias.get(key)

    # ---------- Загрузка ----------

    def _load_all(self) -> None:
        """Центральная точка загрузки всего каталога."""
        # brands и aliases должны быть загружены до применения
        self._load_brands()
        self._load_brand_aliases()
        self._load_products_rows()
        self._load_applicability_and_models()
        if not self._brands:
            # fallback, если brands.xlsx пуст
            self._build_brands_from_products()

    # -- бренды и алиасы --

    def _load_brands(self) -> None:
        path = self._paths.get("brands")
        rows = _read_excel(path)
        self._brands = []
        self._brand_by_name = {}

        for idx, r in enumerate(rows, start=1):
            bid = _as_int(_get(r, "id", "brand_id", "bid"))
            name = _as_str(_get(r, "pagetitle", "name", "brand", "title"))
            if not name:
                continue
            if bid is None:
                bid = idx
            # подчистим случай "BMW," -> "BMW"
            name = name.rstrip(",").strip()
            self._brands.append({"id": bid, "name": name})
            self._brand_by_name[name.lower()] = bid

    def _load_brand_aliases(self) -> None:
        """
        Загружает алиасы брендов из brand_aliases.xlsx (если файл есть).

        Формат строк:
            brand_id / id / bid – идентификатор бренда (из brands.xlsx)
            alias / name / title – строка-алиас (BMW, бмв, бэмв и т.п.)
        """
        self._brand_aliases = {}
        self._brand_id_by_alias = {}

        path = self._paths.get("brand_aliases")
        rows = _read_excel(path)
        if not rows or not self._brands:
            return

        # для быстрого чек-апа существования brand_id
        existing_ids = {b["id"] for b in self._brands}

        for r in rows:
            bid = _as_int(_get(r, "brand_id", "id", "bid"))
            alias_raw = _get(r, "alias", "name", "title")
            alias = _as_str(alias_raw)
            alias_norm = alias.strip().lower()
            if bid is None or not alias_norm:
                continue
            if bid not in existing_ids:
                # алиас к несуществующему бренду – игнорируем
                continue

            self._brand_aliases.setdefault(bid, []).append(alias_norm)
            # Не затёреть официальное имя, если вдруг совпало
            if alias_norm not in self._brand_by_name:
                self._brand_id_by_alias[alias_norm] = bid
                # И для удобства можно позволить использовать алиас как "имя" бренда
                # при прямом поиске по _brand_by_name:
                self._brand_by_name.setdefault(alias_norm, bid)

    # -- товары --

    def _load_products_rows(self) -> None:
        """Каждая строка products -> отдельная карточка (RowID=id)."""
        path = self._paths.get("products")
        rows = _read_excel(path)

        self._product_by_rowid = {}
        self._rowids_by_sku = {}
        self._virtual_auto_dec = -1

        for r in rows:
            rid = _as_int(_get(r, "id"))
            if rid is None:
                continue

            sku_s = _as_str(_get(r, "article", "sku", "productid", "артикул"))
            if not sku_s:
                # если нет артикула — всё равно положим карточку,
                # но без связи по SKU
                sku_s = str(rid)

            # цена – берём первую цифру, что нашли
            price = None
            for fld in ("price", "price1", "price_rur", "цена"):
                v = _get(r, fld)
                if v not in (None, ""):
                    try:
                        price = float(str(v).replace(",", "."))
                    except Exception:
                        price = None
                    break

            madein = _get(r, "made_in", "madein",
                          "manufacturer", "производитель")
            market_image = _get(r, "market_image", "marketimage")
            market_title = _get(r, "market_title", "markettitle")
            message = _get(r, "pole1", "message", "msg")

            images_raw = _get(r, "images", "pics", "pictures")
            images: Optional[List[str]] = None
            if images_raw not in (None, ""):
                parts = [
                    str(s).strip().lstrip("/")
                    for s in str(images_raw).split("||")
                    if str(s).strip()
                ]
                images = parts if parts else None

            # сохраняем все исходные поля, плюс нормализованные ключевые
            rec: Dict[str, Any] = dict(r)
            rec.update(
                {
                    "RowID": rid,
                    "ProductID": sku_s,
                    "SKU": sku_s,
                    "Title": _as_str(
                        _get(
                            r,
                            "title",
                            "pagetitle",
                            "name",
                            "product_name",
                            "наименование",
                        )
                    )
                    or sku_s,
                }
            )
            if price is not None:
                rec["Price"] = price
            if madein not in (None, ""):
                rec["MadeIn"] = _as_str(madein)
            if market_image not in (None, ""):
                rec["MarketImage"] = _as_str(market_image).lstrip("/")
            if market_title not in (None, ""):
                rec["MarketTitle"] = _as_str(market_title)
            if message not in (None, ""):
                rec["Message"] = _as_str(message)
            if images:
                rec["Images"] = images

            # индексация
            self._product_by_rowid[rid] = rec
            self._rowids_by_sku.setdefault(sku_s, []).append(rid)

    # -- применяемость + модели --

    def _load_applicability_and_models(self) -> None:
        """
        Строим индексы через brandmodels:

        - brandmodels:   id -> pagetitle, parent=brand_id
        - applicability: brand(int), model(int), годы, артикул (опционально)
        """
        self._rows_by_bym = {}
        self._years_by_brand = {}
        self._models_by_brand_year = {}

        # 1) brandmodels: id -> pagetitle, parent(brand_id)
        bm_rows = _read_excel(self._paths.get("brandmodels"))
        model_id_to_title: Dict[int, str] = {}
        model_id_to_brand: Dict[int, int] = {}
        for r in bm_rows:
            mid = _as_int(_get(r, "id"))
            bid = _as_int(_get(r, "parent", "brand_id", "brand"))
            title = _as_str(_get(r, "pagetitle", "name", "model", "title"))
            if mid is None or bid is None or not title:
                continue
            model_id_to_title[mid] = title
            model_id_to_brand[mid] = bid

        # helper: убедиться, что бренд с таким id есть в self._brands
        def _ensure_brand_exists(bid: int, hint_title: Optional[str] = None) -> None:
            for b in self._brands:
                if b["id"] == bid:
                    return
            # добавить
            name = None
            if hint_title:
                # "Acura TLX" -> "Acura"
                name = hint_title.split()[0].strip()
            if not name:
                name = f"Brand {bid}"
            self._brands.append({"id": bid, "name": name})
            self._brand_by_name[name.lower()] = bid

        # 2) applicability: brand(int), model(int), годы, article (опционально)
        app_rows = _read_excel(self._paths.get("applicability"))

        def _years_from_row(r: Dict[str, Any]) -> List[int]:
            y_single = _as_int(_get(r, "year", "год"))
            if y_single is not None:
                return [y_single]
            y_from = _as_int(
                _get(r, "year_start", "year_from", "startyear", "год_от")
            )
            y_to = _as_int(_get(r, "year_end", "year_to", "endyear", "год_до"))
            if y_from is not None and y_to is not None and y_to >= y_from:
                return list(range(int(y_from), int(y_to) + 1))
            if y_from is not None:
                return [int(y_from)]
            if y_to is not None:
                return [int(y_to)]
            return []

        for r in app_rows:
            bid = _as_int(_get(r, "brand", "brand_id", "bid", "id"))
            mid = _as_int(_get(r, "model"))
            if bid is None or mid is None:
                continue

            years = _years_from_row(r)
            if not years:
                continue

            # имя модели — из brandmodels.pag e title
            model_title = model_id_to_title.get(mid)
            if not model_title:
                # редкий случай: fallback по текстовым полям, если brandmodels пуст
                model_title = _as_str(
                    _get(r, "pagetitle", "model", "модель", "name")
                ) or str(mid)

            # гарантируем наличие бренда (если нет в brands.xlsx — добавим)
            _ensure_brand_exists(bid, hint_title=model_title)

            # артикул
            article_s = _as_str(
                _get(r, "article", "sku", "productid", "артикул")
            )
            if article_s:
                rowids = list(self._rowids_by_sku.get(article_s, []))
                if not rowids:
                    vrow = self._make_virtual_product(article_s)
                    rowids = [vrow["RowID"]]
            else:
                rowids = []

            # раскладываем по годам
            for y in years:
                key = (bid, int(y), model_title)
                if rowids:
                    self._rows_by_bym.setdefault(key, []).extend(rowids)
                self._years_by_brand.setdefault(bid, set()).add(int(y))
                self._models_by_brand_year.setdefault(
                    (bid, int(y)), set()
                ).add(model_title)

    def _build_brands_from_products(self) -> None:
        """Фоллбек: если brands.xlsx пуст — собрать бренды из products.brend."""
        seen: Dict[str, int] = {}
        brands: List[Dict[str, Any]] = []
        for rec in self._product_by_rowid.values():
            bname = _as_str(rec.get("brend") or rec.get("brand"))
            if not bname:
                continue
            key = bname.lower()
            if key not in seen:
                bid = len(brands) + 1
                seen[key] = bid
                brands.append({"id": bid, "name": bname})
        if brands:
            self._brands = brands
            self._brand_by_name = {b["name"].lower(): b["id"] for b in brands}

    def _make_virtual_product(self, sku: str) -> Dict[str, Any]:
        """
        Создаёт виртуальную карточку (если в products нет артикула),
        чтобы не рвать подбор.
        """
        rid = self._virtual_auto_dec
        self._virtual_auto_dec -= 1
        rec = {
            "RowID": rid,
            "ProductID": sku,
            "SKU": sku,
            "Title": sku,
            "Message": "Доставка от 1 шт. без предоплаты, Гарантия 2 года",
        }
        self._product_by_rowid[rid] = rec
        return rec


# ===== Быстрый smoke-test =====
if __name__ == "__main__":  # pragma: no cover
    cat = Catalog(os.getenv("DATA_DIR") or None)
    brands = cat.list_brands()
    print("Brands:", brands[:10])
    if brands:
        b = brands[0]
        ys = cat.list_years(b)
        print(f"{b} years:", ys[:10])
        if ys:
            ms = cat.list_models(b, ys[0])
            print(f"{b} {ys[0]} models:", ms[:10])
            if ms:
                prods = cat.products_for(b, ys[0], ms[0])
                print(
                    "First model products (up to 6 shown):",
                    [p.get("RowID") for p in prods[:6]],
                )
                if prods:
                    rid = prods[0]["RowID"]
                    print("get_by_rowid:", cat.get_by_rowid(rid))
