from PIL import Image
from dotenv import load_dotenv
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import (
    BufferedInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.filters import Command, CommandStart
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram import Bot, Dispatcher, F, types
import aiohttp
from typing import Any, Dict, List, Optional
from pathlib import Path
from io import BytesIO
from datetime import datetime
from collections import defaultdict
import sys
import re
import os
import logging
import csv
import asyncio
print(">>> bot.py ЗАПУСТИЛСЯ")


# ==================== ЗАГРУЗКА .ENV ====================

load_dotenv()

# ==================== ЛОГИРОВАНИЕ ====================


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
        "RESET": "\033[0m",
        "TIME": "\033[94m",
        "NAME": "\033[97m",
        "FUNC": "\033[93m",
        "BAR": "\033[90m",
    }

    EMOJIS = {
        "DEBUG": "🐛",
        "INFO": "ℹ️",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "🚨",
    }

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        c = self.COLORS
        emoji = self.EMOJIS.get(level, "🔹")
        time_str = c["TIME"] + self.formatTime(record) + c["RESET"]
        name_str = c["NAME"] + record.name + c["RESET"]
        func = getattr(record, "funcName", "")
        func_str = f" [{c['FUNC']}{func}{c['RESET']}]" if func else ""
        bar = c["BAR"] + "─" * 10 + c["RESET"]
        level_str = c.get(level, c["RESET"]) + level + c["RESET"]
        msg = super().format(record)
        return f"{bar} {emoji} {time_str} | {level_str} | {name_str}{func_str} | {msg} {bar}"


logger = logging.getLogger("BotLogger")
log_level_str = os.getenv("LOG_LEVEL", "DEBUG").upper()
log_level = getattr(logging, log_level_str, logging.DEBUG)
logger.setLevel(log_level)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(log_level)
ch.setFormatter(ColoredFormatter(
    fmt="%(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(ch)

logging.getLogger("aiogram").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

# ==================== THROTTLING ====================

user_last_message_time = defaultdict(float)
THROTTLE_SECONDS = 1.0  # минимум секунд между сообщениями пользователя


async def check_throttle(user_id: int) -> bool:
    """Простая защита от спама: не чаще одного сообщения в секунду."""
    loop_time = asyncio.get_event_loop().time()
    last_time = user_last_message_time[user_id]
    if loop_time - last_time < THROTTLE_SECONDS:
        return False
    user_last_message_time[user_id] = loop_time
    return True


# ==================== НАСТРОЙКИ И ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ====================

logger.info("Загрузка конфигурации из .env файла...")

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    logger.critical("В .env не найден BOT_TOKEN. Бот не может запуститься.")
    raise RuntimeError("В .env не найден BOT_TOKEN")

IMG_BASE = (os.getenv("IMG_BASE")
            or "https://alltpms.ru/img/").rstrip("/") + "/"
SITE_BASE = (os.getenv("SITE_BASE") or "https://alltpms.ru").rstrip("/")
PRODUCT_CARDS_LIMIT = 6

LEADS_CSV = Path(os.getenv("LEADS_CSV") or "leads.csv")
MANAGERS_FILE = Path(os.getenv("MANAGERS_CSV") or "managers.csv")

ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")
MANAGER_KEYS_RAW = os.getenv("MANAGER_KEYS")
ALLOWED_MANAGER_KEYS = set(MANAGER_KEYS_RAW.split(
    ",")) if MANAGER_KEYS_RAW else None

data_dir = os.getenv("DATA_DIR") or None

logger.info(f"IMG_BASE: {IMG_BASE}")
logger.info(f"SITE_BASE: {SITE_BASE}")
logger.info(f"LEADS_CSV: {LEADS_CSV}")
logger.info(f"MANAGERS_FILE: {MANAGERS_FILE}")
logger.info(f"DATA_DIR: {data_dir if data_dir else 'по умолчанию (./data)'}")
logger.info(f"ADMIN_CHAT_ID: {ADMIN_CHAT_ID}")
logger.info(f"ALLOWED_MANAGER_KEYS: {ALLOWED_MANAGER_KEYS}")

# ==================== КАТАЛОГ ====================

try:
    from data_loader import Catalog

    catalog = Catalog(data_dir)
    logger.info("Каталог загружен успешно.")
except Exception as e:
    logger.critical(f"Не удалось загрузить каталог: {e}")
    raise

# ==================== BOT & DISPATCHER ====================

bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher(storage=MemoryStorage())

# ==================== МЕНЕДЖЕРЫ, ПРИВЯЗКИ, ИСТОРИЯ ДИАЛОГА ====================

manager_registry: Dict[str, Dict[str, Any]] = {}  # key -> данные менеджера
user_to_manager: Dict[int, str] = {}  # user_id -> manager_key

# user_id -> список строк истории диалога
dialog_history: Dict[int, List[str]] = defaultdict(list)


def add_to_history(user_id: int, who: str, text: str) -> None:
    """
    who: "user" или "bot"
    text: строка без HTML
    """
    prefix = "Пользователь" if who == "user" else "Бот"
    safe_text = (text or "").strip()
    if not safe_text:
        return
    line = f"{prefix}: {safe_text}"
    dialog_history[user_id].append(line)
    # ограничиваем длину истории
    if len(dialog_history[user_id]) > 200:
        dialog_history[user_id] = dialog_history[user_id][-200:]


# ==================== FSM ====================


class LeadForm(StatesGroup):
    nav = State()
    ask_name = State()
    ask_phone = State()
    waiting_for_manual_brand = State()
    waiting_for_manual_year = State()
    waiting_for_manual_model = State()


# ==================== CSV МЕНЕДЖЕРОВ ====================


def init_managers_csv() -> None:
    """Инициализация файла менеджеров, если его нет."""
    if MANAGERS_FILE.exists():
        return
    MANAGERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MANAGERS_FILE.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Дата регистрации", "Ключ", "ID менеджера",
                   "Имя", "Username", "Полное имя"])
    logger.info(f"Создан файл менеджеров: {MANAGERS_FILE}")


def load_managers() -> None:
    """Загрузка менеджеров из CSV в память."""
    init_managers_csv()
    manager_registry.clear()
    try:
        with MANAGERS_FILE.open("r", newline="", encoding="utf-8") as f:
            r = csv.reader(f, delimiter=";")
            headers = next(r, None)
            if not headers:
                return
            for row in r:
                if len(row) < 6:
                    continue
                registered, key, chat_id_str, name, username, full_name = row
                try:
                    chat_id = int(chat_id_str)
                except ValueError:
                    continue
                manager_registry[key] = {
                    "chat_id": chat_id,
                    "name": name,
                    "username": username,
                    "full_name": full_name,
                    "registered": registered,
                }
        logger.info(f"Загружено менеджеров: {len(manager_registry)}")
    except Exception as e:
        logger.error(f"Ошибка чтения файла менеджеров: {e}")


def save_manager(key: str, user: types.User) -> None:
    """Сохранение менеджера в CSV и память."""
    init_managers_csv()
    is_new = not MANAGERS_FILE.exists()
    MANAGERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with MANAGERS_FILE.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            if is_new:
                w.writerow(["Дата регистрации", "Ключ",
                           "ID менеджера", "Имя", "Username", "Полное имя"])
            w.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    key,
                    user.id,
                    user.first_name or "",
                    user.username or "",
                    f"{user.first_name or ''} {user.last_name or ''}".strip(),
                ]
            )
    except Exception as e:
        logger.error(f"Ошибка записи менеджера в CSV: {e}")
        return

    manager_registry[key] = {
        "chat_id": user.id,
        "name": user.first_name or "",
        "username": user.username or "",
        "full_name": f"{user.first_name or ''} {user.last_name or ''}".strip(),
        "registered": datetime.now().isoformat(timespec="seconds"),
    }
    logger.info(f"Менеджер с ключом {key} сохранён.")


# ==================== CSV ЛИДОВ ====================


def init_leads_csv() -> None:
    """Инициализация файла лидов, если его нет."""
    if LEADS_CSV.exists():
        return
    LEADS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with LEADS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(
            [
                "Дата/Время",
                "Марка",
                "Год",
                "Модель",
                "Датчик",
                "Артикул",
                "Цена",
                "Производитель",
                "Имя клиента",
                "Телефон",
                "Username клиента",
                "User ID клиента",
                "Менеджер (ключ)",
                "Менеджер (имя)",
                "Менеджер (ID)",
            ]
        )
    logger.info(f"Создан файл лидов: {LEADS_CSV}")


def save_lead(data: Dict[str, Any]) -> None:
    """Сохранение лида в CSV."""
    init_leads_csv()
    is_new = not LEADS_CSV.exists()
    LEADS_CSV.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LEADS_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            if is_new:
                w.writerow(
                    [
                        "Дата/Время",
                        "Марка",
                        "Год",
                        "Модель",
                        "Датчик",
                        "Артикул",
                        "Цена",
                        "Производитель",
                        "Имя клиента",
                        "Телефон",
                        "Username клиента",
                        "User ID клиента",
                        "Менеджер (ключ)",
                        "Менеджер (имя)",
                        "Менеджер (ID)",
                    ]
                )
            w.writerow(
                [
                    data.get("ts", datetime.now().isoformat(
                        timespec="seconds")),
                    data.get("brand", ""),
                    data.get("year", ""),
                    data.get("model", ""),
                    data.get("product_title", ""),
                    data.get("product_id", ""),
                    data.get("product_price", ""),
                    data.get("product_made", ""),
                    data.get("name", ""),
                    data.get("phone", ""),
                    data.get("username", ""),
                    data.get("user_id", ""),
                    data.get("manager_key", ""),
                    data.get("manager_name", ""),
                    data.get("manager_id", ""),
                ]
            )
    except Exception as e:
        logger.error(f"Ошибка записи лида в CSV: {e}")
        return
    logger.info(
        f"Лид записан: {data.get('name', 'N/A')} / {data.get('phone', 'N/A')}")


# ==================== УТИЛЬНЫЕ ФУНКЦИИ ====================


def validate_phone(s: str) -> bool:
    digits = re.sub(r"\D+", "", s or "")
    return 8 <= len(digits) <= 15


async def fetch_and_compress_image(
    url: str, quality: int = 90, timeout: int = 15, retries: int = 3
) -> Optional[bytes]:
    """Загрузка изображения по URL и сжатие в JPEG."""
    last_error: Optional[Exception] = None
    for _ in range(retries):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        last_error = Exception(f"HTTP {resp.status}")
                        continue
                    content = await resp.read()
            if len(content) > 50 * 1024 * 1024:
                return None
            im = Image.open(BytesIO(content))
            rgb_im = im.convert("RGB")
            buf = BytesIO()
            rgb_im.save(buf, format="JPEG", quality=quality, optimize=True)
            return buf.getvalue()
        except Exception as e:
            last_error = e
            await asyncio.sleep(1)
    if last_error:
        logger.error(f"Ошибка загрузки изображения: {last_error}")
    return None


def product_best_image_filename(prod: dict) -> Optional[str]:
    imgs = prod.get("Images")
    if isinstance(imgs, list) and imgs:
        return os.path.basename(str(imgs[0]).strip())
    raw = prod.get("images")
    if raw:
        parts = [os.path.basename(p.strip())
                 for p in str(raw).split("||") if p.strip()]
        if parts:
            return parts[0]
    mi = prod.get("MarketImage") or prod.get("market_image")
    if mi:
        return os.path.basename(str(mi).strip())
    return None


def product_caption(prod: dict) -> str:
    title = (
        prod.get("MarketTitle")
        or prod.get("market_title")
        or prod.get("Title")
        or prod.get("pagetitle")
        or prod.get("name")
        or prod.get("SKU")
    )
    made_in = prod.get("MadeIn") or prod.get("made_in") or ""
    lines = [f"<b>{title}</b>"]
    if made_in:
        lines.append(f"Производство: <i>{made_in}</i>")
    lines.append("Доставка от 1 шт. без предоплаты, гарантия 2 года")
    return "\n".join(lines)


async def send_product_card(chat_id: int, prod: dict) -> None:
    """Отправка карточки товара с кнопкой 'Выбрать'."""
    row_id = prod.get("RowID")
    cap = product_caption(prod)
    fname = product_best_image_filename(prod)
    kb = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(
            text="Выбрать", callback_data=f"pick:{row_id}")]]
    )
    if fname:
        url = IMG_BASE + fname
        data = await fetch_and_compress_image(url, quality=90)
        if data:
            photo = BufferedInputFile(data, filename=fname)
            await bot.send_photo(chat_id, photo=photo, caption=cap, reply_markup=kb)
            return
    await bot.send_message(chat_id, cap, reply_markup=kb)


def paginate_list(items: List[Any], page: int, page_size: int) -> Dict[str, Any]:
    start = page * page_size
    end = start + page_size
    page_items = items[start:end]
    total_pages = (len(items) + page_size - 1) // page_size if items else 1
    return {
        "items": page_items,
        "current_page": page,
        "total_pages": total_pages,
        "has_prev": page > 0,
        "has_next": end < len(items),
    }


def brands_keyboard(
    brands: List[str], page: int, total_pages: int, base_index: int
) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for offset, name in enumerate(brands):
        idx = base_index + offset
        kb.button(text=name, callback_data=f"br:{idx}")
    kb.adjust(2)

    nav = InlineKeyboardBuilder()
    if page > 0:
        nav.button(text="⬅️ Назад", callback_data=f"nav:brand:prev:{page-1}")
    if page < total_pages - 1:
        nav.button(text="Далее ➡️", callback_data=f"nav:brand:next:{page+1}")
    if nav.buttons:
        nav.adjust(2)
        kb.attach(nav)

    kb.button(text="🔍 Ввести марку вручную", callback_data="input:brand")
    kb.adjust(2, 1)
    return kb.as_markup()


def years_keyboard(years: List[int], page: int, total_pages: int, base_index: int) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for offset, y in enumerate(years):
        idx = base_index + offset
        kb.button(text=str(y), callback_data=f"yr:{idx}")
    kb.adjust(3)

    nav = InlineKeyboardBuilder()
    if page > 0:
        nav.button(text="⬅️ Назад", callback_data=f"nav:year:prev:{page-1}")
    if page < total_pages - 1:
        nav.button(text="Далее ➡️", callback_data=f"nav:year:next:{page+1}")
    if nav.buttons:
        nav.adjust(2)
        kb.attach(nav)

    kb.button(text="🔍 Ввести год вручную", callback_data="input:year")
    kb.adjust(3, 1)
    return kb.as_markup()


def models_keyboard(
    models: List[str], page: int, total_pages: int, base_index: int
) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for offset, m in enumerate(models):
        idx = base_index + offset
        kb.button(text=m, callback_data=f"mdl:{idx}")
    kb.adjust(1)

    nav = InlineKeyboardBuilder()
    if page > 0:
        nav.button(text="⬅️ Назад", callback_data=f"nav:model:prev:{page-1}")
    if page < total_pages - 1:
        nav.button(text="Далее ➡️", callback_data=f"nav:model:next:{page+1}")
    if nav.buttons:
        nav.adjust(2)
        kb.attach(nav)

    kb.button(text="🔍 Ввести модель вручную", callback_data="input:model")
    kb.adjust(1, 1)
    return kb.as_markup()


def connect_manager_keyboard() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="💬 Связаться с менеджером", callback_data="connect_manager")
    kb.adjust(1)
    return kb.as_markup()


def restart_keyboard() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="🔁 Начать подбор заново", callback_data="restart_flow")
    kb.adjust(1)
    return kb.as_markup()


# ==================== ХЕЛПЕРЫ ДЛЯ FSM-ПЕРЕХОДОВ ====================


async def proceed_to_years(message: types.Message, state: FSMContext, brand: str) -> None:
    years = catalog.list_years(brand)
    if not years:
        await message.answer(f"Для марки <b>{brand}</b> нет доступных годов.")
        return
    pag = paginate_list(years, 0, 10)
    kb = years_keyboard(pag["items"], pag["current_page"],
                        pag["total_pages"], base_index=0)
    await state.update_data(
        brand=brand,
        years=years,
        year_page=0,
        total_year_pages=pag["total_pages"],
    )
    text = f"Марка: <b>{brand}</b>\nВыберите год:"
    await message.answer(text, reply_markup=kb)
    add_to_history(message.from_user.id, "bot", "Предложен выбор года.")


async def proceed_to_models(message: types.Message, state: FSMContext, brand: str, year: int) -> None:
    models = catalog.list_models(brand, year)
    if not models:
        await message.answer(f"Для {brand} {year} нет доступных моделей.")
        return
    pag = paginate_list(models, 0, 10)
    kb = models_keyboard(pag["items"], pag["current_page"],
                         pag["total_pages"], base_index=0)
    await state.update_data(
        brand=brand,
        year=year,
        models=models,
        model_page=0,
        total_model_pages=pag["total_pages"],
    )
    text = f"Марка: <b>{brand}</b>\nГод: <b>{year}</b>\nВыберите модель:"
    await message.answer(text, reply_markup=kb)
    add_to_history(message.from_user.id, "bot", "Предложен выбор модели.")


async def show_products(message: types.Message, state: FSMContext, brand: str, year: int, model: str) -> None:
    """Показ списка датчиков + кнопка 'Связаться с менеджером'."""
    prods = catalog.products_for(brand, int(year), model)
    if not prods:
        await message.answer("Для выбранной модели не найдено датчиков.")
        add_to_history(message.from_user.id, "bot",
                       "Не найдено датчиков для выбранной модели.")
        return

    await state.update_data(
        brand=brand,
        year=year,
        model=model,
        product_row_ids=[p.get("RowID") for p in prods],
    )

    await message.answer("🔍 Подбираю подходящие датчики...")
    add_to_history(message.from_user.id, "bot", "Начат показ списка датчиков.")

    for p in prods[:PRODUCT_CARDS_LIMIT]:
        await send_product_card(message.chat.id, p)

    await message.answer(
        "Если хотите, могу связать вас с менеджером для уточнения деталей.",
        reply_markup=connect_manager_keyboard(),
    )
    add_to_history(
        message.from_user.id,
        "bot",
        f"Показано {min(len(prods), PRODUCT_CARDS_LIMIT)} датчиков и предложена связь с менеджером.",
    )


async def start_search_flow(message: types.Message, state: FSMContext):
    """Запустить сценарий подбора с нуля (как /start для клиента)."""
    await state.clear()
    dialog_history[message.from_user.id].clear()

    await state.set_state(LeadForm.nav)

    brands = catalog.list_brands()
    if not brands:
        await message.answer("Каталог пуст. Попробуйте позже.")
        add_to_history(message.from_user.id, "bot",
                       "Каталог пуст при старте подбора.")
        return

    pag = paginate_list(brands, 0, 10)
    kb = brands_keyboard(pag["items"], pag["current_page"],
                         pag["total_pages"], base_index=0)
    await state.update_data(
        brands=brands,
        brand_page=0,
        total_brand_pages=pag["total_pages"],
    )
    text = (
        "Привет! 👋\n"
        "Давайте подберём датчик давления.\n\n"
        "Сначала выберите марку автомобиля:"
    )
    await message.answer(text, reply_markup=kb)
    add_to_history(message.from_user.id, "bot", "Запущен подбор с нуля.")


# ==================== ОБРАБОТКА КЛЮЧА МЕНЕДЖЕРА НА ЛЮБОМ ЭТАПЕ ====================


def is_manager_key_message(text: Optional[str]) -> bool:
    """Проверка, похоже ли сообщение на ключ менеджера."""
    if not text:
        return False
    t = text.strip().lower()
    return t.startswith("key:") or t.startswith("key_")


async def process_manager_key_message(m: types.Message) -> None:
    """Обрабатывает сообщение с ключом менеджера (регистрация)."""

    if not await check_throttle(m.from_user.id):
        return

    text = (m.text or "").strip()
    msg_lower = text.lower().strip()
    key: Optional[str] = None

    if msg_lower.startswith("key:"):
        key = text.split(":", 1)[1].strip()
    elif msg_lower.startswith("key_"):
        key = text.strip()

    if not key:
        return

    # проверка по белому списку
    if ALLOWED_MANAGER_KEYS and key not in ALLOWED_MANAGER_KEYS:
        await m.answer("❌ Этот ключ не разрешён. Обратитесь к администратору.")
        return

    # ключ уже занят?
    if key in manager_registry:
        existing_manager_id = manager_registry[key]["chat_id"]
        if existing_manager_id == m.from_user.id:
            await m.answer("ℹ️ Вы уже зарегистрированы с этим ключом.")
        else:
            await m.answer("❌ Этот ключ уже используется другим менеджером.")
        return

    # регистрируем менеджера
    save_manager(key, m.from_user)

    ref_link = f"https://t.me/ALLTPMS_bot?start={key}"
    welcome_msg = (
        f"🎉 <b>Добро пожаловать, {m.from_user.first_name}!</b>\n"
        f"✅ Вы успешно зарегистрированы в системе как менеджер.\n\n"
        f"🔑 <b>Ваш ключ:</b> <code>{key}</code>\n"
        f"🆔 <b>Ваш ID:</b> <code>{m.from_user.id}</code>\n\n"
        f"🔗 <b>Ваша реферальная ссылка:</b>\n"
        f"<code>{ref_link}</code>\n\n"
        f"📋 Клиенты, перешедшие по этой ссылке, будут привязаны к вам."
    )
    await m.answer(welcome_msg)
    add_to_history(m.from_user.id, "bot",
                   f"Пользователь зарегистрирован как менеджер с ключом {key}.")


@dp.message(lambda m: is_manager_key_message(m.text))
async def manager_key_handler(m: types.Message, state: FSMContext):
    """
    Обработчик ключа менеджера.
    Срабатывает на ЛЮБОМ этапе, независимо от FSM.
    Если сообщение - ключ, другие message-хендлеры уже не выполняются.
    """
    await process_manager_key_message(m)


# ==================== HANDLERS: /start и /reload ====================


@dp.message(CommandStart())
async def cmd_start(m: types.Message, state: FSMContext):
    if not await check_throttle(m.from_user.id):
        return

    add_to_history(m.from_user.id, "user", m.text or "/start")

    # 1) Проверка: не менеджер ли это
    is_manager = False
    manager_key = None
    manager_data = None
    for k, data in manager_registry.items():
        if data["chat_id"] == m.from_user.id:
            is_manager = True
            manager_key = k
            manager_data = data
            break

    if is_manager and manager_key and manager_data:
        ref_link = f"https://t.me/ALLTPMS_bot?start={manager_key}"
        msg = (
            f"💼 <b>Режим менеджера</b>\n\n"
            f"🔑 Ключ: <code>{manager_key}</code>\n"
            f"🆔 Ваш ID: <code>{manager_data['chat_id']}</code>\n"
            f"👤 Имя: {manager_data['name']}\n\n"
            f"🔗 Реферальная ссылка:\n<code>{ref_link}</code>\n\n"
            f"Клиенты по этой ссылке будут закреплены за вами."
        )
        await m.answer(msg)
        add_to_history(m.from_user.id, "bot", "Показана информация менеджера.")
        return

    # 2) Клиент: разбираем параметр /start <key>
    text_parts = (m.text or "").split(maxsplit=1)
    if len(text_parts) > 1:
        param = text_parts[1].strip()
        if param in manager_registry:
            user_to_manager[m.from_user.id] = param
            manager_name = manager_registry[param].get("name") or "менеджер"
            await m.answer(f"✅ Вы подключены к менеджеру: <b>{manager_name}</b>")
            add_to_history(
                m.from_user.id,
                "bot",
                f"Клиент привязан к менеджеру по ключу {param}.",
            )

    # 3) Запускаем стандартный подбор
    await start_search_flow(m, state)


@dp.message(Command("reload"))
async def cmd_reload(m: types.Message, state: FSMContext):
    if not await check_throttle(m.from_user.id):
        return
    add_to_history(m.from_user.id, "user", m.text or "/reload")

    catalog.reload()
    await start_search_flow(m, state)
    add_to_history(m.from_user.id, "bot",
                   "Каталог перезагружен и подбор запущен заново.")


# ==================== HANDLERS: ВЫБОР МАРКИ ====================


@dp.callback_query(F.data.startswith("nav:brand:"))
async def navigate_brands(cq: types.CallbackQuery, state: FSMContext):
    try:
        _, _, direction, page_str = cq.data.split(":")
        new_page = int(page_str)
    except Exception:
        await cq.answer("Некорректный выбор")
        return

    data = await state.get_data()
    brands: List[str] = data.get("brands") or []
    total_pages = data.get("total_brand_pages", 1)

    if not brands:
        await cq.answer("Марки недоступны.")
        return

    if direction == "prev" and new_page >= 0:
        pass
    elif direction == "next" and new_page < total_pages:
        pass
    else:
        await cq.answer("❌")
        return

    pag = paginate_list(brands, new_page, 10)
    kb = brands_keyboard(pag["items"], pag["current_page"],
                         pag["total_pages"], base_index=new_page * 10)
    await state.update_data(brand_page=new_page)

    await cq.message.edit_text(
        "Выберите марку автомобиля:",
        reply_markup=kb,
    )
    await cq.answer()


@dp.callback_query(F.data.startswith("br:"))
async def choose_brand(cq: types.CallbackQuery, state: FSMContext):
    user_id = cq.from_user.id
    try:
        idx = int(cq.data.split(":", 1)[1])
    except Exception:
        await cq.answer("Некорректный выбор")
        return

    data = await state.get_data()
    brands: List[str] = data.get("brands") or []
    if idx < 0 or idx >= len(brands):
        await cq.answer("Некорректный выбор")
        return
    brand = brands[idx]
    await state.update_data(brand=brand)
    add_to_history(user_id, "user", f"Выбрал марку: {brand}")
    await cq.answer()
    await proceed_to_years(cq.message, state, brand)


@dp.callback_query(F.data == "input:brand")
async def prompt_manual_brand(cq: types.CallbackQuery, state: FSMContext):
    await cq.message.answer("Введите название марки автомобиля:")
    await state.set_state(LeadForm.waiting_for_manual_brand)
    await cq.answer()


@dp.message(LeadForm.waiting_for_manual_brand)
async def process_manual_brand(m: types.Message, state: FSMContext):
    if not await check_throttle(m.from_user.id):
        return
    user_input_raw = (m.text or "").strip()
    user_input = user_input_raw.lower()
    add_to_history(m.from_user.id, "user",
                   f"Ввёл марку вручную: {user_input_raw}")

    found = catalog.search_brands(user_input)
    if not found:
        all_brands = catalog.list_brands()
        found = [b for b in all_brands if user_input in b.lower()]

    if not found:
        await m.answer("❌ Не найдено марок, соответствующих запросу. Попробуйте ещё раз.")
        return

    if len(found) == 1:
        brand = found[0]
        await state.update_data(brand=brand)
        await state.set_state(LeadForm.nav)
        add_to_history(m.from_user.id, "bot",
                       f"Марка {brand} выбрана по ручному вводу.")
        await proceed_to_years(m, state, brand)
        return

    pag = paginate_list(found, 0, 10)
    kb = brands_keyboard(pag["items"], pag["current_page"],
                         pag["total_pages"], base_index=0)
    await state.update_data(
        brands=found,
        brand_page=0,
        total_brand_pages=pag["total_pages"],
    )
    await state.set_state(LeadForm.nav)
    await m.answer("Найдено несколько марок. Выберите одну:", reply_markup=kb)
    add_to_history(m.from_user.id, "bot",
                   "Найдено несколько марок по ручному вводу, предложен выбор.")


# ==================== HANDLERS: ВЫБОР ГОДА ====================


@dp.callback_query(F.data.startswith("nav:year:"))
async def navigate_years(cq: types.CallbackQuery, state: FSMContext):
    try:
        _, _, direction, page_str = cq.data.split(":")
        new_page = int(page_str)
    except Exception:
        await cq.answer("Некорректный выбор")
        return

    data = await state.get_data()
    years: List[int] = data.get("years") or []
    total_pages = data.get("total_year_pages", 1)

    if not years:
        await cq.answer("Годы недоступны.")
        return

    if direction == "prev" and new_page >= 0:
        pass
    elif direction == "next" and new_page < total_pages:
        pass
    else:
        await cq.answer("❌")
        return

    pag = paginate_list(years, new_page, 10)
    kb = years_keyboard(pag["items"], pag["current_page"],
                        pag["total_pages"], base_index=new_page * 10)
    await state.update_data(year_page=new_page)

    data = await state.get_data()
    brand = data.get("brand", "")
    text = f"Марка: <b>{brand}</b>\nВыберите год:"
    await cq.message.edit_text(text, reply_markup=kb)
    await cq.answer()


@dp.callback_query(F.data.startswith("yr:"))
async def choose_year(cq: types.CallbackQuery, state: FSMContext):
    user_id = cq.from_user.id
    try:
        idx = int(cq.data.split(":", 1)[1])
    except Exception:
        await cq.answer("Некорректный выбор")
        return

    data = await state.get_data()
    years: List[int] = data.get("years") or []
    brand: str = data.get("brand", "")
    if idx < 0 or idx >= len(years):
        await cq.answer("Некорректный выбор")
        return
    year = int(years[idx])
    await state.update_data(year=year)
    add_to_history(user_id, "user", f"Выбрал год: {year}")
    await cq.answer()
    await proceed_to_models(cq.message, state, brand, year)


@dp.callback_query(F.data == "input:year")
async def prompt_manual_year(cq: types.CallbackQuery, state: FSMContext):
    await cq.message.answer("Введите год выпуска автомобиля (например, 2018):")
    await state.set_state(LeadForm.waiting_for_manual_year)
    await cq.answer()


@dp.message(LeadForm.waiting_for_manual_year)
async def process_manual_year(m: types.Message, state: FSMContext):
    if not await check_throttle(m.from_user.id):
        return
    raw = (m.text or "").strip()
    add_to_history(m.from_user.id, "user", f"Ввёл год вручную: {raw}")
    try:
        year = int(raw)
    except ValueError:
        await m.answer("❌ Пожалуйста, введите год цифрами, например 2018.")
        return

    data = await state.get_data()
    brand: str = data.get("brand", "")
    years = catalog.list_years(brand)
    if year not in years:
        await m.answer("Для этой марки нет такого года в каталоге. Попробуйте другой.")
        return

    await state.set_state(LeadForm.nav)
    await state.update_data(year=year, years=years)
    add_to_history(m.from_user.id, "bot",
                   f"Год {year} принят по ручному вводу.")
    await proceed_to_models(m, state, brand, year)


# ==================== HANDLERS: ВЫБОР МОДЕЛИ ====================


@dp.callback_query(F.data.startswith("nav:model:"))
async def navigate_models(cq: types.CallbackQuery, state: FSMContext):
    try:
        _, _, direction, page_str = cq.data.split(":")
        new_page = int(page_str)
    except Exception:
        await cq.answer("Некорректный выбор")
        return

    data = await state.get_data()
    models: List[str] = data.get("models") or []
    total_pages = data.get("total_model_pages", 1)

    if not models:
        await cq.answer("Модели недоступны.")
        return

    if direction == "prev" and new_page >= 0:
        pass
    elif direction == "next" and new_page < total_pages:
        pass
    else:
        await cq.answer("❌")
        return

    pag = paginate_list(models, new_page, 10)
    kb = models_keyboard(pag["items"], pag["current_page"],
                         pag["total_pages"], base_index=new_page * 10)
    await state.update_data(model_page=new_page)

    data = await state.get_data()
    brand = data.get("brand", "")
    year = data.get("year", "")
    text = f"Марка: <b>{brand}</b>\nГод: <b>{year}</b>\nВыберите модель:"
    await cq.message.edit_text(text, reply_markup=kb)
    await cq.answer()


@dp.callback_query(F.data.startswith("mdl:"))
async def choose_model(cq: types.CallbackQuery, state: FSMContext):
    user_id = cq.from_user.id
    try:
        idx = int(cq.data.split(":", 1)[1])
    except Exception:
        await cq.answer("Некорректный выбор")
        return

    data = await state.get_data()
    models: List[str] = data.get("models") or []
    brand: str = data.get("brand", "")
    year: int = data.get("year", 0)

    if idx < 0 or idx >= len(models):
        await cq.answer("Некорректный выбор")
        return
    model = models[idx]
    await state.update_data(model=model)
    add_to_history(user_id, "user", f"Выбрал модель: {model}")
    await cq.answer()
    await show_products(cq.message, state, brand, year, model)


@dp.callback_query(F.data == "input:model")
async def prompt_manual_model(cq: types.CallbackQuery, state: FSMContext):
    await cq.message.answer("Введите название модели:")
    await state.set_state(LeadForm.waiting_for_manual_model)
    await cq.answer()


@dp.message(LeadForm.waiting_for_manual_model)
async def process_manual_model(m: types.Message, state: FSMContext):
    if not await check_throttle(m.from_user.id):
        return
    raw = (m.text or "").strip()
    add_to_history(m.from_user.id, "user", f"Ввёл модель вручную: {raw}")

    data = await state.get_data()
    brand: str = data.get("brand", "")
    year = data.get("year")
    if not year:
        await m.answer("Сначала выберите год, затем введите модель.")
        return

    models = catalog.list_models(brand, int(year))
    found = [mm for mm in models if raw.lower() in mm.lower()]

    if not found:
        await m.answer("❌ Не найдено моделей по вашему запросу. Попробуйте ещё раз.")
        return

    if len(found) == 1:
        model = found[0]
        await state.set_state(LeadForm.nav)
        await state.update_data(model=model, models=models)
        add_to_history(m.from_user.id, "bot",
                       f"Модель {model} выбрана по ручному вводу.")
        await show_products(m, state, brand, year, model)
        return

    pag = paginate_list(found, 0, 10)
    kb = models_keyboard(pag["items"], pag["current_page"],
                         pag["total_pages"], base_index=0)
    await state.update_data(
        models=found,
        model_page=0,
        total_model_pages=pag["total_pages"],
    )
    await state.set_state(LeadForm.nav)
    await m.answer("Найдено несколько моделей. Выберите одну:", reply_markup=kb)
    add_to_history(m.from_user.id, "bot",
                   "Найдено несколько моделей по ручному вводу, предложен выбор.")


# ==================== HANDLERS: ВЫБОР КОНКРЕТНОГО ДАТЧИКА ====================


@dp.callback_query(F.data.startswith("pick:"))
async def pick_product(cq: types.CallbackQuery, state: FSMContext):
    user_id = cq.from_user.id
    try:
        row_id = int(cq.data.split(":", 1)[1])
    except Exception:
        await cq.answer("Некорректный выбор")
        return

    data = await state.get_data()
    brand = data.get("brand")
    year = data.get("year")
    model = data.get("model")

    # Если состояние потеряно (например, после /start) —
    # не продолжаем сценарий с пустыми полями, а предлагаем перезапуск.
    if not brand or not year or not model:
        await cq.message.answer(
            "Сессия подбора была сброшена (например, после команды /start).\n"
            "Давайте начнём подбор заново 👇",
            reply_markup=restart_keyboard(),
        )
        add_to_history(
            user_id, "bot", "Попытка выбрать датчик при пустом состоянии — предложен перезапуск.")
        await cq.answer()
        return

    await state.update_data(chosen_row_id=row_id)
    add_to_history(user_id, "user", f"Выбрал датчик (RowID={row_id}).")

    text = (
        f"Вы выбрали датчик для:\n"
        f"Марка: <b>{brand}</b>\n"
        f"Год: <b>{year}</b>\n"
        f"Модель: <b>{model}</b>\n\n"
        f"Теперь введите, пожалуйста, ваше имя:"
    )
    await cq.message.answer(text)
    add_to_history(user_id, "bot", "Попросил ввести имя после выбора датчика.")
    await state.set_state(LeadForm.ask_name)
    await cq.answer()


# ==================== HANDLERS: СБОР КОНТАКТОВ ====================


@dp.message(LeadForm.ask_name)
async def ask_phone(m: types.Message, state: FSMContext):
    if not await check_throttle(m.from_user.id):
        return
    name = (m.text or "").strip()
    if not name:
        await m.answer("Пожалуйста, введите имя текстом.")
        return

    await state.update_data(client_name=name)
    add_to_history(m.from_user.id, "user", f"Оставил имя: {name}")
    await m.answer("Спасибо! Теперь отправьте номер телефона (в любом формате).")
    add_to_history(m.from_user.id, "bot", "Попросил номер телефона.")
    await state.set_state(LeadForm.ask_phone)


@dp.message(LeadForm.ask_phone)
async def finish_lead(m: types.Message, state: FSMContext):
    if not await check_throttle(m.from_user.id):
        return
    phone_raw = (m.text or "").strip()
    add_to_history(m.from_user.id, "user", f"Оставил телефон: {phone_raw}")

    if not validate_phone(phone_raw):
        await m.answer("❌ Похоже, номер некорректный. Отправьте его ещё раз, пожалуйста.")
        return

    data = await state.get_data()
    brand = data.get("brand") or ""
    year = data.get("year") or ""
    model = data.get("model") or ""
    row_id = data.get("chosen_row_id")
    client_name = data.get("client_name") or ""
    user = m.from_user

    prod = catalog.get_by_rowid(row_id) if row_id is not None else None
    product_title = ""
    product_id = ""
    product_price = ""
    product_made = ""

    if prod:
        product_title = prod.get("MarketTitle") or prod.get(
            "Title") or prod.get("SKU") or ""
        product_id = prod.get("SKU") or ""
        price = prod.get("Price")
        if price is not None:
            product_price = str(price)
        product_made = prod.get("MadeIn") or ""

    manager_key = user_to_manager.get(user.id)
    manager_name = ""
    manager_id = ""

    target_manager_chat_id: Optional[int] = None
    # 1) Привязанный менеджер
    if manager_key and manager_key in manager_registry:
        mgr = manager_registry[manager_key]
        target_manager_chat_id = mgr["chat_id"]
        manager_name = mgr.get("name", "")
        manager_id = mgr["chat_id"]
    # 2) ADMIN_CHAT_ID
    elif ADMIN_CHAT_ID:
        try:
            target_manager_chat_id = int(ADMIN_CHAT_ID)
        except ValueError:
            target_manager_chat_id = None
    # 3) Любой менеджер из реестра
    if not target_manager_chat_id and manager_registry:
        any_key, any_mgr = next(iter(manager_registry.items()))
        target_manager_chat_id = any_mgr["chat_id"]
        manager_key = any_key
        manager_name = any_mgr.get("name", "")
        manager_id = any_mgr["chat_id"]

    lead_data = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "brand": brand,
        "year": year,
        "model": model,
        "product_title": product_title,
        "product_id": product_id,
        "product_price": product_price,
        "product_made": product_made,
        "name": client_name,
        "phone": phone_raw,
        "username": user.username or "",
        "user_id": user.id,
        "manager_key": manager_key or "",
        "manager_name": manager_name,
        "manager_id": manager_id,
    }

    save_lead(lead_data)

    if target_manager_chat_id:
        username_part = f"@{user.username}" if user.username else "нет username"
        profile_link = (
            f"https://t.me/{user.username}" if user.username else f"tg://user?id={user.id}"
        )
        text = (
            "📥 <b>Новый лид из бота</b>\n\n"
            f"Марка: <b>{brand}</b>\n"
            f"Год: <b>{year}</b>\n"
            f"Модель: <b>{model}</b>\n"
            f"Датчик: <b>{product_title or 'не выбран'}</b>\n"
            f"Артикул: <code>{product_id or '-'}</code>\n"
            f"Цена: {product_price or '-'}\n"
            f"Производитель: {product_made or '-'}\n\n"
            f"Клиент: <b>{client_name}</b>\n"
            f"Телефон: <code>{phone_raw}</code>\n"
            f"Username: {username_part}\n"
            f"Профиль: <a href=\"{profile_link}\">открыть в Telegram</a>\n\n"
            f"User ID: <code>{user.id}</code>"
        )
        await bot.send_message(target_manager_chat_id, text)

    await m.answer(
        "Спасибо! Ваша заявка принята ✅\n"
        "Менеджер свяжется с вами в ближайшее время."
    )
    add_to_history(m.from_user.id, "bot",
                   "Лид оформлен, отправил подтверждение клиенту.")

    await state.clear()


# ==================== HANDLERS: СВЯЗАТЬ С МЕНЕДЖЕРОМ ====================


@dp.callback_query(F.data == "connect_manager")
async def connect_manager(cq: types.CallbackQuery, state: FSMContext):
    user = cq.from_user
    user_id = user.id

    data = await state.get_data()
    brand = data.get("brand") or "-"
    year = data.get("year") or "-"
    model = data.get("model") or "-"
    row_id = data.get("chosen_row_id")

    prod = catalog.get_by_rowid(row_id) if row_id is not None else None
    product_title = ""
    product_id = ""
    if prod:
        product_title = prod.get("MarketTitle") or prod.get(
            "Title") or prod.get("SKU") or ""
        product_id = prod.get("SKU") or ""

    history_lines = dialog_history.get(user_id, [])
    history_text = "\n".join(
        history_lines) if history_lines else "История пуста или недоступна."

    manager_key = user_to_manager.get(user_id)
    target_manager_chat_id: Optional[int] = None

    # 1) Сначала — привязанный к клиенту менеджер
    if manager_key and manager_key in manager_registry:
        target_manager_chat_id = manager_registry[manager_key]["chat_id"]
    # 2) Потом — ADMIN_CHAT_ID (если указан)
    elif ADMIN_CHAT_ID:
        try:
            target_manager_chat_id = int(ADMIN_CHAT_ID)
        except ValueError:
            target_manager_chat_id = None
    # 3) Если всё равно никого — берём ЛЮБОГО доступного менеджера из реестра
    if not target_manager_chat_id and manager_registry:
        any_key, any_mgr = next(iter(manager_registry.items()))
        target_manager_chat_id = any_mgr["chat_id"]

    # Если и тут пусто — реально нет никого
    if not target_manager_chat_id:
        await cq.message.answer(
            "К сожалению, сейчас нет доступного менеджера. Попробуйте позже.",
            reply_markup=restart_keyboard(),
        )
        await cq.answer()
        return

    username_part = f"@{user.username}" if user.username else "нет username"
    profile_link = (
        f"https://t.me/{user.username}" if user.username else f"tg://user?id={user.id}"
    )

    text = (
        "📨 <b>Клиент запросил связь с менеджером</b>\n\n"
        f"Марка: <b>{brand}</b>\n"
        f"Год: <b>{year}</b>\n"
        f"Модель: <b>{model}</b>\n"
        f"Выбранный датчик: <b>{product_title or 'нет / не выбран'}</b>\n"
        f"Артикул: <code>{product_id or '-'}</code>\n\n"
        f"Клиент: <b>{user.first_name or ''} {user.last_name or ''}</b>\n"
        f"Username: {username_part}\n"
        f"Профиль: <a href=\"{profile_link}\">открыть в Telegram</a>\n"
        f"User ID: <code>{user.id}</code>\n\n"
        f"🧾 <b>История диалога:</b>\n"
        f"{history_text}"
    )

    await bot.send_message(target_manager_chat_id, text)
    await cq.message.answer(
        "✅ Я передал ваш диалог и контактные данные менеджеру.\n"
        "Он свяжется с вами в ближайшее время."
    )
    add_to_history(
        user_id, "bot", "Диалог и данные клиента отправлены менеджеру по запросу связи.")
    await cq.answer()


# ==================== HANDLERS: РЕСТАРТ ПОТОКА ====================


@dp.callback_query(F.data == "restart_flow")
async def restart_flow_handler(cq: types.CallbackQuery, state: FSMContext):
    """Хэндлер кнопки 'Начать подбор заново'."""
    dialog_history[cq.from_user.id].clear()
    await cq.answer()
    await start_search_flow(cq.message, state)


# ==================== MAIN ====================


async def main() -> None:
    load_managers()
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен.")
