#!/usr/bin/env python3
"""
inci_aku_scraper.py
===================

İnci Akü bayilerinin Google Maps yorumlarını yüksek hızda kazır, içinde
"inci akü" geçen cümleleri çıkarır ve artımlı olarak JSON'a yazar.

Mimari (4 aşama):
    1. DealerAPIFetcher        - inciaku.com JSON API (requests)
    2. GoogleMapsReviewScraper - Playwright async, paralel, görsel-engelli
    3. ReviewFilter            - regex cümle bazlı "inci akü" filtresi
    4. IncrementalStore        - data-review-id ile dedupe, atomic JSON write

KURULUM
-------
    pip install requests playwright
    playwright install chromium

ÇALIŞTIRMA
----------
    python inci_aku_scraper.py

KAYNAK
------
- Bayi listesi: GET https://www.inciaku.com/clockwork/surface/bayiler/Get
  (sitenin kendi JS'i bu endpoint'ten yüklüyor; HTML scraping yerine direkt
  JSON kullanıyoruz - hem daha hızlı hem stabil).
  Alanlar: FirmaAdi, Adres, Telefon, Turu, Enlem, Boylam, Name
- Maps eşleştirme: telefon ile arama (primary), telefon boşsa adres (fallback).
- Unique review ID = Google'ın `data-review-id` attribute'ü (stabil).
- CONCURRENCY=5: 300 bayi için makul başlangıç; CPU/ağ'a göre tune et.
"""

import asyncio
import json
import logging
import re
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import requests
from playwright.async_api import (
    Browser,
    Locator,
    Page,
    Route,
    TimeoutError as PlaywrightTimeout,
    async_playwright,
)


# ============================================================
# Configuration
# ============================================================
DEALERS_API = "https://www.inciaku.com/clockwork/surface/bayiler/Get"
DEALERS_REFERER = "https://www.inciaku.com/tr/bayiler-ve-servisler/"
OUTPUT_FILE = Path("inci_aku_yorumlari.json")
MAPS_SEARCH_URL = "https://www.google.com/maps/search/{q}?hl=tr"

KEYWORD_RE = re.compile(r"inci\s*akü", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

CONCURRENCY = 5
NAV_TIMEOUT_MS = 30_000
SCROLL_MAX_ROUNDS = 15
SCROLL_GROW_TIMEOUT_MS = 2_500

BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEAVY_RESOURCES = {"image", "media", "font", "stylesheet"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("inci")


# ============================================================
# Models
# ============================================================
@dataclass(frozen=True)
class Dealer:
    name: str
    address: str
    phone: str
    service_type: str = ""
    lat: Optional[float] = None
    lng: Optional[float] = None

    @property
    def search_query(self) -> str:
        """Maps arama önceliği: telefon → adres."""
        phone_digits = re.sub(r"\D", "", self.phone)
        if len(phone_digits) >= 10:
            return phone_digits
        return self.address


@dataclass
class Review:
    id: str
    dealer_name: str
    dealer_location: str
    review_text: str
    review_date: str


# ============================================================
# Stage 1: Bayi listesi (inciaku.com JSON API)
# ============================================================
class DealerAPIFetcher:
    """inciaku.com'un site içi JSON endpoint'inden bayileri çeker."""

    HEADERS = {
        "User-Agent": BROWSER_UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
        "Referer": DEALERS_REFERER,
        "X-Requested-With": "XMLHttpRequest",
    }

    def __init__(self, url: str = DEALERS_API):
        self.url = url

    def fetch(self) -> list[Dealer]:
        resp = requests.get(
            self.url,
            params={"kategori": "", "sehir": "", "ilce": "", "turu": ""},
            headers=self.HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()
        if not isinstance(raw, list):
            log.error("Beklenmeyen API cevabı: %r", type(raw))
            return []

        dealers: list[Dealer] = []
        seen: set[str] = set()
        for item in raw:
            d = self._parse_item(item)
            if not d:
                continue
            key = f"{d.name}|{d.phone}|{d.address}"
            if key in seen:
                continue
            seen.add(key)
            dealers.append(d)

        log.info("Bayi listesi (API): %d kayıt", len(dealers))
        return dealers

    @staticmethod
    def _parse_item(item: dict) -> Optional[Dealer]:
        name = (item.get("FirmaAdi") or "").strip()
        address = (item.get("Adres") or "").strip()
        phone = (item.get("Telefon") or "").strip()
        if not name or not (phone or address):
            return None
        return Dealer(
            name=name,
            address=address,
            phone=phone,
            service_type=(item.get("Turu") or "").strip(),
            lat=DealerAPIFetcher._coord(item.get("Enlem")),
            lng=DealerAPIFetcher._coord(item.get("Boylam")),
        )

    @staticmethod
    def _coord(raw: object) -> Optional[float]:
        """API koordinatları noktasız döndürüyor (örn '41022287' → 41.022287)."""
        if raw is None:
            return None
        s = str(raw).strip()
        if not s:
            return None
        if "." in s:
            try:
                return float(s)
            except ValueError:
                return None
        if len(s) < 3:
            return None
        try:
            return float(s[:2] + "." + s[2:])
        except ValueError:
            return None


# ============================================================
# Stage 3: Cümle bazlı keyword filtresi
# ============================================================
class ReviewFilter:
    """Yorumu cümlelere böler, 'inci akü' geçen cümleleri tutar."""

    @staticmethod
    def filter_text(text: str) -> str:
        if not text:
            return ""
        sentences = SENTENCE_RE.split(text.strip())
        kept = [s.strip() for s in sentences if KEYWORD_RE.search(s)]
        return " ".join(kept)


# ============================================================
# Stage 2: Google Maps async kazıma
# ============================================================
class GoogleMapsReviewScraper:
    """Playwright async ile paralel, görsel-engelli Google Maps yorum kazıyıcı."""

    FIRST_RESULT = "a.hfpxzc"
    REVIEWS_TAB_CANDIDATES = (
        "button[aria-label*='Yorumlar' i]",
        "button[aria-label*='reviews' i]",
        "button[jsaction*='reviewChart']",
    )
    REVIEWS_FEED = "div[role='feed']"
    REVIEW_CARD = "div[data-review-id]"
    REVIEW_TEXT_SELECTORS = (".wiI7pd", ".MyEned")
    REVIEW_DATE_SELECTORS = (".rsqaWe", ".DU9Pgb .xRkPPb")
    EXPAND_MORE = (
        "button[aria-label='Daha fazla'], "
        "button[jsaction*='expandReview']"
    )
    CONSENT_BUTTON = "form[action*='consent'] button"

    def __init__(self, concurrency: int = CONCURRENCY):
        self.sem = asyncio.Semaphore(concurrency)

    async def scrape_all(
        self, dealers: list[Dealer], known_ids: set[str]
    ) -> list[Review]:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                tasks = [
                    self._scrape_one(browser, d, known_ids) for d in dealers
                ]
                batches = await asyncio.gather(*tasks)
            finally:
                await browser.close()

        return [r for batch in batches for r in batch]

    async def _scrape_one(
        self, browser: Browser, dealer: Dealer, known_ids: set[str]
    ) -> list[Review]:
        async with self.sem:
            ctx = await browser.new_context(user_agent=BROWSER_UA, locale="tr-TR")
            await ctx.route("**/*", self._block_heavy)
            page = await ctx.new_page()
            page.set_default_timeout(NAV_TIMEOUT_MS)
            try:
                return await self._scrape_dealer(page, dealer, known_ids)
            except PlaywrightTimeout as e:
                log.warning(
                    "Timeout [%s]: %s", dealer.name, str(e).splitlines()[0]
                )
                return []
            except Exception as e:
                log.warning("Hata [%s]: %s", dealer.name, e)
                return []
            finally:
                await ctx.close()

    @staticmethod
    async def _block_heavy(route: Route) -> None:
        if route.request.resource_type in HEAVY_RESOURCES:
            await route.abort()
        else:
            await route.continue_()

    async def _scrape_dealer(
        self, page: Page, dealer: Dealer, known_ids: set[str]
    ) -> list[Review]:
        url = MAPS_SEARCH_URL.format(q=urllib.parse.quote(dealer.search_query))
        await page.goto(url, wait_until="domcontentloaded")
        await self._handle_consent(page)

        # Çoklu sonuç gelirse listede ilkini tıkla; tek sonuç direkt detay açar
        try:
            first = page.locator(self.FIRST_RESULT).first
            await first.wait_for(timeout=6_000)
            await first.click()
        except PlaywrightTimeout:
            pass  # muhtemelen direkt place detay sayfası

        await self._open_reviews_section(page)

        feed = page.locator(self.REVIEWS_FEED)
        await feed.wait_for(timeout=12_000)

        await self._scroll_feed(page, feed)
        await self._expand_long_reviews(page)
        return await self._collect_reviews(page, dealer, known_ids)

    async def _handle_consent(self, page: Page) -> None:
        if "consent.google.com" not in page.url:
            return
        try:
            await page.locator(self.CONSENT_BUTTON).last.click(timeout=4_000)
            await page.wait_for_url("**/maps/**", timeout=10_000)
        except PlaywrightTimeout:
            log.debug("Consent diyaloğu işlenemedi (atlanıyor)")

    async def _open_reviews_section(self, page: Page) -> None:
        for sel in self.REVIEWS_TAB_CANDIDATES:
            try:
                await page.locator(sel).first.click(timeout=3_000)
                return
            except PlaywrightTimeout:
                continue
        # Sekme yok → yorumlar muhtemelen inline görünür

    async def _scroll_feed(self, page: Page, feed: Locator) -> None:
        for i in range(SCROLL_MAX_ROUNDS):
            prev = await feed.evaluate("el => el.scrollHeight")
            await feed.evaluate("el => el.scrollTo(0, el.scrollHeight)")
            try:
                await page.wait_for_function(
                    f"prev => document.querySelector(\"{self.REVIEWS_FEED}\")"
                    f".scrollHeight > prev",
                    arg=prev,
                    timeout=SCROLL_GROW_TIMEOUT_MS,
                )
            except PlaywrightTimeout:
                log.debug("Scroll sonu (tur %d)", i + 1)
                return

    async def _expand_long_reviews(self, page: Page) -> None:
        buttons = page.locator(self.EXPAND_MORE)
        count = await buttons.count()
        for i in range(count):
            try:
                await buttons.nth(i).click(timeout=800)
            except PlaywrightTimeout:
                continue

    async def _collect_reviews(
        self, page: Page, dealer: Dealer, known_ids: set[str]
    ) -> list[Review]:
        cards = page.locator(self.REVIEW_CARD)
        total = await cards.count()
        log.info("[%s] %d yorum kartı bulundu", dealer.name, total)

        results: list[Review] = []
        for i in range(total):
            card = cards.nth(i)
            rid = await card.get_attribute("data-review-id")
            if not rid or rid in known_ids:
                continue

            raw_text = await self._first_inner_text(card, self.REVIEW_TEXT_SELECTORS)
            if not raw_text:
                continue
            filtered = ReviewFilter.filter_text(raw_text)
            if not filtered:
                continue

            date = await self._first_inner_text(card, self.REVIEW_DATE_SELECTORS) or ""
            results.append(Review(
                id=rid,
                dealer_name=dealer.name,
                dealer_location=dealer.address,
                review_text=filtered,
                review_date=date,
            ))
        return results

    @staticmethod
    async def _first_inner_text(
        card: Locator, selectors: tuple[str, ...]
    ) -> Optional[str]:
        for sel in selectors:
            try:
                txt = await card.locator(sel).first.inner_text(timeout=600)
                if txt.strip():
                    return txt.strip()
            except PlaywrightTimeout:
                continue
        return None


# ============================================================
# Stage 4: Incremental JSON store
# ============================================================
class IncrementalStore:
    """JSON dosyasını okur, dedupe için ID seti döner; atomic write yapar."""

    def __init__(self, path: Path = OUTPUT_FILE):
        self.path = path

    def load(self) -> tuple[list[dict], set[str]]:
        if not self.path.exists():
            return [], set()
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        ids = {r["id"] for r in data if "id" in r}
        log.info("Mevcut JSON: %d yorum (atlanacak)", len(ids))
        return data, ids

    def save(self, existing: list[dict], new: list[Review]) -> None:
        merged = existing + [asdict(r) for r in new]
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)
        log.info(
            "Yazıldı: %s (toplam %d, yeni %d)", self.path, len(merged), len(new)
        )


# ============================================================
# Pipeline orchestrator
# ============================================================
class Pipeline:
    def __init__(self):
        self.dealers = DealerAPIFetcher()
        self.maps = GoogleMapsReviewScraper()
        self.store = IncrementalStore()

    async def run(self) -> None:
        existing, known_ids = self.store.load()

        try:
            dealers = self.dealers.fetch()
        except requests.RequestException as e:
            log.error("Bayi listesi alınamadı: %s", e)
            return

        if not dealers:
            log.error("Bayi listesi boş; çıkılıyor.")
            return

        new_reviews = await self.maps.scrape_all(dealers, known_ids)
        log.info("Filtrelenmiş yeni yorum: %d", len(new_reviews))
        self.store.save(existing, new_reviews)


if __name__ == "__main__":
    asyncio.run(Pipeline().run())
