#!/usr/bin/env python3
"""
inci_aku_scraper.py
===================

İnci Akü bayilerinin Google Maps yorumlarını yüksek hızda kazır, içinde
"inci akü" geçen cümleleri çıkarır ve artımlı olarak JSON'a yazar.

Mimari (4 aşama):
    1. DealerListScraper       - inciaku.com bayi listesi (requests + BS4)
    2. GoogleMapsReviewScraper - Playwright async, paralel, görsel-engelli
    3. ReviewFilter            - regex cümle bazlı "inci akü" filtresi
    4. IncrementalStore        - data-review-id ile dedupe, atomic JSON write

KURULUM
-------
    pip install requests beautifulsoup4 playwright
    playwright install chromium

ÇALIŞTIRMA
----------
    python inci_aku_scraper.py

VARSAYIMLAR (üst bölümdeki sabitleri gerekirse güncelle)
---------------------------------------------------------
- inciaku.com static HTML döndürür. WAF/CDN doğrulaması yapılmadı; HTTP 403
  alınırsa BROWSER_UA güncellenmeli ya da bu aşama da Playwright'a alınmalı.
- Bayi kart selector'ları (CARD_SELECTORS) sayfa görülmeden defansif yazıldı;
  ilk eşleşen kullanılır, hiçbiri tutmazsa log uyarır.
- Search query = "{Bayi Adı} {Konum}". İlk sonuç doğru bayi varsayılır.
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
from bs4 import BeautifulSoup
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
DEALER_URL = "https://www.inciaku.com/tr/bayiler-ve-servisler/"
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
    location: str

    @property
    def search_query(self) -> str:
        return f"{self.name} {self.location}"


@dataclass
class Review:
    id: str
    dealer_name: str
    dealer_location: str
    review_text: str
    review_date: str


# ============================================================
# Stage 1: Bayi listesi (requests + BeautifulSoup)
# ============================================================
class DealerListScraper:
    """inciaku.com'dan bayi adı ve il/ilçe bilgisini çeker."""

    CARD_SELECTORS = (
        ".bayi-card",
        ".dealer-card",
        ".servis-item",
        ".bayilik .col",
        "[class*='bayi']",
    )
    NAME_SELECTORS = (".name", ".bayi-adi", ".title", "h3", "h4", "strong")
    LOC_SELECTORS = (".location", ".adres", ".il-ilce", ".sehir", "p")

    HEADERS = {
        "User-Agent": BROWSER_UA,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
    }

    def __init__(self, url: str = DEALER_URL):
        self.url = url

    def fetch(self) -> list[Dealer]:
        resp = requests.get(self.url, headers=self.HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        cards = self._find_cards(soup)
        if not cards:
            log.error(
                "Bayi kartı bulunamadı. Sayfayı inceleyip CARD_SELECTORS "
                "listesini gerçek HTML class'larına göre güncelle."
            )
            return []

        dealers: list[Dealer] = []
        seen: set[tuple[str, str]] = set()
        for card in cards:
            name = self._first_text(card, self.NAME_SELECTORS)
            loc = self._first_text(card, self.LOC_SELECTORS)
            if not name or not loc:
                continue
            key = (name, loc)
            if key in seen:
                continue
            seen.add(key)
            dealers.append(Dealer(name=name, location=loc))

        log.info("Bayi listesi: %d kayıt", len(dealers))
        return dealers

    def _find_cards(self, soup: BeautifulSoup) -> list:
        for sel in self.CARD_SELECTORS:
            found = soup.select(sel)
            if found:
                log.info("Bayi selector eşleşti: %s (%d kart)", sel, len(found))
                return found
        return []

    @staticmethod
    def _first_text(node, selectors: tuple[str, ...]) -> Optional[str]:
        for sel in selectors:
            el = node.select_one(sel)
            if el and (txt := el.get_text(" ", strip=True)):
                return txt
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
                dealer_location=dealer.location,
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
        self.dealers = DealerListScraper()
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
