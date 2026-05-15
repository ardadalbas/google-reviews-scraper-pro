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
import os
import re
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import requests
from playwright.async_api import (
    Browser,
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
MISMATCH_FILE = Path("inci_aku_telefon_uyusmayan_bayiler.json")

KEYWORD_RE = re.compile(r"inci\s*akü", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

CONCURRENCY = 5
NAV_TIMEOUT_MS = 30_000
FEED_WAIT_MS = 6_000
SCROLL_MAX_ROUNDS = 15
SCROLL_GROW_TIMEOUT_MS = 2_500

# Debug env vars (PowerShell: $env:INCI_HEADED=1; py inci_aku_scraper.py)
HEADLESS = os.environ.get("INCI_HEADED") != "1"
SLOW_MO = int(os.environ.get("INCI_SLOWMO") or "0")
DEALER_LIMIT = int(os.environ.get("INCI_LIMIT") or "0")
DEBUG_DIR = Path("debug")

BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEAVY_RESOURCES = {"image", "media", "font"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("inci")


def normalize_phone(raw: str) -> str:
    """Telefonu sadece rakamlara indirger, ülke kodu (90) ve baştaki 0'ı atar.
    Türk numaraları için son 10 hane hep eşit olmalı."""
    digits = re.sub(r"\D", "", raw or "")
    if digits.startswith("90"):
        digits = digits[2:]
    elif digits.startswith("0"):
        digits = digits[1:]
    return digits


def phones_match(a: str, b: str) -> bool:
    """İki telefonun son 10 hanesi aynıysa True."""
    da, db = normalize_phone(a), normalize_phone(b)
    if not da or not db:
        return False
    return da[-10:] == db[-10:]


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
        """Maps doğal-dil araması: firma adı + tam adres."""
        parts = [p.strip() for p in (self.name, self.address) if p and p.strip()]
        return " ".join(parts)

    @property
    def maps_url(self) -> str:
        """API'den gelen lat/lng URL'in @-kısmına bindirilir → Maps aramayı o
        bölgeye odaklar, fuzzy match isabet oranı dramatik artar."""
        q = urllib.parse.quote(self.search_query)
        base = "https://www.google.com/maps/search/"
        if self.lat is not None and self.lng is not None:
            return f"{base}{q}/@{self.lat},{self.lng},15z?hl=tr"
        return f"{base}{q}?hl=tr"


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
    REVIEW_CARD = "div[data-review-id]"
    CONSENT_BUTTON = "form[action*='consent'] button"

    def __init__(self, concurrency: int = CONCURRENCY):
        self.sem = asyncio.Semaphore(concurrency)
        self.mismatches: list[dict] = []

    async def scrape_all(
        self, dealers: list[Dealer], known_ids: set[str]
    ) -> list[Review]:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=HEADLESS, slow_mo=SLOW_MO)
            try:
                tasks = [
                    self._scrape_one(browser, d, known_ids) for d in dealers
                ]
                batches = await asyncio.gather(*tasks)
            finally:
                await browser.close()

        # Run-içi dedupe: API'de aynı bayi birden fazla kategori entry'siyle
        # gelebiliyor; bunların Maps search'i yanlış yere düşüp (geo-bias) aynı
        # yorumu birden çok bayiden çekebiliyor. ID bazında tekleştir.
        seen: set[str] = set(known_ids)
        merged: list[Review] = []
        for batch in batches:
            for r in batch:
                if r.id in seen:
                    continue
                seen.add(r.id)
                merged.append(r)
        return merged

    async def _scrape_one(
        self, browser: Browser, dealer: Dealer, known_ids: set[str]
    ) -> list[Review]:
        async with self.sem:
            ctx = await browser.new_context(
                user_agent=BROWSER_UA,
                locale="tr-TR",
                viewport={"width": 1920, "height": 1080},
            )
            await ctx.route("**/*", self._block_heavy)
            page = await ctx.new_page()
            page.set_default_timeout(NAV_TIMEOUT_MS)
            try:
                return await self._scrape_dealer(page, dealer, known_ids)
            except PlaywrightTimeout as e:
                log.warning(
                    "Timeout [%s]: %s", dealer.name, str(e).splitlines()[0]
                )
                await self._snapshot(page, dealer)
                return []
            except Exception as e:
                log.warning("Hata [%s]: %s", dealer.name, e)
                await self._snapshot(page, dealer)
                return []
            finally:
                await ctx.close()

    @staticmethod
    async def _snapshot(page: Page, dealer: Dealer) -> None:
        try:
            DEBUG_DIR.mkdir(exist_ok=True)
            safe = re.sub(r"[^\w\-]+", "_", dealer.name)[:50]
            path = DEBUG_DIR / f"{safe}.png"
            await page.screenshot(path=str(path), full_page=False)
            log.info("Screenshot kaydedildi: %s", path)
        except Exception:
            pass  # debug yardımcısı, asıl akışı engellemesin

    @staticmethod
    async def _block_heavy(route: Route) -> None:
        if route.request.resource_type in HEAVY_RESOURCES:
            await route.abort()
        else:
            await route.continue_()

    async def _scrape_dealer(
        self, page: Page, dealer: Dealer, known_ids: set[str]
    ) -> list[Review]:
        await page.goto(dealer.maps_url, wait_until="domcontentloaded")
        await self._handle_consent(page)

        # Çoklu sonuç gelirse listede ilkini tıkla; tek sonuç direkt detay açar
        try:
            first = page.locator(self.FIRST_RESULT).first
            await first.wait_for(timeout=6_000)
            await first.click()
        except PlaywrightTimeout:
            pass  # muhtemelen direkt place detay sayfası

        # Place sayfasının yüklenmesini bekle: URL /maps/search/... iken
        # /maps/place/... olmalı. [role='heading'] search-results filtre
        # başlıklarıyla da eşleşip erken dönüyordu - URL navigasyonu
        # place panel'in gerçekten açıldığının kesin kanıtı.
        try:
            await page.wait_for_url("**/maps/place/**", timeout=8_000)
        except PlaywrightTimeout:
            pass

        # Telefon eşleştirme: Maps'in açtığı place'in telefonu API'deki
        # bayi telefonuyla aynı mı? Aynı değilse Maps yanlış yere düşmüş
        # demektir (geo-bias / fuzzy match yanlışı) - yorumları çekme.
        if not await self._verify_dealer(page, dealer):
            return []

        await self._open_reviews_section(page)

        # div[role='feed'] generic - Maps'te yorum/öneri/foto feed'leri aynı role
        # taşıyor. Doğrudan yorum kartını bekle: false positive yok.
        cards = page.locator(self.REVIEW_CARD)
        await cards.first.wait_for(timeout=FEED_WAIT_MS)

        await self._scroll_to_load_all(page)
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

    async def _verify_dealer(self, page: Page, dealer: Dealer) -> bool:
        """Place'in telefonunu API bayi telefonuyla karşılaştır.

        - Match → True (devam et)
        - Maps'te telefon yok → True (toleranslı, INFO logla)
        - API'de telefon yok → True (karşılaştıracak şey yok)
        - Mismatch → False (yanlış place, atla)
        """
        if not dealer.phone:
            return True

        # Place panel'in telefon butonu yüklenene kadar kısa bekle
        try:
            await page.locator("button[data-item-id^='phone']").first.wait_for(
                timeout=3_000
            )
        except PlaywrightTimeout:
            pass  # buton yoksa toleranslı dön

        maps_phone = await page.evaluate(
            """() => {
                const sels = [
                    "button[data-item-id^='phone']",
                    "button[data-item-id*='phone:tel']",
                    "[aria-label*='Telefon' i][role='button']",
                ];
                for (const s of sels) {
                    const el = document.querySelector(s);
                    if (el) return el.getAttribute('aria-label') || el.innerText || '';
                }
                return null;
            }"""
        )

        if not maps_phone:
            log.info("[%s] Maps'te telefon yok, toleranslı geçildi", dealer.name)
            return True

        if phones_match(maps_phone, dealer.phone):
            return True

        api_norm = normalize_phone(dealer.phone)
        maps_norm = normalize_phone(maps_phone)
        log.info(
            "[%s] Telefon mismatch (API: %s, Maps: %s) - yanlış place, atlandı",
            dealer.name, api_norm, maps_norm,
        )
        # Manuel inceleme için kayıt (asyncio tek thread'de çalışır,
        # list.append GIL altında atomic - lock gerekmez)
        self.mismatches.append({
            "dealer_name": dealer.name,
            "dealer_address": dealer.address,
            "dealer_phone": dealer.phone,
            "dealer_phone_normalized": api_norm,
            "maps_place_phone_raw": maps_phone,
            "maps_place_phone_normalized": maps_norm,
            "maps_url": dealer.maps_url,
        })
        return False

    async def _open_reviews_section(self, page: Page) -> None:
        """Place panelindeki 'Yorumlar' tab'ına tıklar. Tab hiç yoksa bu place'in
        yorumu yok demektir (Maps sıfır-yorumlu yerlerde sekmeyi gizliyor) —
        feed.wait_for sonradan zaten 6s'de timeout olup atlanacak.

        \\b word-boundary regex'i 'Yorumlar' içeren menü item'larıyla karışmasın
        diye gerekli; sadece role='tab' elementler içinde arar.
        """
        tab = page.locator("[role='tab']").filter(
            has_text=re.compile(r"\bYorumlar\b|\bReviews\b", re.I)
        ).first
        try:
            await tab.click(timeout=3_000)
        except PlaywrightTimeout:
            pass

    async def _scroll_to_load_all(self, page: Page) -> None:
        """Yorum listesinin scrollable parent'ını JS ile dinamik bulur, en alta
        scroll eder; data-review-id sayısı artmayı bırakana kadar tekrarlar."""
        scroll_js = """
            () => {
                const cards = document.querySelectorAll('div[data-review-id]');
                if (!cards.length) return false;
                let el = cards[cards.length - 1].parentElement;
                while (el && el.scrollHeight - el.clientHeight < 10) {
                    el = el.parentElement;
                }
                if (!el) return false;
                el.scrollTop = el.scrollHeight;
                return true;
            }
        """
        for _ in range(SCROLL_MAX_ROUNDS):
            prev = await page.locator(self.REVIEW_CARD).count()
            await page.evaluate(scroll_js)
            try:
                await page.wait_for_function(
                    "prev => document.querySelectorAll('div[data-review-id]').length > prev",
                    arg=prev,
                    timeout=SCROLL_GROW_TIMEOUT_MS,
                )
            except PlaywrightTimeout:
                return

    async def _expand_long_reviews(self, page: Page) -> None:
        """Tüm 'Daha fazla' butonlarını tek JS çağrısında tıkla - per-button
        Playwright click 178 kartta 2 dakika sürerdi."""
        await page.evaluate(
            """() => {
                const sel = "button[aria-label='Daha fazla'], "
                          + "button[jsaction*='expandReview']";
                document.querySelectorAll(sel).forEach(b => b.click());
            }"""
        )

    async def _collect_reviews(
        self, page: Page, dealer: Dealer, known_ids: set[str]
    ) -> list[Review]:
        """Tüm kart verisini tek JS evaluate'le çek - per-card Playwright
        sorgusu 178 kartta ~90 saniye sürerken bu yaklaşık 200 ms."""
        raw = await page.evaluate(
            """() => {
                const cards = document.querySelectorAll('div[data-review-id]');
                return Array.from(cards).map(c => ({
                    id: c.getAttribute('data-review-id'),
                    text: (c.querySelector('.wiI7pd, .MyEned')?.innerText || '').trim(),
                    date: (c.querySelector('.rsqaWe, .DU9Pgb .xRkPPb')?.innerText || '').trim(),
                }));
            }"""
        )
        log.info("[%s] %d yorum kartı bulundu", dealer.name, len(raw))

        results: list[Review] = []
        for r in raw:
            rid = r.get("id")
            if not rid or rid in known_ids:
                continue
            text = r.get("text") or ""
            if not text:
                continue
            filtered = ReviewFilter.filter_text(text)
            if not filtered:
                continue
            results.append(Review(
                id=rid,
                dealer_name=dealer.name,
                dealer_location=dealer.address,
                review_text=filtered,
                review_date=r.get("date") or "",
            ))
        return results


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

        if DEALER_LIMIT > 0:
            log.info("DEBUG: INCI_LIMIT=%d, ilk %d bayi taranacak", DEALER_LIMIT, DEALER_LIMIT)
            dealers = dealers[:DEALER_LIMIT]

        new_reviews = await self.maps.scrape_all(dealers, known_ids)
        log.info("Filtrelenmiş yeni yorum: %d", len(new_reviews))
        self.store.save(existing, new_reviews)
        self._save_mismatches()

    def _save_mismatches(self) -> None:
        """Telefon eşleşmeyen bayileri ayrı dosyaya yazar (manuel inceleme için)."""
        if not self.maps.mismatches:
            log.info("Telefon mismatch: 0 bayi")
            return
        tmp = MISMATCH_FILE.with_suffix(MISMATCH_FILE.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.maps.mismatches, f, ensure_ascii=False, indent=2)
        tmp.replace(MISMATCH_FILE)
        log.info(
            "Telefon mismatch: %d bayi → %s (Maps URL'leri ile birlikte)",
            len(self.maps.mismatches), MISMATCH_FILE,
        )


if __name__ == "__main__":
    asyncio.run(Pipeline().run())
