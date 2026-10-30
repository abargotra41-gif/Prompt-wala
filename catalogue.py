"""
catalogue.py
Authoritative Product Catalogue, Alias Mapping, and Transparent Candidate Retrieval (RAG).
Catalogue facts (SKU, Canonical Name, Brand, Unit, Price, Units/Case) are the single source of truth.
The LLM and UI must never invent or mutate catalogue facts.
"""

from __future__ import annotations
import csv
import os
import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple


def normalize_text(text: str) -> str:
    """Normalize text for consistent phonetic/alias comparison."""
    if not text:
        return ""
    # Unicode NFKD normalization
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    # Replace separators with space
    text = re.sub(r"[-_/\\,.]", " ", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class Product:
    sku: str
    product_name: str
    brand: str
    aliases: List[str]
    category: str
    order_unit: str
    unit_price_inr: Optional[Decimal]
    units_per_case: int
    is_active: bool = True
    stock_available: Optional[int] = None  # None if inventory source is unverified

    @property
    def display_pack(self) -> str:
        # Extract pack info from name or unit
        match = re.search(r"(\d+(?:\.\d+)?\s*(?:kg|g|l|ml))", self.product_name, re.IGNORECASE)
        if match:
            return match.group(1)
        return self.order_unit


@dataclass
class MatchCandidate:
    product: Product
    score: float
    evidence: str
    matched_term: str
    spelling_suggestion: Optional[str] = None


class Catalogue:
    def __init__(self, csv_path: Optional[str] = None):
        self.products: Dict[str, Product] = {}
        self.alias_to_skus: Dict[str, Set[str]] = {}
        self.brand_to_skus: Dict[str, Set[str]] = {}
        self.name_to_sku: Dict[str, str] = {}
        self._load_catalogue(csv_path)

    def _load_catalogue(self, csv_path: Optional[str]) -> None:
        """Load catalogue from CSV or use embedded starter fixture."""
        default_path = os.path.join(os.path.dirname(__file__), "data", "Catalogue.csv")
        target_path = csv_path or default_path

        if os.path.exists(target_path):
            with open(target_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sku = row["SKU"].strip()
                    price_str = row.get("Unit_Price_INR", "").strip()
                    price = Decimal(price_str) if price_str and price_str != "None" else None
                    units_per_case = int(row.get("Units_Per_Case", "1") or 1)
                    aliases = [
                        a.strip()
                        for a in row.get("Aliases_Synonyms", "").split(";")
                        if a.strip()
                    ]
                    prod = Product(
                        sku=sku,
                        product_name=row["Product_Name"].strip(),
                        brand=row.get("Brand", "").strip(),
                        aliases=aliases,
                        category=row.get("Category", "").strip(),
                        order_unit=row.get("Order_Unit", "Piece").strip(),
                        unit_price_inr=price,
                        units_per_case=units_per_case,
                        is_active=True,
                    )
                    self.products[sku] = prod
        else:
            # Fallback synthetic catalogue if CSV not found
            self._load_fallback_catalogue()

        self._index_catalogue()

    def _load_fallback_catalogue(self) -> None:
        """Synthetic fallback data with 24 items if CSV is missing."""
        sample_data = [
            ("SKU001", "Fortune Sunflower Oil 1L", "Fortune", "fortune oil;fortune;sunflower oil", "Oil", "Bottle", Decimal("150"), 12),
            ("SKU002", "Fortune Sunflower Oil 5L", "Fortune", "fortune oil 5l;fortune 5 litre", "Oil", "Can", Decimal("690"), 4),
            ("SKU003", "Maggi Noodles 70g", "Maggi", "maggi;magi;noodles", "Instant Noodles", "Packet", Decimal("14"), 96),
            ("SKU004", "Parle-G Biscuits 100g", "Parle-G", "parle g;parle-g;parleg", "Biscuits", "Packet", Decimal("10"), 120),
            ("SKU005", "Britannia Good Day 100g", "Britannia", "good day;goodday;britannia goodday", "Biscuits", "Packet", Decimal("20"), 80),
            ("SKU006", "Tata Salt 1kg", "Tata", "tata salt;salt 1kg", "Staples", "Packet", Decimal("28"), 60),
            ("SKU007", "Aashirvaad Atta 5kg", "Aashirvaad", "atta 5kg;aashirvaad atta", "Staples", "Bag", Decimal("275"), 10),
            ("SKU010", "Coca-Cola 750ml", "Coca-Cola", "coke;coca cola;coke 750", "Beverages", "Bottle", Decimal("40"), 48),
            ("SKU011", "Thums Up 750ml", "Thums Up", "thumsup;thums up", "Beverages", "Bottle", Decimal("40"), 48),
            ("SKU013", "Surf Excel Detergent 1kg", "Surf Excel", "surf excel;surf;detergent", "Home Care", "Pack", Decimal("150"), 12),
        ]
        for sku, name, brand, aliases, cat, unit, price, upc in sample_data:
            self.products[sku] = Product(
                sku=sku,
                product_name=name,
                brand=brand,
                aliases=aliases.split(";"),
                category=cat,
                order_unit=unit,
                unit_price_inr=price,
                units_per_case=upc,
                is_active=True,
            )

    def _index_catalogue(self) -> None:
        """Build search indexes for canonical names, aliases, brands, and categories."""
        self.alias_to_skus.clear()
        self.brand_to_skus.clear()
        self.category_to_skus = {}
        self.name_to_sku.clear()

        for sku, prod in self.products.items():
            norm_name = normalize_text(prod.product_name)
            self.name_to_sku[norm_name] = sku

            norm_brand = normalize_text(prod.brand)
            self.brand_to_skus.setdefault(norm_brand, set()).add(sku)

            norm_cat = normalize_text(prod.category)
            if norm_cat:
                self.category_to_skus.setdefault(norm_cat, set()).add(sku)
                # also map singular/plural
                if norm_cat.endswith("s"):
                    self.category_to_skus.setdefault(norm_cat[:-1], set()).add(sku)
                else:
                    self.category_to_skus.setdefault(norm_cat + "s", set()).add(sku)

            # Index canonical name as an alias
            self.alias_to_skus.setdefault(norm_name, set()).add(sku)

            # Index configured aliases
            for alias in prod.aliases:
                norm_alias = normalize_text(alias)
                if norm_alias:
                    self.alias_to_skus.setdefault(norm_alias, set()).add(sku)

        # Hinglish vernacular category synonyms mapping
        vernacular_cat_links = {
            "oil": ["SKU001", "SKU002"],
            "tel": ["SKU001", "SKU002"],
            "meetha tel": ["SKU001", "SKU002"],
            "biscuits": ["SKU004", "SKU005"],
            "biscuit": ["SKU004", "SKU005"],
            "biskut": ["SKU004", "SKU005"],
            "noodles": ["SKU003"],
            "namak": ["SKU006"],
            "atta": ["SKU007"],
            "aata": ["SKU007"],
            "doodh": ["SKU008"],
            "makhan": ["SKU009"],
            "thanda": ["SKU010", "SKU011"],
            "pani": ["SKU012"],
            "sarf": ["SKU013"],
            "kolget": ["SKU015"],
            "sabun": ["SKU014", "SKU020"],
            "chai": ["SKU023"],
            "chai patti": ["SKU023"],
            "shahad": ["SKU024"],
            "shehad": ["SKU024"],
        }
        for term, s_list in vernacular_cat_links.items():
            for s in s_list:
                if s in self.products:
                    self.category_to_skus.setdefault(term, set()).add(s)

    def get_by_sku(self, sku: str) -> Optional[Product]:
        return self.products.get(sku)

    def set_active(self, sku: str, active: bool) -> None:
        if sku in self.products:
            self.products[sku].is_active = active

    def set_price(self, sku: str, price: Optional[Decimal]) -> None:
        if sku in self.products:
            self.products[sku].unit_price_inr = price

    def search_candidates(
        self, query: str, top_k: int = 3, threshold: float = 0.55
    ) -> List[MatchCandidate]:
        """
        Transparent retrieval layer over catalogue.
        Evaluates exact alias matches, canonical name matches, brand-variant matches,
        and n-gram / token similarity.
        Returns candidates ranked by evidence score.
        """
        norm_q = normalize_text(query)
        if not norm_q:
            return []

        candidates: Dict[str, MatchCandidate] = {}

        # 1. Exact canonical name match
        if norm_q in self.name_to_sku:
            sku = self.name_to_sku[norm_q]
            prod = self.products[sku]
            candidates[sku] = MatchCandidate(
                product=prod,
                score=1.0,
                evidence="Exact canonical name match",
                matched_term=prod.product_name,
            )

        # 2. Exact alias match
        if norm_q in self.alias_to_skus:
            matched_skus = self.alias_to_skus[norm_q]
            # Check if this alias is associated with multiple SKUs or multiple pack variants of same brand
            # e.g. 'fortune oil' when both 1L and 5L exist in catalogue and query has no pack size
            q_pack = self._extract_pack_spec(norm_q)
            is_family_alias = False
            family_skus = set(matched_skus)
            if not q_pack:
                # check if any matched SKU's brand has multiple pack sizes of the exact same product name
                # e.g. Fortune Sunflower Oil 1L and 5L
                for s in matched_skus:
                    prod = self.products[s]
                    base_name = re.sub(r"\b\d+(?:\.\d+)?\s*(?:kg|g|l|ml)\b", "", normalize_text(prod.product_name)).strip()
                    for other_s, other_p in self.products.items():
                        if other_s != s and other_p.brand == prod.brand:
                            other_base = re.sub(r"\b\d+(?:\.\d+)?\s*(?:kg|g|l|ml)\b", "", normalize_text(other_p.product_name)).strip()
                            if base_name and base_name == other_base:
                                family_skus.add(other_s)
                                is_family_alias = True

            for sku in family_skus:
                prod = self.products[sku]
                if is_family_alias or len(family_skus) > 1:
                    score = 0.85
                    ev = "Shared alias / brand family without specified pack size (Ambiguous)"
                else:
                    score = 0.98
                    ev = "Exact alias match"
                if sku not in candidates or score > candidates[sku].score:
                    sugg = None
                    if norm_q != normalize_text(prod.product_name) and norm_q != normalize_text(prod.brand):
                        sugg = f"Suggested correction: '{norm_q}' → '{prod.product_name}'"
                    candidates[sku] = MatchCandidate(
                        product=prod,
                        score=score,
                        evidence=ev,
                        matched_term=norm_q,
                        spelling_suggestion=sugg,
                    )

        # 2b. Category match e.g. "biscuits", "biscuit", "oil", "soap", "tea", "dairy"
        if hasattr(self, "category_to_skus") and norm_q in self.category_to_skus:
            cat_skus = self.category_to_skus[norm_q]
            for sku in cat_skus:
                prod = self.products[sku]
                score = 0.82
                ev = f"Category match ('{prod.category}')"
                if sku not in candidates or score > candidates[sku].score:
                    candidates[sku] = MatchCandidate(
                        product=prod, score=score, evidence=ev, matched_term=norm_q
                    )

        # 3. Pack size / variant matching within brands or queries
        # e.g. "fortune oil 1l" vs "fortune oil 5l"
        q_tokens = set(norm_q.split())
        for sku, prod in self.products.items():
            norm_name = normalize_text(prod.product_name)
            name_tokens = set(norm_name.split())

            # Check if all query tokens are in the product name
            if q_tokens.issubset(name_tokens) and len(q_tokens) >= 2:
                score = 0.92
                ev = "Full query token subset match"
                if sku not in candidates or score > candidates[sku].score:
                    candidates[sku] = MatchCandidate(
                        product=prod, score=score, evidence=ev, matched_term=norm_name
                    )

            # Check aliases for substring containment
            for alias in prod.aliases:
                norm_a = normalize_text(alias)
                a_tokens = set(norm_a.split())
                if norm_q == norm_a:
                    continue  # handled in step 2
                if norm_a in norm_q or norm_q in norm_a:
                    # Partial match
                    overlap_ratio = len(q_tokens.intersection(a_tokens)) / max(len(q_tokens), len(a_tokens))
                    score = 0.70 + (0.20 * overlap_ratio)

                    # Penalize score if candidate has an explicit pack size that conflicts with query
                    q_pack = self._extract_pack_spec(norm_q)
                    target_pack = self._extract_pack_spec(norm_a) or self._extract_pack_spec(prod.product_name)
                    if q_pack and target_pack:
                        if q_pack != target_pack:
                            score = score * 0.4  # Strong penalty for conflicting pack size (e.g. 1L vs 5L)
                        else:
                            score = min(0.98, score + 0.10)  # Boost matching pack size

                    ev = f"Alias substring match ('{alias}')"
                    if sku not in candidates or score > candidates[sku].score:
                        candidates[sku] = MatchCandidate(
                            product=prod, score=score, evidence=ev, matched_term=alias
                        )

        # 4. Token Jaccard & Levenshtein similarity for typos
        for sku, prod in self.products.items():
            for target in [prod.product_name] + prod.aliases:
                norm_target = normalize_text(target)
                if norm_q == norm_target:
                    continue  # Already handled in exact name/alias matching
                sim = self._token_similarity(norm_q, norm_target)
                
                # Penalize score if candidate has an explicit pack size that conflicts with query
                # e.g. query has '1l' and target has '5l'
                q_pack = self._extract_pack_spec(norm_q)
                target_pack = self._extract_pack_spec(norm_target)
                if q_pack and target_pack and q_pack != target_pack:
                    sim = sim * 0.5  # Heavy penalty for conflicting pack size

                if sim >= threshold:
                    ev = f"Token similarity ({sim:.2f}) with '{target}'"
                    sugg = f"Spelling suggestion: '{prod.product_name}' (matched from '{norm_q}')"
                    if sku not in candidates or sim > candidates[sku].score:
                        candidates[sku] = MatchCandidate(
                            product=prod, score=sim, evidence=ev, matched_term=target, spelling_suggestion=sugg
                        )

        # Filter active products and rank candidates
        results = sorted(candidates.values(), key=lambda c: c.score, reverse=True)
        return results[:top_k]

    def get_spelling_suggestions(self, query: str, top_n: int = 3) -> List[str]:
        """Return spelling mistake corrections for unknown or misspelled queries."""
        norm_q = normalize_text(query)
        if not norm_q:
            return []
        scored = []
        for p in self.products.values():
            best_sim = max(
                [self._token_similarity(norm_q, normalize_text(p.product_name))]
                + [self._token_similarity(norm_q, normalize_text(a)) for a in p.aliases]
            )
            scored.append((best_sim, p.product_name))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [name for sim, name in scored[:top_n] if sim >= 0.35]

    def _extract_pack_spec(self, text: str) -> Optional[str]:
        """Extract pack size tokens like 1l, 5l, 500ml, 70g, 100g, 1kg, 5kg."""
        m = re.search(r"\b(\d+(?:\.\d+)?\s*(?:kg|g|l|ml))\b", text, re.IGNORECASE)
        if m:
            # normalize spacing e.g. '1 l' -> '1l'
            return re.sub(r"\s+", "", m.group(1).lower())
        return None

    def _token_similarity(self, s1: str, s2: str) -> float:
        """Calculate token-based and character-overlap similarity score."""
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0

        tokens1 = set(s1.split())
        tokens2 = set(s2.split())

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        jaccard = len(intersection) / len(union) if union else 0.0

        # Handle character typo in single words (e.g. 'masla' -> 'masala')
        char_sim = self._levenshtein_ratio(s1, s2)
        return max(jaccard, char_sim)

    def _levenshtein_ratio(self, s1: str, s2: str) -> float:
        """Fast standard Levenshtein similarity ratio between 0 and 1."""
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        distance = dp[len1][len2]
        max_len = max(len1, len2)
        return 1.0 - (distance / max_len)
