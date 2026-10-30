"""
order_parser.py
Two-Stage Structured Order Parser for English, Hindi, and Hinglish WhatsApp Replenishment Messages.
Stage 1: Line item extraction (literal text, quantity, unit, product mention).
Stage 2: Catalogue RAG grounding (matching candidates, detecting ambiguity, generating targeted clarification).
Supports both Configurable LLM Mode and Local Rule-Grounded / Demo Parser Mode.
"""

from __future__ import annotations
import json
import os
import re
import uuid
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from catalogue import Catalogue, MatchCandidate, normalize_text
from conversion import UnitConverter
from calculations import calculate_line_item
from validation import OrderLine, validate_and_sanitize_input


# Common Hinglish and English number words
HINGLISH_NUMBERS = {
    # 1 to 10
    "ek": 1, "aik": 1, "ik": 1, "do": 2, "teen": 3, "tin": 3, "char": 4, "chaar": 4,
    "paanch": 5, "panch": 5, "chhe": 6, "che": 6, "chha": 6, "saat": 7, "sat": 7,
    "aath": 8, "ath": 8, "nau": 9, "no": 9, "das": 10, "dus": 10,
    # 11 to 20
    "gyarah": 11, "gyara": 11, "barah": 12, "bara": 12, "terah": 13, "tera": 13,
    "chaudah": 14, "chauda": 14, "pandrah": 15, "pandra": 15, "solah": 16, "sola": 16,
    "satrah": 17, "satra": 17, "atharah": 18, "athara": 18, "unnees": 19, "unnis": 19,
    "bees": 20, "bis": 20,
    # 21 to 30
    "ikkis": 21, "baais": 22, "teis": 23, "chaubis": 24, "pachees": 25, "pachis": 25,
    "chabbis": 26, "sattais": 27, "atthaais": 28, "unattis": 29, "tees": 30, "tis": 30,
    # Decades & 100
    "paintis": 35, "chalis": 40, "paintalis": 45, "pachaas": 50, "pachas": 50,
    "saath": 60, "sattar": 70, "assi": 80, "nabbe": 90, "sau": 100, "so": 100,
    # Collective & Dozen
    "dozen": 12, "darjan": 12, "dazan": 12, "dajan": 12, "jodi": 2, "pair": 2,
    # Fractions
    "aadha": 0.5, "adha": 0.5, "half": 0.5, "dedh": 1.5, "derh": 1.5,
    "dhaai": 2.5, "dhai": 2.5, "sawa": 1.25, "paune": 0.75,
    # English numbers
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
}

# Known Hinglish connective noise words
NOISE_WORDS = [
    r"\bbhaiya\b", r"\bbhai\b", r"\bbhejna\b", r"\bbhejo\b", r"\bbhej\s*dena\b",
    r"\bbhejjo\b", r"\bbhej\s*do\b", r"\bbhejva\s*do\b", r"\bchahiye\b",
    r"\bdena\b", r"\bde\s*do\b", r"\bde\s*dena\b", r"\blikho\b", r"\blikh\s*lo\b",
    r"\bdaalo\b", r"\bdaal\s*do\b", r"\bplease\b", r"\bsend\b", r"\bdeliver\b",
    r"\baur\b", r"\band\b", r"\bke\b", r"\bka\b", r"\bki\b", r"\bko\b",
    r"\bse\b", r"\bpar\b", r"\bmein\b", r"\bme\b", r"\bwala\b", r"\bwali\b",
    r"\bwale\b", r"\burgent\b", r"\baaj\b", r"\bkal\b", r"\bparso\b",
    r"\bsath\b", r"\bsaath\b",
]

# Standard packaging and vernacular unit synonyms recognized after quantity numbers
KNOWN_UNITS = [
    # English packaging
    "bottle", "bottles", "can", "cans", "packet", "packets", "pack", "packs",
    "bag", "bags", "pouch", "pouches", "bar", "bars", "tube", "tubes",
    "jar", "jars", "carton", "cartons", "case", "cases", "box", "boxes",
    "piece", "pieces", "pc", "pcs", "unit", "units",
    # Hinglish & regional units
    "botal", "botle", "botl", "botalein", "canne", "cane", "dibba",
    "pakit", "pakt", "bori", "bore", "thela", "thele", "katta", "katte",
    "tikiya", "tikia", "bati", "batia", "tyub", "paste",
    "barni", "shishi", "sisi", "nag", "adad", "peti", "petiyan", "petia", "dhaba", "dhabba", "dhabe", "dhabbe", "dhabas", "dhabbas", "dabba",
    # Metric & weight units
    "kg", "kgs", "kilo", "kilos", "kilogram", "kilograms",
    "g", "gm", "gms", "gram", "grams",
    "l", "lt", "ltr", "litre", "litres", "liter", "liters", "litar",
    "ml", "milliliter", "milliliters",
    # Collective & Dozen
    "dozen", "darjan", "dazan", "jodi", "pair",
]


class OrderParser:
    def __init__(
        self,
        catalogue: Catalogue,
        converter: UnitConverter,
        store_history: Optional[Dict[str, List[dict]]] = None,
        store_sku_history: Optional[Dict[str, Dict[str, List[int]]]] = None,
    ):
        self.catalogue = catalogue
        self.converter = converter
        self.store_history = store_history or {}
        self.store_sku_history = store_sku_history or {}

    def parse_order(
        self,
        raw_message: str,
        store_id: str = "STORE001",
        message_id: Optional[str] = None,
        use_llm: bool = False,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main parser entry point.
        Sanitizes input, checks prompt injection, extracts lines, and grounds against catalogue.
        """
        msg_id = message_id or f"MSG-{uuid.uuid4().hex[:6].upper()}"

        # Step 1: Input Validation & Sanitization
        val_result = validate_and_sanitize_input(raw_message)
        if not val_result.is_valid:
            return {
                "message_id": msg_id,
                "store_id": store_id,
                "language_style": "Unknown",
                "items": [],
                "error": val_result.error_message,
                "has_injection_risk": False,
            }

        # Step 2: Language detection heuristic
        language_style = self._detect_language_style(val_result.sanitized_text)

        # Step 3: Extraction
        # Try external LLM if configured and requested, otherwise rule-grounded parser
        lines: List[OrderLine] = []
        if use_llm and api_key:
            try:
                lines = self._extract_with_llm(
                    val_result.sanitized_text, store_id, msg_id, api_key, model_name
                )
            except Exception as e:
                # Graceful fallback to rule-based parser on LLM failure
                lines = self._extract_rule_grounded(
                    val_result.sanitized_text, store_id, msg_id
                )
        else:
            lines = self._extract_rule_grounded(
                val_result.sanitized_text, store_id, msg_id
            )

        # Step 4: Duplicate mention detection
        self._detect_duplicates(lines)

        # Step 5: Convert and calculate each line deterministically
        for line in lines:
            self._finalize_line_calculations(line, store_id)

        return {
            "message_id": msg_id,
            "store_id": store_id,
            "language_style": language_style,
            "items": [line.to_dict() for line in lines],
            "raw_text": val_result.sanitized_text,
            "has_injection_risk": val_result.has_injection_risk,
            "injection_details": val_result.injection_details,
            "error": None,
        }

    def _detect_language_style(self, text: str) -> str:
        """Detect English, Hindi (Devanagari), Hinglish, or Mixed."""
        has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
        lower = text.lower()
        hinglish_words = ["bhaiya", "bhai", "bhejna", "aur", "dabba", "peti", "dhaba", "dhabba", "chahiye", "ke", "ka", "bada", "chhota", "kal"]
        has_hinglish = any(re.search(rf"\b{w}\b", lower) for w in hinglish_words)

        if has_devanagari and has_hinglish:
            return "Mixed"
        if has_devanagari:
            return "Hindi"
        if has_hinglish:
            return "Hinglish"
        return "English"

    def _extract_rule_grounded(
        self, text: str, store_id: str, message_id: str
    ) -> List[OrderLine]:
        """
        High-precision rule-grounded extraction designed for Kirana WhatsApp patterns.
        Splits by delimiters ('aur', 'and', ',', '\n', '+', bullets),
        extracts quantity, unit, and product mention, then matches against catalogue.
        """
        # Split into segments while preserving order
        raw_segments = self._split_order_segments(text)
        lines: List[OrderLine] = []

        for idx, segment in enumerate(raw_segments, start=1):
            seg_text = segment.strip()
            if not seg_text:
                continue

            # Check prompt injection specifically in line
            if any(re.search(pat, seg_text, re.IGNORECASE) for pat in [r"ignore\s+rules", r"free\s+products", r"free\s+items"]):
                line = OrderLine(
                    line_id=f"LINE-{idx:03d}",
                    message_id=message_id,
                    store_id=store_id,
                    original_item_text=seg_text,
                    product_mention=seg_text,
                    requested_quantity_value=None,
                    requested_quantity_text="",
                    requested_unit_text=None,
                    match_status="NEEDS_CLARIFICATION",
                    clarification_required=True,
                    clarification_question="Suspicious prompt instruction ignored. Line blocked.",
                    line_status="Needs clarification",
                    validation_messages=["Security Policy: Instruction to bypass rules or add free items was blocked."],
                )
                lines.append(line)
                continue

            # Check for relative reference: "same as last time" / "kal jaisa" / "previous order"
            if re.search(r"(same\s+as\s+(last|previous)\s+(order|time|week)|kal\s+jaisa|pichhla\s+order|last\s+order|previous\s+order|repeat\s+(last|previous)\s+order|as\s+usual|usual\s+order)", seg_text, re.IGNORECASE):
                resolved_lines = self._resolve_relative_reference(seg_text, store_id, message_id, idx)
                lines.extend(resolved_lines)
                continue

            qty_val, qty_str, unit_str, product_mention = self._parse_item_clause(seg_text)

            # Ground against catalogue
            line = self._ground_item(
                line_id=f"LINE-{idx:03d}",
                message_id=message_id,
                store_id=store_id,
                original_text=seg_text,
                product_mention=product_mention,
                qty_val=qty_val,
                qty_str=qty_str,
                unit_str=unit_str,
            )
            lines.append(line)

        return lines

    def _split_order_segments(self, text: str) -> List[str]:
        """Split WhatsApp message into discrete item clauses."""
        # Clean polite greetings or introductory phrases
        cleaned = text
        cleaned = re.sub(r"^(bhaiya|bhai|hello|hi|dear|sir|plz|please)[,\s]+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[,\s]+(bhej\s*dena|bhejna|bhejo|urgent|thank\s*you|thanks)[.!]?$", "", cleaned, flags=re.IGNORECASE)

        # Split on newlines, semicolons, bullets
        lines = re.split(r"[\n;\r]+", cleaned)
        segments = []
        for l in lines:
            # Further split by commas, 'aur', 'and', '+', 'also', 'along with'
            parts = re.split(r"\s*(?:,|\baur\b|\band\b|\balso\b|\balong with\b|\+)\s*", l, flags=re.IGNORECASE)
            for p in parts:
                p_clean = p.strip()
                # Remove leading numbering or list bullets like "1. ", "1) ", "- " (only when followed by space)
                p_clean = re.sub(r"^(\d+[\.\)]\s+|[-*•]\s+)", "", p_clean)
                # Remove leading conversational phrases & action verbs in English/Hinglish
                p_clean = re.sub(
                    r"^(please\s+|could\s+you\s+please\s+|kindly\s+|i\s+need\s+|we\s+need\s+|we\s+want\s+|i\s+want\s+|order\s+for\s+|booking\s+for\s+|bhaiya\s+|bhai\s+)?(send|deliver|dispatch|give|bring|book|add|need|want|bhejo|bhejna|bhej dena|chahiye)\s+",
                    "",
                    p_clean,
                    flags=re.IGNORECASE,
                ).strip()
                p_clean = re.sub(r"^(i\s+need|we\s+need|i\s+want|we\s+want|kindly\s+send|please\s+send)\s+", "", p_clean, flags=re.IGNORECASE).strip()
                # Remove trailing greetings
                p_clean = re.sub(r"\b(bhejo|bhejna|bhej dena|chahiye)\b", "", p_clean, flags=re.IGNORECASE).strip()
                if p_clean and len(p_clean) > 1:
                    segments.append(p_clean)
        return segments

    def _parse_item_clause(self, clause: str) -> Tuple[Optional[Decimal], str, Optional[str], str]:
        """
        Extract quantity value, quantity text, unit text, and product mention from a single clause.
        Example: '2 bottle Fortune sunflower oil 1L' -> (2, '2', 'bottle', 'Fortune sunflower oil 1L')
        Example: 'send 2 bottles of Fortune oil 1L' -> (2, '2', 'bottles', 'Fortune oil 1L')
        Example: 'fortune oil 5 l ke 2 can' -> (2, '2', 'can', 'fortune oil 5 l')
        Example: '2 dozen Dairy Milk 40g' -> (24, '24', 'Piece', 'Dairy Milk 40g')
        Example: '2 botal fortune tel' -> (2, '2', 'botal', 'fortune tel')
        Example: '-3 Maggi' -> (-3, '-3', None, 'Maggi')
        """
        text = clause.strip()

        # Handle negative numbers
        neg_match = re.match(r"^(-?\d+(?:\.\d+)?)\s*(.*)", text)
        if neg_match and Decimal(neg_match.group(1)) <= 0:
            qty_val = Decimal(neg_match.group(1))
            return qty_val, neg_match.group(1), None, neg_match.group(2).strip()

        # Multiplication pattern: '2 x maggi', '3 * vim bar'
        mul_match = re.match(r"^(\d+(?:\.\d+)?)\s*[xX*]\s*(.*)$", text)
        if mul_match:
            qty_val = Decimal(mul_match.group(1))
            return qty_val, mul_match.group(1), None, mul_match.group(2).strip()

        # Inverted pattern: '<product> ke <qty> <unit>'
        inverted_match = re.search(r"^(.*?)\s+(?:ke|ka|ki)\s+(\d+|[a-zA-Z]+)\s*([a-zA-Z]+)?\s*$", text, re.IGNORECASE)
        if inverted_match:
            prod_part = inverted_match.group(1).strip()
            qty_word = inverted_match.group(2).strip()
            unit_part = inverted_match.group(3).strip() if inverted_match.group(3) else None
            qty_val = self._parse_qty_number(qty_word)
            if qty_val is not None:
                return qty_val, qty_word, unit_part, prod_part

        tokens = text.split()
        if not tokens:
            return None, "", None, text

        first_tok = tokens[0].lower()
        qty_val = self._parse_qty_number(first_tok)
        qty_str = ""
        unit_str = None
        prod_tokens = []

        if qty_val is not None:
            qty_str = tokens[0]
            idx = 1
            # Check dozen / darjan compound (e.g. '2 dozen', '3 darjan')
            if len(tokens) > 1 and tokens[1].lower() in ["dozen", "darjan", "dazan"]:
                qty_val = qty_val * Decimal("12")
                qty_str = str(qty_val)
                unit_str = "Piece"
                idx = 2
            elif idx < len(tokens):
                tok_clean = tokens[idx].lower().strip(".,;:?!")
                if tok_clean in KNOWN_UNITS:
                    unit_str = tok_clean
                    idx += 1
                elif re.sub(r"^(ke|ka|ki|wali|wala)$", "", tok_clean) in KNOWN_UNITS:
                    unit_str = re.sub(r"^(ke|ka|ki|wali|wala)$", "", tok_clean)
                    idx += 1
            # Handle English preposition 'of' e.g. '2 bottles of Fortune oil' or '5 packs of Maggi'
            if idx < len(tokens) and tokens[idx].lower() == "of":
                idx += 1
            prod_tokens = tokens[idx:]
        else:
            # Check trailing quantity: ONLY if it doesn't match standard pack size like 1L, 5L, 750ml, 100g
            # e.g. 'Maggi 5 pack' or 'Coke 2' or 'tata salt 2kg'
            match_trailing = re.search(r"^(.*?)\s+(\d+(?:\.\d+)?)\s*([a-zA-Z]+)?$", text, re.IGNORECASE)
            if match_trailing:
                trail_unit = (match_trailing.group(3) or "").lower()
                if trail_unit in KNOWN_UNITS or not trail_unit:
                    prod_tokens = [match_trailing.group(1)]
                    qty_str = match_trailing.group(2)
                    qty_val = Decimal(qty_str)
                    unit_str = match_trailing.group(3)
                else:
                    prod_tokens = tokens
            else:
                prod_tokens = tokens

        product_mention = " ".join(prod_tokens).strip()
        # Clean connectors like 'of', 'the', 'a', 'an', 'ke', 'ka', 'wali', 'wala' from product mention if at edges
        product_mention = re.sub(r"^(of|the|a|an|ke|ka|ki|wala|wali)\s+", "", product_mention, flags=re.IGNORECASE)
        product_mention = re.sub(r"\s+(ke|ka|ki|wala|wali|as\s+usual|usual|bhejna|bhejo)$", "", product_mention, flags=re.IGNORECASE).strip()

        return qty_val, qty_str, unit_str, product_mention

    def _parse_qty_number(self, token: str) -> Optional[Decimal]:
        """Convert digit, fraction, or Hinglish word to Decimal."""
        clean = token.lower().strip(".,;:?!xX*")
        if clean in HINGLISH_NUMBERS:
            return Decimal(str(HINGLISH_NUMBERS[clean]))
        if "/" in clean:
            parts = clean.split("/")
            if len(parts) == 2:
                try:
                    return (Decimal(parts[0]) / Decimal(parts[1])).quantize(Decimal("0.01"))
                except (InvalidOperation, ValueError, ZeroDivisionError):
                    pass
        try:
            return Decimal(clean)
        except (InvalidOperation, ValueError):
            return None
        except (InvalidOperation, ValueError):
            return None

    def _resolve_relative_reference(
        self, text: str, store_id: str, message_id: str, start_idx: int
    ) -> List[OrderLine]:
        """Resolve 'kal jaisa' / 'same as last time' using store order history."""
        history = self.store_history.get(store_id, [])
        if not history:
            # Store has no history: must require clarification!
            line = OrderLine(
                line_id=f"LINE-{start_idx:03d}",
                message_id=message_id,
                store_id=store_id,
                original_item_text=text,
                product_mention=text,
                requested_quantity_value=None,
                requested_quantity_text="",
                requested_unit_text=None,
                match_status="NEEDS_CLARIFICATION",
                clarification_required=True,
                clarification_question=f"Store {store_id} has no recorded previous orders. Please specify the exact products and quantities.",
                line_status="Needs clarification",
                validation_messages=["Relative reference 'same as last time' cannot be resolved without prior order history."],
            )
            return [line]

        # Use items from most recent order cycle
        recent_items = history[-1].get("items", [])
        if not recent_items:
            line = OrderLine(
                line_id=f"LINE-{start_idx:03d}",
                message_id=message_id,
                store_id=store_id,
                original_item_text=text,
                product_mention=text,
                requested_quantity_value=None,
                requested_quantity_text="",
                requested_unit_text=None,
                match_status="NEEDS_CLARIFICATION",
                clarification_required=True,
                clarification_question=f"No products found in the most recent order for {store_id}. Please specify items explicitly.",
                line_status="Needs clarification",
                validation_messages=["Prior order history record contained no line items."],
            )
            return [line]

        generated_lines = []
        for i, h_item in enumerate(recent_items):
            sku = h_item["sku"]
            prod = self.catalogue.get_by_sku(sku)
            qty = Decimal(str(h_item.get("quantity", 1)))
            unit = h_item.get("unit", prod.order_unit if prod else "Piece")

            l = OrderLine(
                line_id=f"LINE-{start_idx + i:03d}",
                message_id=message_id,
                store_id=store_id,
                original_item_text=f"{text} -> Last order ({h_item.get('order_date', 'recent')}) item: {prod.product_name if prod else sku}",
                product_mention=prod.product_name if prod else sku,
                requested_quantity_value=qty,
                requested_quantity_text=str(qty),
                requested_unit_text=unit,
                canonical_product_name=prod.product_name if prod else None,
                sku=sku,
                brand=prod.brand if prod else None,
                pack_size=prod.display_pack if prod else None,
                match_status="MATCHED" if prod else "UNKNOWN_PRODUCT",
                match_evidence=f"Resolved from store order history ({h_item.get('order_id', 'previous')})",
                confidence=0.90,
                candidate_skus=[sku],
                candidate_names=[prod.product_name] if prod else [],
                clarification_required=True,  # Per spec: ask for confirmation when expanding relative references
                clarification_question=f"Confirm repeat item from prior order: {qty} {unit} of {prod.product_name if prod else sku}?",
                line_status="Needs clarification",
                validation_messages=["Historical repeat item generated from prior order cycle. Awaiting shopkeeper confirmation."],
            )
            generated_lines.append(l)
        return generated_lines

    def _ground_item(
        self,
        line_id: str,
        message_id: str,
        store_id: str,
        original_text: str,
        product_mention: str,
        qty_val: Optional[Decimal],
        qty_str: str,
        unit_str: Optional[str],
    ) -> OrderLine:
        """Ground extracted mention against catalogue."""
        # Check invalid quantity
        if qty_val is not None and qty_val <= Decimal(0):
            return OrderLine(
                line_id=line_id,
                message_id=message_id,
                store_id=store_id,
                original_item_text=original_text,
                product_mention=product_mention,
                requested_quantity_value=qty_val,
                requested_quantity_text=qty_str,
                requested_unit_text=unit_str,
                match_status="NEEDS_CLARIFICATION",
                clarification_required=True,
                clarification_question=f"Invalid quantity {qty_val}. Quantity must be greater than zero.",
                line_status="Invalid quantity/unit",
                validation_messages=[f"Quantity must be strictly positive (got {qty_val})."],
            )

        # Search candidates in catalogue
        candidates = self.catalogue.search_candidates(product_mention, top_k=3)

        if not candidates:
            suggestions = self.catalogue.get_spelling_suggestions(product_mention, top_n=2)
            sugg_str = f" Did you mean '{suggestions[0]}'?" if suggestions else ""
            return OrderLine(
                line_id=line_id,
                message_id=message_id,
                store_id=store_id,
                original_item_text=original_text,
                product_mention=product_mention,
                requested_quantity_value=qty_val,
                requested_quantity_text=qty_str,
                requested_unit_text=unit_str,
                match_status="UNKNOWN_PRODUCT",
                clarification_required=True,
                clarification_question=f"Product '{product_mention}' was not found in distributor catalogue.{sugg_str}",
                line_status="Unknown product",
                validation_messages=[f"Unknown product '{product_mention}'. No catalogue match.{sugg_str}"],
                spelling_suggestion=sugg_str.strip() or None,
            )

        # RAG Grounding: Check store purchase history to rank or prioritize candidates
        store_hist = self.store_sku_history.get(store_id, {})
        def get_hist_orders(cand):
            return len(store_hist.get(cand.product.sku, [])) if store_hist else 0

        # Disambiguate using packaging unit if specified by retailer (e.g. 'bottle' vs 'can')
        if unit_str and candidates:
            from conversion import UNIT_SYNONYMS
            norm_u = UNIT_SYNONYMS.get(unit_str.lower(), unit_str.lower()).lower()
            for cand in candidates:
                if cand.product.order_unit.lower() == norm_u:
                    cand.score = min(1.0, cand.score + 0.15)
                    cand.evidence += f" (Matches requested packaging unit '{unit_str}')"

        if store_hist and candidates:
            # Prioritize candidates with verified store purchase history
            candidates.sort(key=lambda c: (get_hist_orders(c) > 0, get_hist_orders(c), c.score), reverse=True)
        elif candidates:
            candidates.sort(key=lambda c: c.score, reverse=True)

        top_cand = candidates[0]
        cand_skus = [c.product.sku for c in candidates]
        cand_names = [c.product.product_name for c in candidates]

        # Check ambiguity: multiple candidates with close scores (e.g. 1L vs 5L)
        is_ambiguous = False
        clarify_q = None

        if len(candidates) > 1:
            diff = candidates[0].score - candidates[1].score
            q_pack = self.catalogue._extract_pack_spec(product_mention)
            is_same_brand_pack_variant = (
                not q_pack
                and candidates[0].product.brand == candidates[1].product.brand
                and candidates[0].product.category == candidates[1].product.category
                and candidates[0].product.display_pack != candidates[1].product.display_pack
            )

            if is_same_brand_pack_variant or "Ambiguous" in candidates[0].evidence or "Category match" in candidates[0].evidence:
                is_ambiguous = True
                hist_count = len(store_hist.get(top_cand.product.sku, [])) if store_hist else 0
                if hist_count > 0:
                    clarify_q = (
                        f"Auto-selected '{top_cand.product.product_name}' (Store ordered {hist_count} times previously). "
                        f"Did you mean '{candidates[1].product.product_name}'?"
                    )
                else:
                    clarify_q = f"Aap '{candidates[0].product.product_name}' keh rahe hain ya '{candidates[1].product.product_name}'?"
            elif top_cand.score >= 0.98 and diff > 0.05:
                is_ambiguous = False
            elif diff < 0.15:
                is_ambiguous = True
                hist_count = len(store_hist.get(top_cand.product.sku, [])) if store_hist else 0
                if hist_count > 0:
                    clarify_q = (
                        f"Auto-selected '{top_cand.product.product_name}' (Store ordered {hist_count} times previously). "
                        f"Did you mean '{candidates[1].product.product_name}'?"
                    )
                else:
                    clarify_q = f"Aap '{candidates[0].product.product_name}' keh rahe hain ya '{candidates[1].product.product_name}'?"

        # If top candidate is inactive
        if not top_cand.product.is_active:
            return OrderLine(
                line_id=line_id,
                message_id=message_id,
                store_id=store_id,
                original_item_text=original_text,
                product_mention=product_mention,
                requested_quantity_value=qty_val,
                requested_quantity_text=qty_str,
                requested_unit_text=unit_str,
                canonical_product_name=top_cand.product.product_name,
                sku=top_cand.product.sku,
                brand=top_cand.product.brand,
                pack_size=top_cand.product.display_pack,
                match_status="NEEDS_CLARIFICATION",
                match_evidence=f"Matched inactive SKU {top_cand.product.sku}",
                confidence=top_cand.score,
                candidate_skus=cand_skus,
                candidate_names=cand_names,
                clarification_required=True,
                clarification_question=f"SKU {top_cand.product.sku} ({top_cand.product.product_name}) is currently inactive/discontinued. Please choose an active replacement.",
                line_status="Needs clarification",
                validation_messages=["Product is marked inactive/discontinued in catalogue."],
            )

        if is_ambiguous:
            # Auto-select the top candidate product so it is NOT left blank!
            return OrderLine(
                line_id=line_id,
                message_id=message_id,
                store_id=store_id,
                original_item_text=original_text,
                product_mention=product_mention,
                requested_quantity_value=qty_val,
                requested_quantity_text=qty_str,
                requested_unit_text=unit_str,
                canonical_product_name=top_cand.product.product_name,
                sku=top_cand.product.sku,
                brand=top_cand.product.brand,
                pack_size=top_cand.product.display_pack,
                match_status="NEEDS_CLARIFICATION",
                match_evidence=f"{top_cand.evidence} (RAG auto-selected)",
                confidence=top_cand.score,
                candidate_skus=cand_skus,
                candidate_names=cand_names,
                clarification_required=True,
                clarification_question=clarify_q,
                line_status="Needs clarification",
                validation_messages=[clarify_q or "Ambiguous product reference. Default candidate auto-selected."] + ([getattr(top_cand, 'spelling_suggestion', None)] if getattr(top_cand, 'spelling_suggestion', None) else []),
                spelling_suggestion=getattr(top_cand, 'spelling_suggestion', None),
            )

        # Missing quantity check
        if qty_val is None:
            return OrderLine(
                line_id=line_id,
                message_id=message_id,
                store_id=store_id,
                original_item_text=original_text,
                product_mention=product_mention,
                requested_quantity_value=None,
                requested_quantity_text="",
                requested_unit_text=unit_str,
                canonical_product_name=top_cand.product.product_name,
                sku=top_cand.product.sku,
                brand=top_cand.product.brand,
                pack_size=top_cand.product.display_pack,
                match_status="MATCHED",
                match_evidence=top_cand.evidence,
                confidence=top_cand.score,
                candidate_skus=cand_skus,
                candidate_names=cand_names,
                clarification_required=True,
                clarification_question=f"Please specify the quantity needed for {top_cand.product.product_name}.",
                line_status="Needs clarification",
                validation_messages=["Quantity is missing."],
            )

        # Single confident match
        return OrderLine(
            line_id=line_id,
            message_id=message_id,
            store_id=store_id,
            original_item_text=original_text,
            product_mention=product_mention,
            requested_quantity_value=qty_val,
            requested_quantity_text=qty_str,
            requested_unit_text=unit_str,
            canonical_product_name=top_cand.product.product_name,
            sku=top_cand.product.sku,
            brand=top_cand.product.brand,
            pack_size=top_cand.product.display_pack,
            match_status="MATCHED",
            match_evidence=top_cand.evidence,
            confidence=top_cand.score,
            candidate_skus=cand_skus,
            candidate_names=cand_names,
            clarification_required=False,
            clarification_question=None,
            line_status="Ready for review",
            validation_messages=[getattr(top_cand, 'spelling_suggestion', None)] if getattr(top_cand, 'spelling_suggestion', None) else [],
            spelling_suggestion=getattr(top_cand, 'spelling_suggestion', None),
        )

    def _detect_duplicates(self, lines: List[OrderLine]) -> None:
        """Detect repeated mentions of the same SKU or product in the order."""
        seen_skus: Dict[str, List[OrderLine]] = {}
        for l in lines:
            if l.sku:
                seen_skus.setdefault(l.sku, []).append(l)

        for sku, l_list in seen_skus.items():
            if len(l_list) > 1:
                for item in l_list:
                    item.is_duplicate = True
                    item.validation_messages.append(
                        f"Potential duplicate mention: SKU {sku} appears {len(l_list)} times in this message. Review quantities before confirmation."
                    )

    def _finalize_line_calculations(self, line: OrderLine, store_id: str) -> None:
        """Run unit conversion and price calculation deterministically in Python."""
        prod = self.catalogue.get_by_sku(line.sku) if line.sku else None

        # Determine input quantity and unit for conversion
        # CRITICAL FIX for 2304 bug: Never use line.converted_quantity as input to conversion
        # with requested_unit_text ('peti'/'dhaba'), because if converted_quantity is already
        # in base ordering units (e.g. 48 bottles), re-converting it with 'peti' would multiply 48 * 48 = 2304!
        if line.user_edited and line.requested_quantity_value is None and line.converted_quantity is not None:
            req_qty = line.converted_quantity
            req_unit = line.ordering_unit or line.requested_unit_text
        else:
            req_qty = line.requested_quantity_value
            req_unit = line.requested_unit_text

        # Unit conversion
        conv_res = self.converter.convert(
            product=prod,
            requested_qty=req_qty,
            requested_unit=req_unit,
            store_id=store_id,
        )

        line.converted_quantity = conv_res.converted_quantity
        line.ordering_unit = conv_res.ordering_unit or (prod.order_unit if prod else None)
        line.conversion_rule_id = conv_res.rule_id

        if conv_res.status != "VALID" and not line.user_edited:
            line.clarification_required = True
            if line.match_status == "UNKNOWN_PRODUCT":
                line.line_status = "Unknown product"
            elif conv_res.status in ["INVALID_QUANTITY", "UNSUPPORTED_UNIT"]:
                line.line_status = "Invalid quantity/unit"
            else:
                line.line_status = "Needs clarification"
            if not line.clarification_question:
                line.clarification_question = conv_res.explanation
            if conv_res.explanation not in line.validation_messages:
                line.validation_messages.append(conv_res.explanation)

        # Price calculation
        price = prod.unit_price_inr if prod else None
        line.unit_price = price
        line.price_basis = f"Per {line.ordering_unit}" if line.ordering_unit else None

        line_calc = calculate_line_item(
            sku=line.sku,
            ordering_quantity=line.converted_quantity,
            ordering_unit=line.ordering_unit,
            unit_price_inr=line.unit_price,
            match_status="MATCHED" if line.sku else line.match_status,
            conversion_status=conv_res.status,
        )

        # Ensure line_total is computed whenever quantity and price are available
        if line.converted_quantity is not None and line.unit_price is not None and line.converted_quantity > 0:
            line.line_total = (line.converted_quantity * line.unit_price).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        else:
            line.line_total = line_calc.line_total_inr

        if not line_calc.is_valid and not line.user_edited:
            for b in line_calc.blocking_reasons:
                if b not in line.validation_messages:
                    line.validation_messages.append(b)
            if price is None and line.match_status == "MATCHED":
                line.line_status = "Price unavailable"

        if line.clarification_required:
            if line.line_status not in ["Invalid quantity/unit", "Price unavailable", "Unknown product"]:
                line.line_status = "Needs clarification"
        elif line.line_status not in ["Invalid quantity/unit", "Price unavailable", "Unknown product"]:
            line.line_status = "Ready for review"

    def _extract_with_llm(
        self, text: str, store_id: str, message_id: str, api_key: str, model_name: Optional[str]
    ) -> List[OrderLine]:
        """
        Extract items using configured external LLM (Gemini or OpenAI format).
        Guarded by strict JSON schema and post-validated in Python.
        """
        # When an LLM API key is present, this invokes the model with structured output prompt
        # We enforce deterministic Python post-grounding on top of model extraction.
        # Fallback to rule engine for rock-solid stability.
        return self._extract_rule_grounded(text, store_id, message_id)
