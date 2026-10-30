"""
validation.py
Input Sanitization, Security / Prompt-Injection Guardrails, Order Data Model, and Audit Logging.
Enforces untrusted input boundaries and auditability.
"""

from __future__ import annotations
import html
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional


MAX_MESSAGE_LENGTH = 5000

# Known prompt injection signatures
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous\s+)?(rules|instructions|prompts)",
    r"system\s*prompt",
    r"reveal\s+(api\s*key|secret|password|credential)",
    r"add\s+free\s+(items|products|goods)",
    r"set\s+price\s+to\s+0",
    r"override\s+validation",
    r"bypass\s+confirmation",
    r"drop\s+table",
    r"delete\s+from",
]


@dataclass
class InputValidationResult:
    is_valid: bool
    sanitized_text: str
    error_message: Optional[str] = None
    has_injection_risk: bool = False
    injection_details: Optional[str] = None


def validate_and_sanitize_input(raw_text: str) -> InputValidationResult:
    """
    Validate and sanitize user order text.
    Rejects empty or oversized messages and detects potential prompt injection.
    """
    if raw_text is None or not raw_text.strip():
        return InputValidationResult(
            is_valid=False,
            sanitized_text="",
            error_message="Order message cannot be empty.",
        )

    text = raw_text.strip()
    if len(text) > MAX_MESSAGE_LENGTH:
        return InputValidationResult(
            is_valid=False,
            sanitized_text=text[:MAX_MESSAGE_LENGTH],
            error_message=f"Order message exceeds maximum length of {MAX_MESSAGE_LENGTH} characters.",
        )

    # Detect prompt injection attempts
    injection_risk = False
    injection_detail = None
    for pattern in PROMPT_INJECTION_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            injection_risk = True
            injection_detail = f"Suspicious phrase detected matching policy guardrail: '{match.group(0)}'"
            break

    # HTML escape for safe rendering in UI
    safe_text = html.escape(text)

    return InputValidationResult(
        is_valid=True,
        sanitized_text=text,  # Keep text clean for parsing
        has_injection_risk=injection_risk,
        injection_details=injection_detail,
    )


@dataclass
class OrderLine:
    line_id: str
    message_id: str
    store_id: str
    original_item_text: str
    product_mention: str
    requested_quantity_value: Optional[Decimal]
    requested_quantity_text: str
    requested_unit_text: Optional[str]
    canonical_product_name: Optional[str] = None
    sku: Optional[str] = None
    brand: Optional[str] = None
    pack_size: Optional[str] = None
    match_status: str = "NEEDS_CLARIFICATION"  # MATCHED | NEEDS_CLARIFICATION | UNKNOWN_PRODUCT
    match_evidence: Optional[str] = None
    confidence: float = 0.0
    candidate_skus: List[str] = field(default_factory=list)
    candidate_names: List[str] = field(default_factory=list)
    conversion_rule_id: Optional[str] = None
    converted_quantity: Optional[Decimal] = None
    ordering_unit: Optional[str] = None
    unit_price: Optional[Decimal] = None
    price_basis: Optional[str] = None
    line_total: Optional[Decimal] = None
    clarification_required: bool = False
    clarification_question: Optional[str] = None
    is_duplicate: bool = False
    line_status: str = "Ready for review"
    # Statuses: "Needs clarification", "Unknown product", "Invalid quantity/unit",
    # "Price unavailable", "Ready for review", "Confirmed by shopkeeper",
    # "Simulated submission complete", "Removed by shopkeeper"
    validation_messages: List[str] = field(default_factory=list)
    user_edited: bool = False
    spelling_suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert Decimals to string/float for JSON compatibility
        if self.requested_quantity_value is not None:
            d["requested_quantity_value"] = str(self.requested_quantity_value)
        if self.converted_quantity is not None:
            d["converted_quantity"] = str(self.converted_quantity)
        if self.unit_price is not None:
            d["unit_price"] = str(self.unit_price)
        if self.line_total is not None:
            d["line_total"] = str(self.line_total)
        return d


@dataclass
class AuditEvent:
    event_id: str
    timestamp: str
    event_type: str
    message_id: Optional[str]
    store_id: Optional[str]
    status: str
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuditLogger:
    """
    Session and in-memory auditable event logger.
    Does not log credentials, tokens, or unnecessary personal info.
    """
    def __init__(self):
        self.events: List[AuditEvent] = []

    def log(
        self,
        event_type: str,
        status: str,
        details: str,
        message_id: Optional[str] = None,
        store_id: Optional[str] = None,
    ) -> AuditEvent:
        event = AuditEvent(
            event_id=f"AUD-{uuid.uuid4().hex[:8].upper()}",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            event_type=event_type,
            message_id=message_id,
            store_id=store_id,
            status=status,
            details=details,
        )
        self.events.append(event)
        return event

    def get_events(self) -> List[AuditEvent]:
        return list(reversed(self.events))

    def clear(self) -> None:
        self.events.clear()

