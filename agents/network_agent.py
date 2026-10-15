from __future__ import annotations

import base64
import math
import re

from .base import AgentBase, EvidenceItem, EvidenceType

URL_PATTERN = re.compile(
    r'https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+'
)
IP_PATTERN = re.compile(
    r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
)
DOMAIN_PATTERN = re.compile(
    r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b'
)
BASE64_PATTERN = re.compile(
    r'[A-Za-z0-9+/]{20,}={0,2}'
)

KNOWN_BENIGN_DOMAINS = {
    "google.com", "googleapis.com", "android.com", "gstatic.com",
    "googleusercontent.com", "facebook.com", "crashlytics.com",
    "firebase.io", "firebaseio.com", "firebase.google.com",
    "apple.com", "github.com", "githubusercontent.com",
    "xml.org", "w3.org", "apache.org", "xmlpull.org",
    "schemas.android.com",
}


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    length = len(s)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


class NetworkAgent(AgentBase):
    name = "network"

    def analyze(self, apk_path: str, dalvik_objects=None) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        strings = self._extract_strings(apk_path, dalvik_objects)

        evidence.extend(self._find_urls(strings))
        evidence.extend(self._find_ips(strings))
        evidence.extend(self._find_encoded_urls(strings))

        return evidence

    def _extract_strings(self, apk_path: str, dalvik_objects=None) -> list[str]:
        if dalvik_objects is None:
            from androguard.misc import AnalyzeAPK
            _, dalvik_objects, _ = AnalyzeAPK(apk_path)

        strings = set()
        for dex in dalvik_objects:
            try:
                for s in dex.get_strings():
                    if len(s) > 5:
                        strings.add(s)
            except Exception:
                continue
        return list(strings)

    def _find_urls(self, strings: list[str]) -> list[EvidenceItem]:
        items = []
        seen = set()

        for s in strings:
            for match in URL_PATTERN.finditer(s):
                url = match.group()
                if url in seen:
                    continue
                seen.add(url)

                is_benign = any(d in url for d in KNOWN_BENIGN_DOMAINS)
                entropy = _entropy(url)

                items.append(EvidenceItem(
                    evidence_type=EvidenceType.NETWORK_INDICATOR,
                    source_agent=self.name,
                    entity=url,
                    attributes={
                        "indicator_type": "url",
                        "entropy": round(entropy, 2),
                        "is_known_benign": is_benign,
                        "is_https": url.startswith("https"),
                        "length": len(url),
                    },
                    confidence=0.5 if is_benign else 0.9,
                    description=f"URL: {url[:80]}{'...' if len(url) > 80 else ''}",
                ))

        return items

    def _find_ips(self, strings: list[str]) -> list[EvidenceItem]:
        items = []
        seen = set()

        for s in strings:
            for match in IP_PATTERN.finditer(s):
                ip = match.group()
                if ip in seen:
                    continue
                seen.add(ip)

                octets = [int(o) for o in ip.split(".")]
                if not all(0 <= o <= 255 for o in octets):
                    continue
                if ip.startswith("10.") or ip.startswith("192.168.") or ip.startswith("127."):
                    continue
                if ip in ("0.0.0.0", "255.255.255.255"):
                    continue

                port = None
                port_match = re.search(re.escape(ip) + r":(\d+)", s)
                if port_match:
                    port = int(port_match.group(1))

                items.append(EvidenceItem(
                    evidence_type=EvidenceType.NETWORK_INDICATOR,
                    source_agent=self.name,
                    entity=ip,
                    attributes={
                        "indicator_type": "ip",
                        "port": port,
                        "is_raw_ip": True,
                        "non_standard_port": port is not None and port not in (80, 443, 8080, 8443),
                    },
                    confidence=0.85,
                    description=f"Raw IP: {ip}" + (f":{port}" if port else ""),
                ))

        return items

    def _find_encoded_urls(self, strings: list[str]) -> list[EvidenceItem]:
        items = []
        seen = set()

        for s in strings:
            for match in BASE64_PATTERN.finditer(s):
                b64 = match.group()
                try:
                    decoded = base64.b64decode(b64, validate=True).decode("utf-8", errors="ignore")
                except Exception:
                    continue

                urls_in_decoded = URL_PATTERN.findall(decoded)
                for url in urls_in_decoded:
                    if url in seen:
                        continue
                    seen.add(url)
                    items.append(EvidenceItem(
                        evidence_type=EvidenceType.NETWORK_INDICATOR,
                        source_agent=self.name,
                        entity=url,
                        attributes={
                            "indicator_type": "encoded_url",
                            "encoding": "base64",
                            "is_known_benign": any(d in url for d in KNOWN_BENIGN_DOMAINS),
                        },
                        confidence=0.95,
                        description=f"Base64-encoded URL: {url[:80]}",
                    ))

        return items
