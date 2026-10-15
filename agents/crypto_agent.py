from __future__ import annotations

import re

from .base import AgentBase, EvidenceItem, EvidenceType

CRYPTO_APIS = {
    "Ljavax/crypto/Cipher;->getInstance": "cipher",
    "Ljavax/crypto/Cipher;->init": "cipher",
    "Ljavax/crypto/Cipher;->doFinal": "cipher",
    "Ljavax/crypto/spec/SecretKeySpec;-><init>": "key_spec",
    "Ljavax/crypto/spec/IvParameterSpec;-><init>": "iv_spec",
    "Ljavax/crypto/KeyGenerator;->getInstance": "keygen",
    "Ljavax/crypto/Mac;->getInstance": "mac",
    "Ljava/security/MessageDigest;->getInstance": "digest",
    "Ljava/security/MessageDigest;->digest": "digest",
    "Ljava/security/KeyStore;->getInstance": "keystore",
    "Ljava/security/KeyPairGenerator;->getInstance": "keypair",
    "Ljava/security/Signature;->getInstance": "signature",
    "Landroid/security/keystore/KeyGenParameterSpec;": "android_keystore",
    "Ljavax/net/ssl/SSLContext;->getInstance": "ssl",
    "Ljavax/net/ssl/TrustManager;": "trust_manager",
    "Ljavax/net/ssl/X509TrustManager;": "trust_manager",
    "Ljavax/net/ssl/HttpsURLConnection;->setDefaultHostnameVerifier": "hostname_verifier",
}

WEAK_ALGORITHMS = {"DES", "DESede", "RC4", "MD5", "SHA1", "RC2", "Blowfish"}
STRONG_ALGORITHMS = {"AES", "RSA", "SHA-256", "SHA-512", "ECDSA", "ChaCha20"}

HARDCODED_KEY_PATTERN = re.compile(
    r'(?:key|secret|password|passwd|token|apikey)\s*[:=]\s*["\']([^"\']{8,})["\']',
    re.IGNORECASE,
)


class CryptoAgent(AgentBase):
    name = "crypto"

    def analyze(self, apk_path: str, dalvik_objects=None, analysis=None) -> list[EvidenceItem]:
        if dalvik_objects is None or analysis is None:
            from androguard.misc import AnalyzeAPK
            _, dalvik_objects, analysis = AnalyzeAPK(apk_path)

        evidence: list[EvidenceItem] = []

        evidence.extend(self._find_crypto_api_usage(analysis))
        evidence.extend(self._find_hardcoded_keys(dalvik_objects))
        evidence.extend(self._check_ssl_pinning(analysis))

        return evidence

    def _find_crypto_api_usage(self, analysis) -> list[EvidenceItem]:
        items = []
        seen = set()

        for method in analysis.get_methods():
            method_name = str(method.get_method())
            for api_pattern, category in CRYPTO_APIS.items():
                if api_pattern in method_name and method_name not in seen:
                    seen.add(method_name)

                    callers = []
                    try:
                        for _, caller, _ in method.get_xref_from():
                            callers.append(str(caller))
                    except Exception:
                        pass

                    items.append(EvidenceItem(
                        evidence_type=EvidenceType.CRYPTO_OPERATION,
                        source_agent=self.name,
                        entity=method_name,
                        attributes={
                            "crypto_category": category,
                            "caller_count": len(callers),
                        },
                        related_entities=callers[:10],
                        description=f"Crypto API ({category}): {method_name}",
                    ))

        return items

    def _find_hardcoded_keys(self, dalvik_objects) -> list[EvidenceItem]:
        items = []
        seen_keys = set()

        for dex in dalvik_objects:
            try:
                for s in dex.get_strings():
                    for match in HARDCODED_KEY_PATTERN.finditer(s):
                        key_val = match.group(1)
                        if key_val in seen_keys:
                            continue
                        seen_keys.add(key_val)

                        items.append(EvidenceItem(
                            evidence_type=EvidenceType.CRYPTO_OPERATION,
                            source_agent=self.name,
                            entity=f"hardcoded_key:{key_val[:16]}...",
                            attributes={
                                "crypto_category": "hardcoded_key",
                                "key_length": len(key_val),
                                "is_hardcoded": True,
                            },
                            confidence=0.7,
                            description=f"Potential hardcoded key/secret ({len(key_val)} chars)",
                        ))
            except Exception:
                continue

        return items

    def _check_ssl_pinning(self, analysis) -> list[EvidenceItem]:
        items = []
        has_custom_trust_manager = False
        has_hostname_bypass = False

        for method in analysis.get_methods():
            method_name = str(method.get_method())
            if "X509TrustManager" in method_name and "checkServerTrusted" in method_name:
                has_custom_trust_manager = True
            if "setDefaultHostnameVerifier" in method_name or "ALLOW_ALL" in method_name:
                has_hostname_bypass = True

        if has_custom_trust_manager:
            items.append(EvidenceItem(
                evidence_type=EvidenceType.CRYPTO_OPERATION,
                source_agent=self.name,
                entity="custom_trust_manager",
                attributes={
                    "crypto_category": "ssl_bypass",
                    "has_hostname_bypass": has_hostname_bypass,
                    "risk_level": 3,
                },
                confidence=0.75,
                description="Custom TrustManager detected — may bypass SSL certificate validation",
            ))

        return items
