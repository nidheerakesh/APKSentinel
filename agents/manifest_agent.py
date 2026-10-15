from __future__ import annotations

from .base import AgentBase, EvidenceItem, EvidenceType

DANGEROUS_PERMISSIONS = {
    "android.permission.RECEIVE_SMS": 3,
    "android.permission.READ_SMS": 3,
    "android.permission.SEND_SMS": 3,
    "android.permission.READ_PHONE_STATE": 2,
    "android.permission.CALL_PHONE": 2,
    "android.permission.READ_CONTACTS": 2,
    "android.permission.READ_CALL_LOG": 2,
    "android.permission.CAMERA": 2,
    "android.permission.RECORD_AUDIO": 2,
    "android.permission.ACCESS_FINE_LOCATION": 2,
    "android.permission.ACCESS_COARSE_LOCATION": 1,
    "android.permission.INTERNET": 1,
    "android.permission.READ_EXTERNAL_STORAGE": 1,
    "android.permission.WRITE_EXTERNAL_STORAGE": 1,
    "android.permission.SYSTEM_ALERT_WINDOW": 3,
    "android.permission.BIND_ACCESSIBILITY_SERVICE": 3,
    "android.permission.REQUEST_INSTALL_PACKAGES": 3,
    "android.permission.BIND_DEVICE_ADMIN": 3,
    "android.permission.RECEIVE_BOOT_COMPLETED": 1,
    "android.permission.WAKE_LOCK": 1,
    "android.permission.GET_ACCOUNTS": 2,
    "android.permission.USE_CREDENTIALS": 2,
    "android.permission.MANAGE_ACCOUNTS": 2,
    "android.permission.AUTHENTICATE_ACCOUNTS": 2,
    "android.permission.READ_PHONE_NUMBERS": 2,
    "android.permission.FOREGROUND_SERVICE": 1,
    "android.permission.QUERY_ALL_PACKAGES": 2,
    "android.permission.REQUEST_DELETE_PACKAGES": 2,
    "android.permission.BIND_NOTIFICATION_LISTENER_SERVICE": 3,
    "android.permission.WRITE_SETTINGS": 2,
}


class ManifestAgent(AgentBase):
    name = "manifest"

    def analyze(self, apk_path: str) -> list[EvidenceItem]:
        from androguard.core.apk import APK

        apk = APK(apk_path)
        evidence: list[EvidenceItem] = []

        evidence.extend(self._extract_permissions(apk))
        evidence.extend(self._extract_components(apk))
        evidence.extend(self._check_accessibility(apk))

        return evidence

    def _extract_permissions(self, apk) -> list[EvidenceItem]:
        items = []
        permissions = apk.get_permissions()
        for perm in permissions:
            risk = DANGEROUS_PERMISSIONS.get(perm, 0)
            items.append(EvidenceItem(
                evidence_type=EvidenceType.PERMISSION,
                source_agent=self.name,
                entity=perm,
                attributes={
                    "risk_level": risk,
                    "is_dangerous": risk >= 2,
                    "category": self._categorize_permission(perm),
                },
                confidence=1.0,
                description=f"Permission: {perm} (risk: {risk})",
            ))
        return items

    def _extract_components(self, apk) -> list[EvidenceItem]:
        items = []

        for activity in apk.get_activities():
            items.append(EvidenceItem(
                evidence_type=EvidenceType.COMPONENT,
                source_agent=self.name,
                entity=f"activity:{activity}",
                attributes={
                    "component_type": "activity",
                    "is_exported": self._is_exported(apk, activity, "activity"),
                    "has_intent_filter": self._has_intent_filter(apk, activity, "activity"),
                },
                description=f"Activity: {activity}",
            ))

        for service in apk.get_services():
            items.append(EvidenceItem(
                evidence_type=EvidenceType.COMPONENT,
                source_agent=self.name,
                entity=f"service:{service}",
                attributes={
                    "component_type": "service",
                    "is_exported": self._is_exported(apk, service, "service"),
                    "has_intent_filter": self._has_intent_filter(apk, service, "service"),
                },
                description=f"Service: {service}",
            ))

        for receiver in apk.get_receivers():
            items.append(EvidenceItem(
                evidence_type=EvidenceType.COMPONENT,
                source_agent=self.name,
                entity=f"receiver:{receiver}",
                attributes={
                    "component_type": "receiver",
                    "is_exported": self._is_exported(apk, receiver, "receiver"),
                    "has_intent_filter": self._has_intent_filter(apk, receiver, "receiver"),
                },
                description=f"Receiver: {receiver}",
            ))

        return items

    def _check_accessibility(self, apk) -> list[EvidenceItem]:
        items = []
        services = apk.get_services()
        permissions = apk.get_permissions()

        if "android.permission.BIND_ACCESSIBILITY_SERVICE" in permissions:
            items.append(EvidenceItem(
                evidence_type=EvidenceType.PERMISSION,
                source_agent=self.name,
                entity="accessibility_abuse_indicator",
                attributes={
                    "risk_level": 3,
                    "services_count": len(services),
                    "has_sms_perms": any("SMS" in p for p in permissions),
                    "has_overlay": "android.permission.SYSTEM_ALERT_WINDOW" in permissions,
                },
                confidence=0.8,
                description="Accessibility service with high-risk permission combination",
                related_entities=[
                    p for p in permissions
                    if DANGEROUS_PERMISSIONS.get(p, 0) >= 2
                ],
            ))

        return items

    def _is_exported(self, apk, component: str, comp_type: str) -> bool:
        try:
            xml = apk.get_android_manifest_xml()
            ns = "{http://schemas.android.com/apk/res/android}"
            for elem in xml.iter():
                if elem.tag == comp_type and elem.get(f"{ns}name") == component:
                    exported = elem.get(f"{ns}exported")
                    if exported is not None:
                        return exported.lower() == "true"
                    return any(child.tag == "intent-filter" for child in elem)
        except Exception:
            pass
        return False

    def _has_intent_filter(self, apk, component: str, comp_type: str) -> bool:
        try:
            xml = apk.get_android_manifest_xml()
            ns = "{http://schemas.android.com/apk/res/android}"
            for elem in xml.iter():
                if elem.tag == comp_type and elem.get(f"{ns}name") == component:
                    return any(child.tag == "intent-filter" for child in elem)
        except Exception:
            pass
        return False

    @staticmethod
    def _categorize_permission(perm: str) -> str:
        perm_upper = perm.upper()
        if "SMS" in perm_upper:
            return "sms"
        if "PHONE" in perm_upper or "CALL" in perm_upper:
            return "telephony"
        if "CONTACT" in perm_upper:
            return "contacts"
        if "CAMERA" in perm_upper:
            return "camera"
        if "LOCATION" in perm_upper:
            return "location"
        if "STORAGE" in perm_upper:
            return "storage"
        if "ACCESSIBILITY" in perm_upper:
            return "accessibility"
        if "INTERNET" in perm_upper or "NETWORK" in perm_upper:
            return "network"
        if "INSTALL" in perm_upper or "PACKAGE" in perm_upper:
            return "install"
        return "other"
