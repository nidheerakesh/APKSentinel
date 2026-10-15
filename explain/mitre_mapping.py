"""
MITRE ATT&CK for Android mapping.
Maps extracted features (permissions, API calls) to ATT&CK tactics and techniques.
Based on: Tanha & Kafaie, IEEE Access 2025 (Table 4, Figure 3).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MitreMapping:
    feature: str
    feature_type: str
    technique: str
    technique_id: str
    tactic: str


PERMISSION_MAPPINGS: dict[str, list[tuple[str, str, str]]] = {
    "ACCESS_COARSE_LOCATION": [("Location Tracking", "T1430", "Collection, Discovery")],
    "ACCESS_FINE_LOCATION": [("Location Tracking", "T1430", "Collection, Discovery")],
    "READ_CALL_LOG": [("Protected User Data", "T1636", "Collection")],
    "READ_CONTACTS": [("Protected User Data", "T1636", "Collection")],
    "READ_PHONE_STATE": [("Protected User Data", "T1636", "Collection")],
    "READ_SMS": [("Protected User Data", "T1636", "Collection")],
    "SEND_SMS": [("SMS Control", "T1582", "Impact")],
    "CALL_PHONE": [("Call Control", "T1616", "Collection, Command and Control, Impact")],
    "RECORD_AUDIO": [("Audio Capture", "T1429", "Collection")],
    "RECEIVE_BOOT_COMPLETED": [("Boot or Logon Initialization Scripts", "T1398", "Persistence")],
    "CAMERA": [("Video Capture", "T1512", "Collection")],
    "GET_ACCOUNTS": [("Stored Application Data", "T1409", "Collection")],
    "READ_EXTERNAL_STORAGE": [("Data from Local System", "T1533", "Collection")],
    "WRITE_EXTERNAL_STORAGE": [("Data from Local System", "T1533", "Collection")],
    "DISABLE_KEYGUARD": [("Lockscreen Bypass", "T1461", "Initial Access")],
    "ACCESS_WIFI_STATE": [("System Network Configuration Discovery", "T1422", "Discovery")],
    "ACCESS_NETWORK_STATE": [("System Network Configuration Discovery", "T1422", "Discovery")],
    "INTERNET": [("Standard Application Layer Protocol", "T1437", "Command and Control")],
    "INSTALL_PACKAGES": [("Input Injection", "T1516", "Defense Evasion")],
    "REQUEST_INSTALL_PACKAGES": [("Input Injection", "T1516", "Defense Evasion")],
    "SYSTEM_ALERT_WINDOW": [("Input Capture (GUI)", "T1417", "Collection")],
    "RECEIVE_SMS": [("Protected User Data", "T1636", "Collection")],
    "READ_PHONE_NUMBERS": [("Protected User Data", "T1636", "Collection")],
    "BIND_ACCESSIBILITY_SERVICE": [("Abuse Accessibility Features", "T1453", "Privilege Escalation")],
    "WAKE_LOCK": [("Foreground Persistence", "T1541", "Persistence")],
    "CHANGE_WIFI_STATE": [("System Network Configuration Discovery", "T1422", "Discovery")],
    "PROCESS_OUTGOING_CALLS": [("Call Control", "T1616", "Collection")],
    "WRITE_SETTINGS": [("Modify System Settings", "T1508", "Defense Evasion")],
}

API_MAPPINGS: dict[str, list[tuple[str, str, str]]] = {
    "getLastKnownLocation": [("Location Tracking", "T1430", "Collection, Discovery")],
    "requestLocationUpdates": [("Location Tracking", "T1430", "Collection, Discovery")],
    "getDeviceId": [("System Network Configuration Discovery", "T1422", "Discovery")],
    "getSubscriberId": [("System Network Configuration Discovery", "T1422", "Discovery")],
    "getSimSerialNumber": [("System Network Configuration Discovery", "T1422", "Discovery")],
    "getPhoneType": [("System Network Configuration Discovery", "T1422", "Discovery")],
    "getSimState": [("System Network Configuration Discovery", "T1422", "Discovery")],
    "getLine1Number": [("System Network Configuration Discovery", "T1422", "Discovery")],
    "sendTextMessage": [("SMS Control", "T1582", "Impact")],
    "sendDataMessage": [("SMS Control", "T1582", "Impact")],
    "sendMultipartTextMessage": [("SMS Control", "T1582", "Impact")],
    "exec": [("Command and Scripting Interpreter", "T1623", "Execution")],
    "getRuntime": [("Command and Scripting Interpreter", "T1623", "Execution")],
    "loadLibrary": [("Command and Scripting Interpreter", "T1623", "Execution")],
    "loadClass": [("Obfuscated Files or Information", "T1406", "Defense Evasion")],
    "DexClassLoader": [("Obfuscated Files or Information", "T1406", "Defense Evasion")],
    "getExternalStorageDirectory": [("Data from Local System", "T1533", "Collection")],
    "openConnection": [("Standard Application Layer Protocol", "T1437", "Command and Control")],
    "HttpURLConnection": [("Standard Application Layer Protocol", "T1437", "Command and Control")],
    "getContentResolver": [("Stored Application Data", "T1409", "Collection")],
    "abortBroadcast": [("Broadcast Receivers", "T1402", "Persistence")],
    "setComponentEnabledSetting": [("Hide Artifacts", "T1628", "Defense Evasion")],
    "getPackageManager": [("Software Discovery", "T1418", "Discovery")],
    "getInstalledPackages": [("Software Discovery", "T1418", "Discovery")],
    "getInstalledApplications": [("Software Discovery", "T1418", "Discovery")],
    "MediaRecorder": [("Audio Capture", "T1429", "Collection")],
    "startRecording": [("Audio Capture", "T1429", "Collection")],
    "getCellLocation": [("Location Tracking", "T1430", "Collection")],
    "createFromPdu": [("Protected User Data", "T1636", "Collection")],
    "getMessageBody": [("Protected User Data", "T1636", "Collection")],
    "Cipher": [("Encrypted Channel", "T1521", "Command and Control")],
    "SecretKeySpec": [("Encrypted Channel", "T1521", "Command and Control")],
    "KeyGenerator": [("Encrypted Channel", "T1521", "Command and Control")],
    "Base64": [("Obfuscated Files or Information", "T1406", "Defense Evasion")],
}


def map_features_to_mitre(
    node_importances: dict[str, float],
    node_names: list[str] | None = None,
) -> list[MitreMapping]:
    mappings = []
    seen = set()

    for feature_name, importance in sorted(
        node_importances.items(), key=lambda x: x[1], reverse=True
    ):
        # Check permissions
        for perm_key, entries in PERMISSION_MAPPINGS.items():
            if perm_key in feature_name:
                for technique, tech_id, tactic in entries:
                    key = (perm_key, tech_id)
                    if key not in seen:
                        seen.add(key)
                        mappings.append(MitreMapping(
                            feature=f"android.permission.{perm_key}",
                            feature_type="Permission",
                            technique=technique,
                            technique_id=tech_id,
                            tactic=tactic,
                        ))

        # Check API calls
        for api_key, entries in API_MAPPINGS.items():
            if api_key in feature_name:
                for technique, tech_id, tactic in entries:
                    key = (api_key, tech_id)
                    if key not in seen:
                        seen.add(key)
                        mappings.append(MitreMapping(
                            feature=api_key,
                            feature_type="API Call",
                            technique=technique,
                            technique_id=tech_id,
                            tactic=tactic,
                        ))

    return mappings


def get_tactic_summary(mappings: list[MitreMapping]) -> dict[str, list[str]]:
    tactics: dict[str, list[str]] = {}
    for m in mappings:
        for tactic in m.tactic.split(", "):
            tactic = tactic.strip()
            if tactic not in tactics:
                tactics[tactic] = []
            entry = f"{m.technique} ({m.technique_id})"
            if entry not in tactics[tactic]:
                tactics[tactic].append(entry)
    return tactics
