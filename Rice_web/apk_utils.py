import os
import zipfile

try:
    from androguard.core.apk import APK
except Exception:
    from androguard.core.bytecodes.apk import APK


def _bytes_contains(blob: bytes, patterns):
    if not blob:
        return 0.0
    blob = blob.lower()
    for pattern in patterns:
        if isinstance(pattern, str):
            pattern = pattern.encode('utf-8', errors='ignore')
        if pattern.lower() in blob:
            return 1.0
    return 0.0


def _names_contain(file_names, suffix_or_text):
    suffix_or_text = suffix_or_text.lower()
    for name in file_names:
        lname = name.lower()
        if lname.endswith(suffix_or_text) or suffix_or_text in lname:
            return 1.0
    return 0.0


def extract_apk_features(apk_path, feature_names):
    if not os.path.exists(apk_path):
        raise FileNotFoundError(f'APK not found: {apk_path}')

    apk = APK(apk_path)
    if hasattr(apk, 'is_valid_APK') and not apk.is_valid_APK():
        raise ValueError('Invalid APK file.')

    permissions = set(apk.get_permissions() or [])
    package_name = apk.get_package() if hasattr(apk, 'get_package') else 'Unknown'
    app_name = apk.get_app_name() if hasattr(apk, 'get_app_name') else 'Unknown'

    manifest_text = ''
    try:
        manifest_xml = apk.get_android_manifest_xml()
        manifest_text = str(manifest_xml).lower()
    except Exception:
        manifest_text = ''

    file_names = []
    dex_blob = b''
    other_blob = b''
    try:
        with zipfile.ZipFile(apk_path, 'r') as zf:
            file_names = zf.namelist()
            for name in file_names:
                lname = name.lower()
                try:
                    raw = zf.read(name)
                except Exception:
                    continue
                if lname.endswith('.dex'):
                    dex_blob += raw[:4_000_000]
                elif lname.endswith(('.xml', '.txt', '.json', '.arsc', '.so')):
                    other_blob += raw[:1_000_000]
    except zipfile.BadZipFile:
        raise ValueError('Uploaded file is not a readable APK/ZIP archive.')

    searchable = manifest_text.encode('utf-8', errors='ignore') + b'\n' + dex_blob + b'\n' + other_blob

    perm_map = {
        'SEND_SMS': ['android.permission.SEND_SMS'],
        'DELETE_PACKAGES': ['android.permission.DELETE_PACKAGES', 'android.permission.REQUEST_DELETE_PACKAGES'],
        'PHONE_STATE': ['android.permission.READ_PHONE_STATE'],
        'RECEIVE_SMS': ['android.permission.RECEIVE_SMS'],
        'READ_SMS': ['android.permission.READ_SMS'],
        'CAMERA': ['android.permission.CAMERA'],
        'ACCESS_FINE_LOCATION': ['android.permission.ACCESS_FINE_LOCATION'],
        'INSTALL_PACKAGES': ['android.permission.INSTALL_PACKAGES', 'android.permission.REQUEST_INSTALL_PACKAGES'],
        'ACCESS_NETWORK_STATE': ['android.permission.ACCESS_NETWORK_STATE'],
        'BLUETOOTH': ['android.permission.BLUETOOTH', 'android.permission.BLUETOOTH_ADMIN', 'android.permission.BLUETOOTH_CONNECT'],
        'ACCESS_WIFI_STATE': ['android.permission.ACCESS_WIFI_STATE'],
        'BROADCAST_SMS': ['android.permission.BROADCAST_SMS'],
        'CALL_PHONE': ['android.permission.CALL_PHONE'],
        'CALL_PRIVILEGED': ['android.permission.CALL_PRIVILEGED'],
        'CLEAR_APP_CACHE': ['android.permission.CLEAR_APP_CACHE'],
        'CLEAR_APP_USER_DATA': ['android.permission.CLEAR_APP_USER_DATA'],
        'CONTROL_LOCATION_UPDATES': ['android.permission.CONTROL_LOCATION_UPDATES'],
        'INTERNET': ['android.permission.INTERNET'],
    }

    action_map = {
        'android.intent.action.BOOT_COMPLETED': ['android.intent.action.boot_completed'],
        'android.intent.action.BATTERY_LOW': ['android.intent.action.battery_low'],
        'android.intent.action.ACTION_POWER_CONNECTED': ['android.intent.action.action_power_connected', 'android.intent.action.power_connected'],
    }

    code_map = {
        'TelephonyManager.*getDeviceId': [b'getdeviceid', b'telephonymanager', b'getDeviceId'],
        'TelephonyManager.*getSubscriberId': [b'getsubscriberid', b'telephonymanager', b'getSubscriberId'],
        'abortBroadcast': [b'abortbroadcast'],
        'Ljava.net.InetSocketAddress': [b'inetsocketaddress', b'ljava/net/inetsocketaddress'],
        'io.File.*delete(': [b'delete(', b'java/io/file'],
        'chown': [b'chown'],
        'chmod': [b'chmod'],
        'mount': [b'mount'],
        'System.*loadLibrary': [b'loadlibrary', b'system.loadlibrary'],
    }

    features = {name: 0.0 for name in feature_names}

    for feat, candidates in perm_map.items():
        if feat in features:
            features[feat] = 1.0 if any(candidate in permissions for candidate in candidates) else 0.0

    for feat, tokens in action_map.items():
        if feat in features:
            features[feat] = 1.0 if any(token in manifest_text for token in tokens) else 0.0

    for feat, patterns in code_map.items():
        if feat in features:
            features[feat] = _bytes_contains(searchable, patterns)

    if '.apk' in features:
        features['.apk'] = _names_contain(file_names, '.apk')
    if '.zip' in features:
        features['.zip'] = _names_contain(file_names, '.zip')
    if '.dex' in features:
        features['.dex'] = _names_contain(file_names, '.dex')
    if '.so' in features:
        features['.so'] = _names_contain(file_names, '.so')
    if '.exe' in features:
        features['.exe'] = _names_contain(file_names, '.exe')

    meta = {
        'package_name': package_name,
        'app_name': app_name,
        'permissions_count': len(permissions),
    }
    return features, meta
