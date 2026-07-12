"""Self-signed TLS certificate generation/caching for the phone-tilt server.

iOS Safari only fires DeviceOrientationEvent in a secure context (HTTPS), and
there's no CA-signed option for a LAN-only address, so we generate and cache
a self-signed cert per machine (see certs/ in .gitignore) rather than depend
on a system `openssl` binary that Windows doesn't ship by default.
"""
from __future__ import annotations

import datetime as dt
import ipaddress
import ssl
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

_REGEN_MARGIN = dt.timedelta(days=30)


def ensure_self_signed_cert(cert_dir: str | Path, ip: str, valid_days: int = 365) -> tuple[Path, Path]:
    """Return (cert_path, key_path) under cert_dir, (re)generating as needed."""
    cert_dir = Path(cert_dir)
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path = cert_dir / "cert.pem"
    key_path = cert_dir / "key.pem"

    if cert_path.exists() and key_path.exists() and _is_cert_valid(cert_path, ip):
        return cert_path, key_path

    _generate_cert(cert_path, key_path, ip, valid_days)
    return cert_path, key_path


def build_ssl_context(cert_path: str | Path, key_path: str | Path) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    return context


def _is_cert_valid(cert_path: Path, ip: str) -> bool:
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    except (ValueError, OSError):
        return False

    if cert.not_valid_after_utc - _REGEN_MARGIN < dt.datetime.now(dt.timezone.utc):
        return False  # expired, or expiring soon enough to regenerate now

    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    except x509.ExtensionNotFound:
        return False
    san_ips = {str(name) for name in san.value.get_values_for_type(x509.IPAddress)}
    san_dns = set(san.value.get_values_for_type(x509.DNSName))
    return ip in san_ips or ip in san_dns


def _generate_cert(cert_path: Path, key_path: Path, ip: str, valid_days: int) -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, ip)])
    now = dt.datetime.now(dt.timezone.utc)

    san_names: list[x509.GeneralName] = [
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        x509.DNSName("localhost"),
    ]
    try:
        san_names.insert(0, x509.IPAddress(ipaddress.ip_address(ip)))
    except ValueError:
        san_names.insert(0, x509.DNSName(ip))

    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=valid_days))
        .add_extension(x509.SubjectAlternativeName(san_names), critical=False)
        .sign(key, hashes.SHA256())
    )

    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
