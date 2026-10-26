# The MIT License (MIT)
# Copyright © 2026 Entrius

"""The Cathedral receipt inside ``attestation.json`` (``signer.kind == 'cathedral'``, the receipt at ``signer.receipt``).

Cathedral (cathedral.computer) runs the official run itself in a fresh TDX VM that the miner has no shell in, and signs
a receipt that binds the result, the solver, the image, the seed block, the attempt count and who submitted it. The
format is ``cathedral_challenge_receipt_v1`` (a TDX quote) or ``cathedral_challenge_receipt_v1_1``, which also allows an
AMD SEV-SNP report from Cathedral's sealed capacity and changes nothing else. ``attestation.verify`` has already checked the signature over gittensor's
own digest with a key pinned in ``challenge.json``; ``receipt_error`` checks the rest, offline, and never runs code.

Not checked here: Intel's signature chain over the TDX quote, and AMD's (ARK, ASK, VCEK) over the SNP report. Without it the trust level is ``platform`` (Cathedral's
signature, a quote that is well formed and bound to the receipt) and ``min_trust: hardware`` is refused.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

from gittensor.challenges.attestation import CATHEDRAL, POLARIS, Attestation, digest, ed25519_ok, trusted_pubkey
from gittensor.challenges.head_to_head import canonical

SCHEMA = 'cathedral_challenge_receipt_v1'
SCHEMA_V1_1 = 'cathedral_challenge_receipt_v1_1'  # v1 plus ``tee.type: amd_sev_snp``; v1 receipts verify as before
TDX, SNP = 'intel_tdx', 'amd_sev_snp'
TEE_TYPES = {SCHEMA: (TDX,), SCHEMA_V1_1: (TDX, SNP)}
TRUST_LEVELS = ('platform', 'hardware')
PLATFORM_SIGNER = 'cathedral_platform'  # what a receipt calls its own signer, as against ``signer.kind`` outside it
CLAIMS_DOMAIN = b'cathedral-challenge-receipt-v1/claims\0'
IMAGE_DOMAIN = b'cathedral-tdx-image-v1\0'
TOP_KEYS = {'schema', 'receipt_id', 'issued_at', 'claims', 'tee', 'trust', 'signer', 'ext', 'signature'}
CLAIM_KEYS = {'attestation_digest', 'challenge', 'evaluator', 'run', 'submission', 'outputs'}
QUOTE_HEADER = 48
QUOTE_BODY = (  # the TDX v4 report body: (field, bytes), in order
    *(('tee_tcb_svn', 16), ('mrseam', 48), ('mrsignerseam', 48), ('seam_attributes', 8), ('td_attributes', 8)),
    *(('xfam', 8), ('mrtd', 48), ('mrconfigid', 48), ('mrowner', 48), ('mrownerconfig', 48)),
    *(('rtmr0', 48), ('rtmr1', 48), ('rtmr2', 48), ('rtmr3', 48), ('report_data', 64)),
)
SNP_TEE_KEYS = {'type', 'binding', 'report_b64', 'report_sha256', 'cert_chain', 'measurement'}
SNP_REPORT_SIZE = 1184
SNP_REPORT = (  # the AMD SEV-SNP ATTESTATION_REPORT fields checked here: (field, offset, bytes), little-endian
    *(('version', 0x00, 4), ('policy', 0x08, 8), ('vmpl', 0x30, 4), ('signature_algo', 0x34, 4)),
    *(('key_info', 0x48, 4), ('report_data', 0x50, 64), ('measurement', 0x90, 48), ('reported_tcb', 0x180, 8)),
    *(('chip_id', 0x1A0, 64), ('signature', 0x2A0, 144)),  # the signature (r, s) covers bytes 0..0x2A0
)
SNP_VERSIONS = (3, 4, 5)
SNP_ECDSA_P384_SHA384 = 1
SNP_POLICY_RESERVED_ONE, SNP_POLICY_MIGRATE_MA, SNP_POLICY_DEBUG = 1 << 17, 1 << 18, 1 << 19


@dataclass(frozen=True)
class Author:
    """Who opened the PR: the receipt must name this GitHub account. ``hotkey`` is ``None`` when not looked up."""

    login: str
    github_id: int
    hotkey: str | None = None


def receipt_bytes(doc: object) -> bytes:
    """Cathedral's canonical bytes: sorted keys, compact, ASCII, no NaN. Differs from ``canonical`` only in escaping."""
    return json.dumps(doc, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


def has_float(value: object) -> bool:
    if isinstance(value, dict):
        return any(has_float(v) for v in value.values())
    if isinstance(value, list):
        return any(has_float(v) for v in value)
    return isinstance(value, float)


def timestamp(value: str) -> datetime:
    return datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%fZ')


def quote_fields(quote: bytes) -> dict[str, bytes]:
    """The fields of a TDX v4 quote's report body. Raises ``ValueError`` when it is not one."""
    if len(quote) < QUOTE_HEADER + sum(size for _, size in QUOTE_BODY):
        raise ValueError('the quote is shorter than a TDX v4 quote')
    if int.from_bytes(quote[0:2], 'little') != 4 or int.from_bytes(quote[4:8], 'little') != 0x81:
        raise ValueError('the quote is not a TDX v4 quote')
    fields, offset = {}, QUOTE_HEADER
    for name, size in QUOTE_BODY:
        fields[name], offset = quote[offset : offset + size], offset + size
    return fields


def snp_fields(report: bytes) -> dict[str, bytes]:
    """The fields of an SNP attestation report. Raises ``ValueError`` when it is not one."""
    if len(report) != SNP_REPORT_SIZE:
        raise ValueError(f'the SNP report is {len(report)} bytes, not {SNP_REPORT_SIZE}')
    return {name: report[offset : offset + size] for name, offset, size in SNP_REPORT}


def image_measurement(fields: dict[str, bytes]) -> str:
    h = hashlib.sha256(IMAGE_DOMAIN)
    for name in ('td_attributes', 'xfam', 'mrtd', 'rtmr0', 'rtmr1', 'rtmr2', 'rtmr3'):
        h.update(fields[name])
    return f'tdx-image-sha256:{h.hexdigest()}'


def pinned_error(config) -> str | None:
    """Why no Cathedral attestation can be accepted under ``config`` yet, or ``None``: challenge.json does not name
    Cathedral, or pins no key that parses (a placeholder). A miner checks this before paying for a run."""
    policy = config.attestation
    if policy.get('kind') not in (CATHEDRAL, POLARIS):
        return 'challenge.json does not name Cathedral as its attestation'
    keys = policy.get('trusted_keys')
    if not isinstance(keys, dict) or not any(trusted_pubkey(keys, key_id) for key_id in keys):
        return 'challenge.json pins no usable Cathedral key yet'
    return None


def receipt_error(att: Attestation, config, author: Author | None = None) -> str | None:
    """Why the attestation's Cathedral receipt is not acceptable for ``config``, or ``None``. When challenge.json says
    ``attestation.kind`` is ``cathedral`` (or ``polaris``, the platform's planned name, whose signer is Cathedral) any
    other signer is refused, a dev key included, so relabeling a Cathedral
    attestation as a dev one never skips the receipt. Otherwise a signer that is not ``cathedral`` has no receipt:
    ``None``. ``author`` adds the check that the receipt names the PR's author."""
    policy, kind = config.attestation, att.signer.get('kind')
    if policy.get('kind') in (CATHEDRAL, POLARIS) and kind != CATHEDRAL:
        return f'challenge.json accepts only a Cathedral attestation, not a {kind!r} signer'
    if kind != CATHEDRAL:
        return None
    try:
        return _receipt_error(att, config, policy, author)
    except (KeyError, TypeError, ValueError, AttributeError, binascii.Error) as e:
        return f'the receipt is malformed ({e!r})'


def _receipt_error(att: Attestation, config, policy: dict, author: Author | None) -> str | None:
    receipt = att.signer.get('receipt')
    if not isinstance(receipt, dict) or receipt.get('schema') not in TEE_TYPES:
        return f'no {" or ".join(TEE_TYPES)} in signer.receipt'
    if set(receipt) != TOP_KEYS or set(receipt['claims']) != CLAIM_KEYS:
        return 'the receipt has missing or unknown fields'
    if has_float(receipt):
        return 'the receipt holds a float'
    pubkey = trusted_pubkey(policy.get('trusted_keys'), att.signer.get('key_id'))
    entry = policy['trusted_keys'][att.signer['key_id']] if pubkey else {}
    if pubkey is None or not timestamp(entry['valid_from']) <= timestamp(receipt['issued_at']) < timestamp(
        entry['valid_until']
    ):
        return f'key {att.signer.get("key_id")} is not a trusted key for a receipt issued {receipt["issued_at"]}'
    body = {k: v for k, v in receipt.items() if k != 'signature'}
    if not ed25519_ok(pubkey, base64.b64decode(receipt['signature']['value_base64']).hex(), receipt_bytes(body)):
        return 'the receipt signature does not verify'

    claims, result = receipt['claims'], att.result
    attested = digest(att.result, att.seed_block, att.image).hex()
    result_sha = hashlib.sha256(canonical(result).encode()).hexdigest()
    run, sub, out = claims['run'], claims['submission'], claims['outputs']
    # Cathedral counts runs per (GitHub id, challenge, round): a round is one KING, so editing the solver does not
    # reset the count. The cap is the maintainer's: absent, attempts are only recorded.
    attempt, cap = run['attempt'], policy.get('max_attempts')
    if type(attempt) is not int or attempt < 1:
        return f'the receipt attempt {attempt!r} is not a run count from 1'
    if cap is not None and (type(cap) is not int or attempt > cap):
        return f'attempt {attempt} is over max_attempts {cap!r}: challenge.json allows {cap!r} runs per miner and round'
    if (level := receipt['trust']['claimed_level']) not in TRUST_LEVELS:
        return f'the receipt claims trust level {level!r}, not one of {", ".join(TRUST_LEVELS)}'
    expected = {
        'receipt signer key': (receipt['signer']['key_id'], att.signer['key_id']),
        'receipt signer kind': (receipt['signer']['kind'], PLATFORM_SIGNER),
        'receipt signer algorithm': (receipt['signer']['algorithm'], 'ed25519'),
        'attestation digest': (claims['attestation_digest'], attested),
        'result hash': (out['result_sha256'], result_sha),
        'challenge id': (claims['challenge']['challenge_id'], config.challenge_id),
        'module': (claims['challenge']['module'], config.module),
        'tier': (claims['challenge']['tier'], config.tier),
        'seeds': (claims['challenge']['seeds'], config.seeds),
        'image': (claims['evaluator']['image'], config.image),
        'egress': (claims['evaluator']['egress'], 'none'),
        'seed block': (run['seed_block'], att.seed_block),
        'seed block hash': (run['seed_block_hash'], result.get('seed_block_hash')),
        'exit code': (run['exit_code'], 0),
        'challenger sha': (sub['challenger_tree_sha'], result['challenger']['sha']),
        'king sha': (sub['king_tree_sha'], result['king']['sha']),
        'crown': (out['crown'], result['crown']),
    }
    if 'repo' in policy:  # which challenge repo's runs these are; absent, the challenge id above is all there is
        expected['repo'] = (claims['challenge']['repo'], policy['repo'])
    if wrong := [f'{name} ({got!r} is not {want!r})' for name, (got, want) in expected.items() if got != want]:
        return f'the receipt does not match the attestation or challenge.json: {", ".join(wrong)}'
    if error := evidence_error(receipt, claims, policy):
        return error
    if policy.get('min_trust', 'platform') != 'platform':
        chain = 'AMD VCEK chain' if receipt['tee']['type'] == SNP else 'Intel quote chain'
        return f'min_trust {policy["min_trust"]!r} needs the {chain} check, which is not built in yet'
    if author:
        return author_error(sub, author)
    return None


def evidence_error(receipt: dict, claims: dict, policy: dict) -> str | None:
    """The receipt's ``tee`` is a kind its schema allows, and that evidence checks out."""
    kind = receipt['tee'].get('type')
    if kind not in TEE_TYPES[receipt['schema']]:
        return f'tee.type {kind!r} is not allowed under {receipt["schema"]}'
    return snp_error(receipt['tee'], claims, policy) if kind == SNP else quote_error(receipt, claims, policy)


def quote_error(receipt: dict, claims: dict, policy: dict) -> str | None:
    """The TDX quote is well formed, not a debug TD, and carries the receipt's claims in its report data."""
    tee = receipt['tee']
    quote = base64.b64decode(tee['quote_b64'], validate=True)
    if tee['type'] != 'intel_tdx' or hashlib.sha256(quote).hexdigest() != tee['quote_sha256']:
        return 'the receipt has no TDX quote, or its hash is wrong'
    fields = quote_fields(quote)
    if fields['td_attributes'][0] & 1:
        return 'the quote is from a debug TD'
    commitment = hashlib.sha256(CLAIMS_DOMAIN + receipt_bytes(claims)).digest()
    binding = tee['binding']
    if binding == 'rd-claims-v1':
        bound = fields['report_data'] == commitment + bytes(32)
    elif binding == 'rd-guestquote-v0':
        bound = (
            fields['report_data'][:32] == hashlib.sha256((commitment.hex() + tee['e2e_pubkey_b64']).encode()).digest()
        )
    else:
        return f'unknown quote binding {binding!r}'
    if not bound:
        return 'the quote does not carry these claims'
    if image_measurement(fields) != tee['image_measurement']:
        return 'the quote does not match the receipt image measurement'
    allowed = policy.get('tdx_image_measurements') or []
    if binding == 'rd-claims-v1' and allowed and tee['image_measurement'] not in allowed:
        return f'runner measurement {tee["image_measurement"]} is not in challenge.json'
    return None


def snp_error(tee: dict, claims: dict, policy: dict) -> str | None:
    """The SNP report is well formed, from a production guest (VMPL 0, debug and migration agent off, VCEK-signed),
    carries the receipt's claims in its report data, and has a measurement challenge.json pins. Unlike the TDX
    allowlist the pin is required: an empty ``snp_measurements`` refuses every SNP receipt."""
    if set(tee) != SNP_TEE_KEYS or tee['binding'] != 'rd-claims-v1':
        return 'the SNP evidence has missing or unknown fields, or a binding other than rd-claims-v1'
    report = base64.b64decode(tee['report_b64'], validate=True)
    if len(report) != SNP_REPORT_SIZE or hashlib.sha256(report).hexdigest() != tee['report_sha256']:
        return 'the receipt has no SNP report, or its hash is wrong'
    fields = snp_fields(report)
    number = {name: int.from_bytes(fields[name], 'little') for name in ('version', 'policy', 'vmpl', 'key_info')}
    policy_bits, key_info = number['policy'], number['key_info']
    if number['version'] not in SNP_VERSIONS:
        return f'SNP report version {number["version"]} is not one of {SNP_VERSIONS}'
    if policy_bits & SNP_POLICY_DEBUG:
        return 'the SNP guest policy allows debug'
    if not policy_bits & SNP_POLICY_RESERVED_ONE or policy_bits & SNP_POLICY_MIGRATE_MA:
        return 'the SNP guest policy has reserved bit 17 clear or allows a migration agent'
    if number['vmpl'] != 0 or int.from_bytes(fields['signature_algo'], 'little') != SNP_ECDSA_P384_SHA384:
        return 'the SNP report is not from VMPL 0 or not ECDSA P-384 with SHA-384'
    if key_info & 0b10 or (key_info >> 2) & 0b111 != 0:  # MASK_CHIP_KEY set, or SIGNING_KEY not the VCEK (0)
        return 'the SNP report is not signed by a VCEK'
    if not all(any(fields[name]) for name in ('chip_id', 'measurement', 'reported_tcb', 'signature')):
        return 'the SNP report has a zero chip id, measurement, TCB or signature'
    commitment = hashlib.sha256(CLAIMS_DOMAIN + receipt_bytes(claims)).digest()
    if fields['report_data'] != commitment + bytes(32):
        return 'the SNP report does not carry these claims'
    if fields['measurement'].hex() != tee['measurement']:
        return 'the SNP report does not match the receipt measurement'
    if tee['measurement'] not in (policy.get('snp_measurements') or []):
        return f'SNP measurement {tee["measurement"]} is not in challenge.json'
    return None


def author_error(sub: dict, author: Author) -> str | None:
    """The receipt was issued to this GitHub account (and hotkey), for a solver under its own name."""
    if sub['github_id'] != author.github_id or sub['github_login'].lower() != author.login.lower():
        return f'the receipt was issued to {sub["github_login"]} ({sub["github_id"]}), not {author.login} ({author.github_id})'
    if sub['challenger_path'].split('/')[1:2] != [author.login]:
        return f'the receipt ran {sub["challenger_path"]}, not a solver of {author.login}'
    if author.hotkey and sub['hotkey'] != author.hotkey:
        return f'the receipt was issued to hotkey {sub["hotkey"]}, not {author.hotkey}'
    return None
