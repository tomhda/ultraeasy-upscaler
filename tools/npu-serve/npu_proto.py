"""親 <-> ワーカー共通バイナリプロトコル (UW2P)。

B2 (b2/pipe2.py・worker_f.py・worker_g.py) の不整合を持ち込まないための共通定義:
  - QUIT 8B vs 16B のような送受の非対称をなくし、全メッセージを同一ヘッダで読む。
  - ``read_exact`` と対になる ``write_all`` (短い write を扱い、戻り値を確認)。
  - 途中 EOF (TruncatedError) と正常終了 EOF (CleanEOF) を区別。
  - READY / DATA / ERROR / QUIT / PING を同じ規則で読む。
  - READY で名前・dtype・shape・バイト長を交換し、F 出力と G 入力を名前で照合
    (mean/std は同形状なので形状だけでは区別できない)。

ヘッダ (全てリトルエンディアン、合計 24B):
  magic(4) | version(u16) | type(u16) | generation(u32) | request_id(u32) | payload_len(u64)

generation はワーカーの起動世代。親が復旧で再起動するたび +1 し、
古い世代の応答を混ぜないために照合する。request_id は 1 往復の対応付け用で、
応答は要求と同じ値を返す。READY の request_id は 0。

このモジュールは標準ライブラリのみに依存し、NPU 環境 (PY180) と
アプリ側テスト環境の双方から import できる。numpy への依存は置かない
(非有限値率などの数値処理は呼び出し側で行う)。
"""
from __future__ import annotations

import json
import struct
from typing import BinaryIO, Callable

MAGIC = b"UW2P"
VERSION = 1

HEADER_LEN = 24
_HEADER_STRUCT = struct.Struct("<4sHHIIQ")

T_READY = 1
T_DATA = 2
T_ERROR = 3
T_QUIT = 4
T_PING = 5

_TYPE_NAMES = {
    T_READY: "READY",
    T_DATA: "DATA",
    T_ERROR: "ERROR",
    T_QUIT: "QUIT",
    T_PING: "PING",
}

#: DATA ペイロードのテンソル数上限 (誤同期時の暴走読みを防ぐ)。
MAX_TENSORS = 16
#: 1 テンソルあたりのバイト数上限 (F main 4MiB・G 出力 3MiB に対する余裕)。
MAX_TENSOR_BYTES = 64 * 1024 * 1024
#: ERROR/READY ペイロード (JSON) のバイト数上限。
MAX_JSON_BYTES = 256 * 1024


class ProtocolError(RuntimeError):
    """マジック不一致・版不一致・長さ異常などの同期破綻。"""


class CleanEOF(EOFError):
    """メッセージ境界での正常な EOF (相手の正常終了)。"""


class TruncatedError(EOFError):
    """メッセージ途中の EOF (kill・クラッシュ・本文途中切断)。"""

    def __init__(self, need: int, got: int, what: str = "payload") -> None:
        super().__init__(f"truncated {what} (need={need}, got={got})")
        self.need = need
        self.got = got
        self.what = what


def type_name(mtype: int) -> str:
    return _TYPE_NAMES.get(mtype, f"UNKNOWN({mtype})")


def encode_header(mtype: int, generation: int, request_id: int, payload_len: int) -> bytes:
    return _HEADER_STRUCT.pack(MAGIC, VERSION, mtype, generation, request_id, payload_len)


def decode_header(buf: bytes) -> tuple[int, int, int, int]:
    """24B ヘッダを (mtype, generation, request_id, payload_len) に解く。"""
    if len(buf) != HEADER_LEN:
        raise ProtocolError(f"bad header length: {len(buf)} != {HEADER_LEN}")
    magic, version, mtype, generation, request_id, payload_len = _HEADER_STRUCT.unpack(buf)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic: {magic!r} (protocol desync)")
    if version != VERSION:
        raise ProtocolError(f"unsupported version: {version} != {VERSION}")
    if mtype not in _TYPE_NAMES:
        raise ProtocolError(f"unknown message type: {mtype}")
    return mtype, generation, request_id, payload_len


def read_exact(read: Callable[[int], bytes | None], size: int, what: str = "payload") -> bytes:
    """read(n) を繰り返してちょうど size バイト読む。

    read は ``read(n) -> bytes | None`` (None も EOF とみなす)。
    境界での EOF (1 バイトも読めない) は CleanEOF、
    途中 EOF は TruncatedError。
    """
    buf = bytearray()
    while len(buf) < size:
        try:
            chunk = read(size - len(buf))
        except EOFError as exc:
            raise TruncatedError(size, len(buf), what) from exc
        if not chunk:
            if len(buf) == 0:
                raise CleanEOF(f"clean EOF at {what} boundary")
            raise TruncatedError(size, len(buf), what)
        buf += chunk
    return bytes(buf)


def write_all(write: Callable[[bytes], int | None], data: bytes) -> int:
    """短い write を扱い、全バイトを書き切る。戻り値は書き込んだ総量。

    write は ``write(bytes) -> 書き込んだバイト数 | None``
    (None は「全量書き込み」扱い。BufferedWriter は常に int を返す)。
    0 返却が続く・None 以外の異常は ProtocolError。
    """
    view = memoryview(data)
    total = 0
    idle = 0
    while total < len(view):
        ret = write(view[total:].tobytes())
        if ret is None:
            total = len(view)
            break
        if ret < 0:
            raise ProtocolError(f"write returned {ret}")
        if ret == 0:
            idle += 1
            if idle > 1000:
                raise ProtocolError(f"write stalled at {total}/{len(view)} bytes")
            continue
        idle = 0
        total += ret
    if total != len(view):
        raise ProtocolError(f"short write: {total}/{len(view)} bytes")
    return total


def send_message(
    writer: BinaryIO,
    mtype: int,
    generation: int,
    request_id: int,
    payload: bytes = b"",
) -> None:
    write_all(writer.write, encode_header(mtype, generation, request_id, len(payload)))
    if payload:
        write_all(writer.write, payload)
    writer.flush()


def recv_header(reader: BinaryIO) -> tuple[int, int, int, int]:
    raw = read_exact(reader.read, HEADER_LEN, "header")
    return decode_header(raw)


def recv_payload(reader: BinaryIO, payload_len: int, what: str = "payload") -> bytes:
    if payload_len == 0:
        return b""
    return read_exact(reader.read, payload_len, what)


# ---------------------------------------------------------------- tensor bunch
# DATA ペイロード形式: <I count> + count x <Q nbytes> + 生バイト連結。
# テンソルの順序は READY で交換した入出力記述の順序 (名前順ソート) とする。

_TENSOR_COUNT_STRUCT = struct.Struct("<I")
_TENSOR_LEN_STRUCT = struct.Struct("<Q")


def pack_tensors(blobs: list[bytes]) -> bytes:
    if len(blobs) > MAX_TENSORS:
        raise ProtocolError(f"too many tensors: {len(blobs)} > {MAX_TENSORS}")
    out = bytearray(_TENSOR_COUNT_STRUCT.pack(len(blobs)))
    for blob in blobs:
        if len(blob) > MAX_TENSOR_BYTES:
            raise ProtocolError(f"tensor too large: {len(blob)} > {MAX_TENSOR_BYTES}")
        out += _TENSOR_LEN_STRUCT.pack(len(blob))
    for blob in blobs:
        out += blob
    return bytes(out)


def unpack_tensors(payload: bytes) -> list[bytes]:
    if len(payload) < _TENSOR_COUNT_STRUCT.size:
        raise ProtocolError(f"DATA payload too short: {len(payload)}")
    (count,) = _TENSOR_COUNT_STRUCT.unpack_from(payload, 0)
    if count > MAX_TENSORS:
        raise ProtocolError(f"too many tensors: {count} > {MAX_TENSORS}")
    offset = _TENSOR_COUNT_STRUCT.size
    lengths: list[int] = []
    for _ in range(count):
        if offset + _TENSOR_LEN_STRUCT.size > len(payload):
            raise ProtocolError("DATA payload truncated in length table")
        (length,) = _TENSOR_LEN_STRUCT.unpack_from(payload, offset)
        offset += _TENSOR_LEN_STRUCT.size
        if length > MAX_TENSOR_BYTES:
            raise ProtocolError(f"tensor too large: {length} > {MAX_TENSOR_BYTES}")
        lengths.append(length)
    blobs: list[bytes] = []
    for length in lengths:
        if offset + length > len(payload):
            raise ProtocolError(
                f"DATA payload truncated in body (need={length}, "
                f"remain={len(payload) - offset})"
            )
        blobs.append(payload[offset:offset + length])
        offset += length
    if offset != len(payload):
        raise ProtocolError(f"DATA payload has {len(payload) - offset} trailing bytes")
    return blobs


# ---------------------------------------------------------------- JSON payload

def pack_json(obj: dict) -> bytes:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    if len(raw) > MAX_JSON_BYTES:
        raise ProtocolError(f"JSON payload too large: {len(raw)} > {MAX_JSON_BYTES}")
    return raw


def unpack_json(payload: bytes) -> dict:
    if len(payload) > MAX_JSON_BYTES:
        raise ProtocolError(f"JSON payload too large: {len(payload)} > {MAX_JSON_BYTES}")
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError(f"bad JSON payload: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError(f"JSON payload must be an object, got {type(obj).__name__}")
    return obj
