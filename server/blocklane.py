"""Cross-process shared block memory (docs/blocklane-contract.md): one bounded mmap region per cache
directory the server's processes agree on, content-keyed like rangeread's _mem but visible to all of
them. A hot block one process fetched is served to the rest without a disk read.

Degradation contract: any setup failure (platform without a usable mmap, locked file, short disk)
turns the tier into a permanent miss — never a reason to fail a fetch, as xproc treats its locks.

Layout of the .shm file, fixed-size slots:
    header   8B magic "BNGEshm1", 4B slot count, 4B slot bytes
    index    slot_count * 32B: key_hash u64, block_index u32, stamp u32, size u32, reserved 16B
    payload  slot_count * slot_bytes of block data

key_hash is 64-bit FNV-1a of the file id (the abspath of the block's cache_dir today). stamp is a
monotonic tick for LRU eviction; sha integrity is unnecessary — a slot is verified by hash+index and
the reader's own block-length check already catches a torn payload the same way .blk does.
"""
import mmap
import os
import struct
import sys
import threading
import time

MAGIC = b"BNGEshm1"
HEADER = struct.Struct("<8sII")                   # magic, slots, slot_bytes
ENTRY = struct.Struct("<QIII8s")                  # key_hash, block_index, stamp, size, spare
ENTRY_SZ = ENTRY.size

_slots_default = 256                              # 256 * 256 KiB = 64 MiB shared
_stores = {}                                      # cache_dir -> _Shared
_stores_lock = threading.Lock()


def _fnv1a64(b):
    h = 0xcbf29ce484222325
    for c in b:
        h = ((h ^ c) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def _tick():
    return int(time.monotonic() * 1000) & 0x7FFFFFFF


class _Shared:
    """One mmap-backed slot table. All field access is guarded by the owning process's lock only for
    its own writers; readers tolerate a torn entry by treating it as a miss (the hash+size check
    would fail), which the contract permits."""

    def __init__(self, path, slots, slot_bytes):
        self.path, self.slots, self.slot_bytes = path, slots, slot_bytes
        self.size = HEADER.size + slots * ENTRY_SZ + slots * slot_bytes
        self._lk = threading.Lock()
        try:
            if sys.platform == "win32":
                # a named mapping: every process attaching the same tag shares the pages, none owns
                # a file (a .shm file in the cache dir cannot be unlinked while mapped on Windows,
                # and the pagefile backs it just as well). The last process out drops the pages —
                # warm state dies with the workers, disk .blk is the durable tier regardless.
                tag = "bnge-shm-" + ("%016x" % _fnv1a64(os.path.abspath(path).encode("utf-8")))
                self.mm = mmap.mmap(-1, self.size, tagname=tag)
            else:
                new = not os.path.exists(path) or os.path.getsize(path) < self.size
                fd = os.open(path, os.O_RDWR | os.O_CREAT)
                try:
                    if new:
                        os.ftruncate(fd, self.size)
                    self.mm = mmap.mmap(fd, self.size)
                finally:
                    os.close(fd)
            magic, ms, mb = HEADER.unpack(self.mm[:HEADER.size])
            if magic == MAGIC and (ms, mb) != (slots, slot_bytes):
                # attaching to a live region built with other dims: adopt them
                self.mm.close()
                self.slots, self.slot_bytes = ms, mb
                self.size = HEADER.size + ms * ENTRY_SZ + ms * mb
                if sys.platform == "win32":
                    self.mm = mmap.mmap(-1, self.size, tagname=tag)
                else:
                    fd = os.open(path, os.O_RDWR)
                    try:
                        self.mm = mmap.mmap(fd, self.size)
                    finally:
                        os.close(fd)
            elif magic != MAGIC:
                # fresh region (zeroed): initialise. Two processes racing here write
                # identical bytes — idempotent.
                self.mm[:HEADER.size] = HEADER.pack(MAGIC, slots, slot_bytes)
                self.mm[HEADER.size:] = b"\0" * (self.size - HEADER.size)
            self.ok = True
        except (OSError, ValueError, BufferError):
            self.ok = False
            self.mm = None

    def _entry_at(self, s):
        off = HEADER.size + s * ENTRY_SZ
        return ENTRY.unpack_from(self.mm, off), off

    def _data_off(self, s):
        return HEADER.size + self.slots * ENTRY_SZ + s * self.slot_bytes

    def read(self, key_hash, index):
        if not self.ok:
            return None
        for s in range(self.slots):
            (kh, bi, _st, sz, _sp), off = self._entry_at(s)
            if kh == key_hash and bi == index and sz:
                d = self._data_off(s)
                b = bytes(self.mm[d:d + sz])
                if len(b) == sz:                # torn slot read: a miss, not a corrupt block
                    self.mm[off + 12:off + 16] = struct.pack("<I", _tick())
                    return b
                return None
        return None

    def write(self, key_hash, index, b):
        if not self.ok or len(b) > self.slot_bytes:
            return
        with self._lk:
            # a matching slot first, else the oldest stamp (empty slots stamp 0 win for free)
            free, oldest, oldest_st = None, 0, 0x8FFFFFFF
            for s in range(self.slots):
                (kh, bi, st, sz, _sp), _off = self._entry_at(s)
                if kh == key_hash and bi == index:
                    free = s
                    break
                if not sz:
                    free = s
                    break
                if st < oldest_st:
                    oldest, oldest_st = s, st
            slot = free if free is not None else oldest
            d = self._data_off(slot)
            self.mm[d:d + len(b)] = b
            off = HEADER.size + slot * ENTRY_SZ
            self.mm[off:off + ENTRY_SZ] = ENTRY.pack(key_hash, index, _tick(), len(b), b"")

    def drop(self):
        if self.mm is not None:
            try:
                self.mm.close()
            except (OSError, BufferError):
                pass


def _shm_path(cache_dir):
    return os.path.join(cache_dir, ".shm")


def store_for(cache_dir, slot_bytes=256 * 1024, slots=None):
    """The shared store for one cache_dir (one per directory, like RemoteFile's _key)."""
    key = os.path.abspath(cache_dir)
    with _stores_lock:
        sh = _stores.get(key)
        if sh is None:
            slots = slots or int(os.environ.get("BNGE_SHARED_SLOTS") or _slots_default)
            if os.environ.get("BNGE_SHARED_MEM") == "off":
                slots = 0
            os.makedirs(key, exist_ok=True)
            sh = _Shared(_shm_path(key), max(slots, 1), slot_bytes) if slots else _Broken()
            _stores[key] = sh
        return sh


class _Broken:
    ok = False
    def read(self, k, i):
        return None
    def write(self, k, i, b):
        pass
    def drop(self):
        pass


def key_of(cache_dir):
    return _fnv1a64(os.path.abspath(cache_dir).encode("utf-8"))
