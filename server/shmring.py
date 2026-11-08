"""Shared-memory ring channel between processes (docs/blocklane-contract.md's transport lane):
a bounded single-producer/single-consumer ring in one mmap region. The producer writes frames
[u32 len][payload] wrapping around the buffer; the consumer reads them in order, zero copies
beyond the one into the caller's bytes — where a localhost socket copies payload -> kernel ->
payload again, this is write-once/read-once page sharing.

Two roles for it in this tree:
  * server-side: worker -> earthd bulk transport for multi-MB tiles (the TCP serialize/copy
    cost per tile the event loop pays today).
  * game-side seam: the same region opened by name from LuaJIT FFI (probes/p2*_ffi.lua
    already probe that boundary) — a mod could read frames with no socket at all.

Contract: a full ring drops NOTHING — write() returns False and the caller falls back to its
existing transport (the socket), matching 'never a reason to fail a download'. A torn frame
(header written but payload short — producer died mid-write) makes the ring look empty at the
frame start; the producer's seq bump happens only after the whole frame landed, so readers can
never see a half frame: write_seq is the commit point.

Layout:
    header   8B magic "BNGErng1", 8B capacity, 8B write_seq, 8B read_seq, 8B reserved
    frames   [u32 len][payload]...  (a u32 len of 0 at a frame position = wrapped: skip to 0)

Windows: named mapping (tagname) as blocklane does; POSIX: a named file under the cache dir.
"""
import mmap
import os
import struct
import sys
import time

MAGIC = b"BNGErng1"
HDR = struct.Struct("<8sQQQQ")                    # magic, capacity, write_seq, read_seq, reserved
LEN = struct.Struct("<I")
WSEQ_OFF = 16                                     # header field offsets (magic+capacity first)
RSEQ_OFF = 24
HDR_SZ = HDR.size
FRAME_CAP = 64 * 1024 * 1024                      # bytes of frame space (env BNGE_RING_BYTES)

_channels = {}                                    # name -> _Ring
_ch_lock = __import__("threading").Lock()


def _path(name):
    base = os.environ.get("BNGE_RING_DIR") or os.path.join(os.environ.get("TEMP", "/tmp"))
    return os.path.join(base, f"bnge-ring-{name}")


class _Ring:
    def __init__(self, name, capacity=None):
        self.name = name
        self.capacity = capacity or int(os.environ.get("BNGE_RING_BYTES") or FRAME_CAP)
        size = HDR_SZ + self.capacity
        path = _path(name)
        try:
            if sys.platform == "win32":
                tag = "bnge-ring-" + name
                self.mm = mmap.mmap(-1, size, tagname=tag)
            else:
                new = not os.path.exists(path) or os.path.getsize(path) < size
                fd = os.open(path, os.O_RDWR | os.O_CREAT)
                try:
                    if new:
                        os.ftruncate(fd, size)
                    self.mm = mmap.mmap(fd, size)
                finally:
                    os.close(fd)
            magic, cap, _w, _r, _ = HDR.unpack(self.mm[:HDR_SZ])
            if magic != MAGIC or cap != self.capacity:
                self.mm[:HDR_SZ] = HDR.pack(MAGIC, self.capacity, 0, 0, 0)
            self.ok = True
        except (OSError, ValueError, BufferError):
            self.ok = False
            self.mm = None

    def _wseq(self):
        return struct.unpack_from("<Q", self.mm, WSEQ_OFF)[0]

    def _rseq(self):
        return struct.unpack_from("<Q", self.mm, RSEQ_OFF)[0]

    # --- producer --------------------------------------------------------------------------------
    def write(self, b):
        """Frame b into the ring; False when it cannot fit right now (caller: socket fallback).
        wseq counts every written byte including wrap padding; used space is wseq - rseq."""
        need = LEN.size + len(b)
        if need > self.capacity:
            return False                        # a frame bigger than the ring never fits
        w, r = self._wseq(), self._rseq()
        pos = w % self.capacity
        pad = self.capacity - pos if pos + need > self.capacity else 0
        if w - r + pad + need > self.capacity:
            return False                        # full: the reader must drain first
        if pad:
            if pos + LEN.size <= self.capacity:  # marker fits: written; else the reader's
                self.mm[HDR_SZ + pos:HDR_SZ + pos + LEN.size] = LEN.pack(0)  # own 'pos past end' check skips it
            w += pad
            pos = 0
        self.mm[HDR_SZ + pos:HDR_SZ + pos + LEN.size] = LEN.pack(len(b))
        self.mm[HDR_SZ + pos + LEN.size:HDR_SZ + pos + need] = b
        # len and payload fully visible before the commit — write_seq moves last
        struct.pack_into("<Q", self.mm, WSEQ_OFF, w + need)
        return True

    # --- consumer --------------------------------------------------------------------------------
    def read(self, timeout_s=0.0):
        """The next frame, or None (empty after timeout_s of polling, xproc-style)."""
        t0 = time.monotonic()
        while True:
            w, r = self._wseq(), self._rseq()
            if r < w:
                pos = r % self.capacity
                if pos + LEN.size > self.capacity:   # the pad's marker didn't fit: implicit wrap
                    self._advance(r + (self.capacity - pos))
                    continue
                n = LEN.unpack_from(self.mm, HDR_SZ + pos)[0]
                if n == 0:                      # wrap pad: resume at 0
                    self._advance(r + (self.capacity - pos))
                    continue
                b = bytes(self.mm[HDR_SZ + pos + LEN.size:HDR_SZ + pos + LEN.size + n])
                self._advance(r + LEN.size + n)
                return b
            if time.monotonic() - t0 >= timeout_s:
                return None
            time.sleep(0.0005)

    def _advance(self, new_r):
        struct.pack_into("<Q", self.mm, RSEQ_OFF, new_r)


def channel(name, capacity=None):
    """The named ring (created on first use; 'bnge-ring-<name>')."""
    with _ch_lock:
        ch = _channels.get(name)
        if ch is None:
            ch = _Ring(name, capacity)
            _channels[name] = ch
        return ch
