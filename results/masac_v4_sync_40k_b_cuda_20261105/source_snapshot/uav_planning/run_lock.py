"""Nonblocking process lock on POSIX and native Windows; never silently skipped."""
from __future__ import annotations

import os
from pathlib import Path


class RunLock:
    def __init__(self, path, metadata=''):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+b')
        self.held = False
        try:
            self.file.seek(0, 2)
            if self.file.tell() == 0:
                self.file.write(b'\0')
                self.file.flush()
            self.file.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.held = True
            self.file.seek(1)
            self.file.truncate(1)
            self.file.write(metadata.encode('utf-8'))
            self.file.flush()
        except OSError as exc:
            self.close()
            raise RuntimeError(f'Another runner holds lock, or lock unavailable: {self.path}') from exc

    def close(self):
        if self.file.closed:
            return
        try:
            if self.held:
                self.file.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.file, fcntl.LOCK_UN)
        finally:
            self.held = False
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
