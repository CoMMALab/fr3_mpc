"""Shared memory utilities for multiprocessing."""

import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory


class SharedMemoryNumpyArray:
    """Helper class to store a numpy array in shared memory."""

    def __init__(self, arr: np.ndarray, ctx: mp.context.BaseContext):
        """Create a shared memory numpy array.

        Args:
            arr: The numpy array to store in shared memory. Size and dtype must
                 be fixed.
            ctx: The multiprocessing context to use for shared memory.
        """
        self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=self.shm.buf)
        shared_arr[:] = arr[:]
        self.shape = arr.shape
        self.dtype = arr.dtype
        self.lock = ctx.Lock()

    def __getitem__(self, key: int) -> np.ndarray:
        """Get an item from the shared array."""
        shm = shared_memory.SharedMemory(name=self.shm.name)
        arr = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
        return np.copy(arr[key])  # Need to copy here to avoid segfaults

    def __setitem__(self, key: int, value: np.ndarray) -> None:
        """Set an item in the shared array."""
        with self.lock:
            shm = shared_memory.SharedMemory(name=self.shm.name)
            arr = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
            arr[key] = value

    def __str__(self) -> str:
        """Return the string representation of the shared array."""
        shm = shared_memory.SharedMemory(name=self.shm.name)
        arr = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
        return str(arr)

    def __del__(self) -> None:
        """Clean up the shared memory on deletion."""
        self.shm.close()
        self.shm.unlink()
