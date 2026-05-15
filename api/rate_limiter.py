import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import HTTPException, Request


class InMemoryRateLimiter:
    def __init__(self, max_requests: int = 120, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.events: Dict[str, Deque[float]] = defaultdict(deque)

    def check(self, request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        now = time.time()
        q = self.events[client]
        while q and now - q[0] > self.window_seconds:
            q.popleft()
        if len(q) >= self.max_requests:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        q.append(now)


limiter = InMemoryRateLimiter()
