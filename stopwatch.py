import time


class Stopwatch():
    def __init__(self):
        self.total = 0.0
        self._start = 0.0
        self.running = False

    def start(self):
        if self.running:
            return

        self._start = time.time()
        self.running = True

    def stop(self):
        if not self.running:
            return

        self.total += time.time() - self._start
        self.running = False

    def elapsed(self):
        return self.total