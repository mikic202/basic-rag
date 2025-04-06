from time import sleep, time

### TODO: Refactor


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class RequestController(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self._last_request = time()
        self._available_requests = 4

    def attempt_request(self) -> None:
        if self._available_requests > 0:
            self._available_requests -= 1
            return
        sleep(60)
        self._available_requests = 4
