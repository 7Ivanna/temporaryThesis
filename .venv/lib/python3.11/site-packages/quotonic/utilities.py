import collections
import functools
from typing import Any, Callable


# This class is adapted from the Python decorator library
# https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
class memoized:
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated)."""

    def __init__(self, func: Callable) -> None:
        self.func = self.__wrapped__ = func
        self.cache: dict = {}

    def __call__(self, *args: Any) -> Any:
        if not isinstance(args, collections.abc.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self) -> Any:
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj: Any) -> Callable:
        """Support instance methods."""
        return functools.partial(self.__call__, obj)
