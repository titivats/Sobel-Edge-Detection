from __future__ import annotations


ISOLATION_LEVEL_READ_UNCOMMITTED = 1
ISOLATION_LEVEL_READ_COMMITTED = 2
ISOLATION_LEVEL_REPEATABLE_READ = 3
ISOLATION_LEVEL_SERIALIZABLE = 4


class _Adapted:
    encoding = "utf8"

    def __init__(self, value):
        self.value = value

    def getquoted(self):
        return repr(self.value).encode("utf-8")


def adapt(value):
    return _Adapted(value)
