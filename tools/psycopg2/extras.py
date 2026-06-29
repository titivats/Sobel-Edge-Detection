from __future__ import annotations


class Range:
    pass


class DateRange(Range):
    pass


class DateTimeRange(Range):
    pass


class DateTimeTZRange(Range):
    pass


class NumericRange(Range):
    pass


def Inet(value):
    return value


class Json:
    def __init__(self, adapted):
        self.adapted = adapted

    def getquoted(self):
        return repr(self.adapted).encode("utf-8")


def register_hstore(*args, **kwargs):
    return None
