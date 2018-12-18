from datetime import datetime, timedelta

def chop_microseconds(t):
    if isinstance(t, timedelta):
        delta = t
        return delta - timedelta(microseconds=delta.microseconds)
    elif isinstance(t, datetime):
        return t.replace(microsecond=0)
