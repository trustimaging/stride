
import re

# Code to handle nodelists adapted from https://www.nsc.liu.se/~kent/python-hostlist

__all__ = ['expand_hostlist']

# Configuration to guard against ridiculously long expanded lists
MAX_SIZE = 100000


def expand_hostlist(hostlist):
    """
    Expand a hostlist expression string to a Python list.

    Example: expand_hostlist("n[9-11],d[01-02]") ==> 
             ['n9', 'n10', 'n11', 'd01', 'd02']

    Unless allow_duplicates is true, duplicates will be purged
    from the results. If sort is true, the output will be sorted.
    """

    results = []
    bracket_level = 0
    part = ""

    for c in hostlist + ",":
        if c == "," and bracket_level == 0:
            # Comma at top level, split!
            if part:
                results.extend(expand_part(part))
            part = ""
            bad_part = False
        else:
            part += c

        if c == "[": bracket_level += 1
        elif c == "]": bracket_level -= 1

        if bracket_level > 1:
            raise RuntimeError("nested brackets")
        elif bracket_level < 0:
            raise RuntimeError("unbalanced brackets")

    if bracket_level > 0:
        raise RuntimeError("unbalanced brackets")

    return results


def expand_part(s):
    """
    Expand a part (e.g. "x[1-2]y[1-3][1-3]") (no outer level commas).
    """

    # Base case: the empty part expand to the singleton list of ""
    if s == "":
        return [""]

    # Split into:
    # 1) prefix string (may be empty)
    # 2) rangelist in brackets (may be missing)
    # 3) the rest

    m = re.match(r'([^,\[]*)(\[[^\]]*\])?(.*)', s)
    (prefix, rangelist, rest) = m.group(1, 2, 3)

    # Expand the rest first (here is where we recurse!)
    rest_expanded = expand_part(rest)

    # Expand our own part
    if not rangelist:
        # If there is no rangelist, our own contribution is the prefix only
        us_expanded = [prefix]
    else:
        # Otherwise expand the rangelist (adding the prefix before)
        us_expanded = expand_rangelist(prefix, rangelist[1:-1])

    # Combine our list with the list from the expansion of the rest
    # (but guard against too large results first)
    if len(us_expanded) * len(rest_expanded) > MAX_SIZE:
        raise RuntimeError("results too large")

    return [us_part + rest_part
            for us_part in us_expanded
            for rest_part in rest_expanded]


def expand_rangelist(prefix, rangelist):
    """
    Expand a rangelist (e.g. "1-10,14"), putting a prefix before.
    """

    # Split at commas and expand each range separately
    results = []
    for range_ in rangelist.split(","):
        results.extend(expand_range(prefix, range_))
    return results


def expand_range(prefix, range_):
    """
    Expand a range (e.g. 1-10 or 14), putting a prefix before.
    """

    # Check for a single number first
    m = re.match(r'^[0-9]+$', range_)
    if m:
        return ["%s%s" % (prefix, range_)]

    # Otherwise split low-high
    m = re.match(r'^([0-9]+)-([0-9]+)$', range_)
    if not m:
        raise RuntimeError("bad range")

    (s_low, s_high) = m.group(1 ,2)
    low = int(s_low)
    high = int(s_high)
    width = len(s_low)

    if high < low:
        raise RuntimeError("start > stop")
    elif high - low > MAX_SIZE:
        raise RuntimeError("range too large")

    results = []
    for i in range(low, high + 1):
        results.append("%s%0*d" % (prefix, width, i))
    return results
