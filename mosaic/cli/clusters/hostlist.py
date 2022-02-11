
import re

# Code to handle nodelists adapted from https://www.nsc.liu.se/~kent/python-hostlist

__all__ = ['expand_hostlist']

# Configuration to guard against ridiculously long expanded lists
MAX_SIZE = 100000


def expand_hostlist(hostlist, allow_duplicates=False, sort=False):
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
            if part: results.extend(expand_part(part))
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

    if not allow_duplicates:
        results = remove_duplicates(results)
    if sort:
        results = numerically_sorted(results)
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
    (prefix, rangelist, rest) = m.group(1 ,2 ,3)

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


def remove_duplicates(l):
    """
    Remove duplicates from a list (but keep the order).
    """
    seen = set()
    results = []
    for e in l:
        if e not in seen:
            results.append(e)
            seen.add(e)
    return results


def numerically_sorted(l):
    """
    Sort a list of hosts numerically.

    E.g. sorted order should be n1, n2, n10; not n1, n10, n2.
    """

    return sorted(l, key=numeric_sort_key)


numeric_sort_key_regexp = re.compile("([0-9]+)|([^0-9]+)")


def numeric_sort_key(x):
    """
    Compose a sorting key to compare strings "numerically":

    We split numerical (integer) and non-numerical parts into a list,
    making sure that the numerical parts are converted to Python ints,
    and then sort on the lists. Thus, if we sort x10y and x9z8, we will
    compare ["x", 10, "y"] with ["x", 9, "x", "8"] and return x9z8
    before x10y".

    Python 3 complication: We cannot compare int and str, so while we can
    compare x10y and x9z8, we cannot compare x10y and 9z8. Kludge: insert
    a blank string first if the list would otherwise start with an integer.
    This will give the same ordering as before, as integers seem to compare
    smaller than strings in Python 2.
    """

    keylist = [int(i_ni[0]) if i_ni[0] else i_ni[1]
               for i_ni in numeric_sort_key_regexp.findall(x)]
    if keylist and isinstance(keylist[0], int):
        keylist.insert(0, "")
    return keylist


