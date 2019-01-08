

def myround(num):
    return round(num, 3)


def estimation_value(indicator_sum, n):
    return round(indicator_sum / n, 3)


def get_option(v, opts, f=None):
    if f is None:
        def f(a, b):
            return abs(a - b)
    a = []
    for opt in opts:
        a.append(round(f(v, opt), 3))

    # print(a)
    return to_letter(a.index(min(a)))


def to_letter(i):
    return ['a', 'b', 'c', 'd', 'e', 'f'][i]
