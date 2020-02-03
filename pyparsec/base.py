from collections import namedtuple, defaultdict

import logging
logger = logging.getLogger(__name__)

ParseResult = namedtuple("ParseResult", ["succeeded", "value", "rest", "expects"])


def fail(expects, remaining=None):
    return ParseResult(False, None, remaining or [], expects)


class Parser:
    def __call__(self, components):
        return self.try_parse(components)

    def try_parse(self, components):
        return fail([], components)

    def parse(self, components, partial=False):
        res = self.try_parse(components)
        if res.succeeded and (partial or not res.rest):
            return res.value
        elif res.succeeded:
            raise ValueError("Expected end of input; got " + str(res.rest))
        else:
            raise ValueError("Expected one of " + str(res.expects) + "; got " + str(res.rest))

    def possible_next(self, components):
        r = self.try_parse(components)
        if not r.succeeded and (not components or r.rest != components):
            return r.expects

        return []


class only(Parser):
    def __init__(self, v):
        self._v = v

    def try_parse(self, components):
        if components and components[0] == self._v:
            return ParseResult(True, components[0], components[1:], [])
        return fail([self._v], components)


class one_of(Parser):
    def __init__(self, values):
        self._values = values

    def try_parse(self, components):
        if components and components[0] in self._values:
            return ParseResult(True, components[0], components[1:], [])
        return fail(self._values, components)


class or_(Parser):
    def __init__(self, *parsers):
        self._parsers = parsers

    def try_parse(self, components):
        for p in self._parsers:
            r = p.try_parse(components)
            if r.succeeded:
                return r
        if self._parsers:
            return r
        else:
            return fail([])

    def possible_next(self, components):
        res = []
        for p in self._parsers:
            for ex in p.possible_next(components):
                if ex not in res:
                    res.append(ex)
        return res


class seq_(Parser):
    def __init__(self, *parsers, throw_away=None):
        self._parsers = parsers
        self._throw_away = throw_away

    def try_parse(self, components):
        remaining_components = list(components)
        values = []
        for p in self._parsers:
            r = p.try_parse(remaining_components)
            if not r.succeeded:
                return ParseResult(False, [v for v in values if v != self._throw_away], r.rest, r.expects)
            values.append(r.value)
            remaining_components = r.rest
        return ParseResult(True, [v for v in values if v != self._throw_away], remaining_components, [])

    def possible_next(self, components):
        res = []
        remaining_components = list(components)
        for i,p in enumerate(self._parsers):
            res += p.possible_next(remaining_components)
            r = p.try_parse(remaining_components)
            if not r.succeeded:
                break
            remaining_components = r.rest
        return res


class optional(Parser):
    def __init__(self, p, default=None):
        self._p = p
        self._default = default

    def try_parse(self, components):
        r = self._p(components)
        if r.succeeded:
            return r
        return ParseResult(True, self._default, components, [])

    def possible_next(self, components):
        return self._p.possible_next(components)


def transform(f):
    class transformer(Parser):
        def __init__(self, p, *args, **kwargs):
            if not isinstance(p, Parser):
                raise ValueError("First argument must be a Parser")

            self._f = f
            self._p = p
            self._args = args
            self._kwargs = kwargs

        def try_parse(self, components):
            r = self._p.try_parse(components)
            if r.succeeded:
                return ParseResult(True, self._f(r.value, *self._args, **self._kwargs), r.rest, [])
            else:
                return r

        def possible_next(self, components):
            return self._p.possible_next(components)
    return transformer


def transform_possible(f):
    class transformer_possible(Parser):
        def __init__(self, p, *args, **kwargs):
            if not isinstance(p, Parser):
                raise ValueError("First argument must be a Parser")

            self._f = f
            self._p = p
            self._args = args
            self._kwargs = kwargs

        def try_parse(self, components):
            return self._p.try_parse(components)

        def possible_next(self, components):
            pn = self._p.possible_next(components)
            return self._f(pn, *self._args, **self._kwargs)
    return transformer_possible


class any_(Parser):
    def __init__(self, fake_name):
        self._fake_name = fake_name
    def try_parse(self, components):
        if components:
            return ParseResult(True, components[0], components[1:], [])
        return fail([self._fake_name], components)


@transform
def named(value, name):
    return {name: value}

@transform
def drop(value):
    return None

@transform
def ll(v):
    return [v] if v else []

@transform
def exl(vs):
    return vs[0] if len(vs) > 0 else None

@transform
def concat(vvs):
    return [v for l in vvs for v in l]


class many(Parser):
    def __init__(self, p):
        if p.try_parse([]).succeeded:
            logger.warning("Warning: calling many() on a parser which will accept an empty list")
        self._p = p

    def try_parse(self, components):
        values = []
        remaining_components = list(components)
        while remaining_components:
            r = self._p.try_parse(remaining_components)
            if not r.succeeded:
                break
            values.append(r.value)
            remaining_components = r.rest

        return ParseResult(True, values, remaining_components, [])

    def possible_next(self, components):
        remaining_components = list(components)
        while remaining_components:
            r = self._p.try_parse(remaining_components)
            if not r.succeeded:
                break
            remaining_components = r.rest

        return self._p.possible_next(remaining_components)


class many_of(Parser):
    def __init__(self, parsers_with_max):
        self._parsers_with_max = parsers_with_max

    def _advance_state(self, components):
        values = []
        remaining_components = list(components)
        remaining_parsers = list(self._parsers_with_max)
        while remaining_components:
            if not any(c is None or c > 0 for p, c in remaining_parsers):
                break

            for i, (p, count) in list(enumerate(remaining_parsers)):
                if count is not None and count <= 0:
                    continue

                r = p.try_parse(remaining_components)
                if r.succeeded:
                    values.append(r.value)
                    remaining_components = r.rest
                    remaining_parsers[i] = (p, count-1 if count is not None else None)
                    break
            else:
                break

        return values, remaining_components, remaining_parsers

    def try_parse(self, components):
        values, remaining_components, _ = self._advance_state(components)
        return ParseResult(True, values, remaining_components, [])

    def possible_next(self, components):
        _, remaining_components, remaining_parsers = self._advance_state(components)
        remaining_combinator = or_(*[p for p, c in remaining_parsers if c is None or c > 0])
        return remaining_combinator.possible_next(remaining_components)

