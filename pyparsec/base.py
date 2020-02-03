from collections import namedtuple, defaultdict

import logging
logger = logging.getLogger(__name__)

ParseResult = namedtuple("ParseResult", ["succeeded", "value", "rest", "expects"])  # todo: expects description vs. values


# ## General scheme:
#
# We want to parse some sequence of tokens, according to a grammar, into a value. A parser has a grammar (patterns of tokens which it accepts),
# and a transformation with which it will turn acceptable inputs into output values. The transformation may be, e.g., the identity, in which case
# the result will be more or less a parse tree; or it may do some more work, e.g., evaluating an arithmetic expression. Parsers can be combined
# by, e.g., concatenation or alternation, to build up complex grammars from simple ones.
#
# For some Parser `p`, `p.try_parse(seq)` -> `r` - try to (partially) parse `seq` with `p`.
# `seq` should be list-like, with elements of arbitrary type. `r` will be a ParseResult, which will be successful if `p` accepted some prefix of `seq`.
# If successful, `value` will be the transformed result, and `rest` will be the remaining suffix of `seq`. If unsuccessful, then `expects` will be
# a list of tokens which could have continued the parse (in some sense - this isn't very well-defined and is only used for feedback messages).
# `p.parse(seq)` will _completely_ parse `seq`, returns only the result value, and throws an exception if the parse was not successful (including
# if there is anything left over).
#
# The basic parser is `only(x)` - this is a parser which matches a sequence `[y]` where `x == y`. e.g.,
#   `only(1).try_parse([1, 2])` -> `ParseResult(succeeded=True,  rest=[2],    value=1)`
#   `only(3).try_parse([1, 2])` -> `ParseResult(succeeded=False, rest=[1, 2], expects=[1])`
# Other important parsers are
# * seq(), concatenating parsers -
#   `seq(only(1), only(2)).try_parse([1, 2])` -> `ParseResult(succeeded=True, rest=[], value=[1, 2])`
# * or_(), alternating -
#   `or_(only(1), only(3)).try_parse([1, 2])` -> `ParseResult(succeeded=True, rest=[2], value=1)`
#   `or_(only(1), only(3)).try_parse([3, 2])` -> `ParseResult(succeeded=True, rest=[2], value=3)`
# * optional(), making a parser optional -
#   `optional(only(1)).try_parse([2])`    -> `ParseResult(succeeded=True, rest=[2], value=None)`
#   `optional(only(1)).try_parse([1, 2])` -> `ParseResult(succeeded=True, rest=[2], value=1)`
# * many(), repeating a parser -
#   `many(only(1)).try_parse([2])`       -> `ParseResult(succeeded=True, rest=[2], value=[])`
#   `many(only(1)).try_parse([1, 2])`    -> `ParseResult(succeeded=True, rest=[2], value=[1])`
#   `many(only(1)).try_parse([1, 1, 2])` -> `ParseResult(succeeded=True, rest=[2], value=[1, 1])`
#
# Parsers support arithmetic-like operations, and will implicitly convert non-parsers into parsers when doing this. e.g.,
# `(only('a') | 'bc') + 'def'` will fully match either 'adef' or 'bcdef'. Note that the list-like strings 'bc' and 'def' are
# treated as sequences of tokens and not individual tokens - this parser will _not_ match ['bc', 'def']. To use list-like types as individual
# tokens, you'd need to use the constructors explicitly, e.g., `seq(or_(only('a'), only('bc')), only_('def')`.
#
# Transformers liked named() apply transformations to the output value of a parser, while leaving the grammar unchanged.
#
# The provided parsers are enough to make a regular grammar (they are as expressive as regular expressions). In general, a parser is an arbitrary
# partial function and can parse anything. The implementation here is simplistic and doesn't really handle ambiguous or pathological parses. It also
# doesn't fully backtrack, so `many(a) + ab` will fail to parse `aab`, despite matching the grammar `a*ab`. Arbitrary backtracking like this would
# require try_parse() to return all possible partial parses rather than a single one, which is doable but substantially increases the complexity of
# any parser with children. I think that in practice most parsers which are suceptible to this can be trivially rewritten to a form which isn't,
# e.g., `a + many(a) + b` has the same grammar (though a slightly different output value) and avoids the issue.


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

    def __add__(self, other):
        return seq(self, _coerce(other))

    def __radd__(self, other):
        return seq(_coerce(other), self)

    def __or__(self, other):
        return or_(self, _coerce(other))

    def __ror__(self, other):
        return or_(_coerce(other), self)

    def __mul__(self, n):
        return seq(*(self for i in range(n)))

    def __invert__(self):
        return not_(self)



def _coerce(obj):
    if isinstance(obj, Parser):  # Already a parser
        return obj
    elif isinstance(obj, str):  # String-like
        if len(obj) == 1:  # Token
            return only(obj)
        else:  # Sequence of tokens
            return seq(*(only(c) for c in obj))
    elif hasattr(obj, '__iter__'):  # List-like: treat it as a sequence
        return seq(*(_coerce(x) for x in obj))
    else:  # Treat it as a token
        return only(obj)

def parser(x):
    return _coerce(x)


class only(Parser):
    def __init__(self, v):
        self._v = v

    def try_parse(self, components):
        if components and self._v == components[0]:
            return ParseResult(True, components[0], components[1:], [])
        return fail([self._v], components)



class Flattens:
    # Flattens([a, Flattens([b, c])])  ->  Flattens([a, b, c])
    # This is an optimisation for seq() and or_() to avoid deep nesting, e.g., a + b + c + d becomes a single seq()
    # of four elements, rather than three nested seq()s with two elements each.
    def __init__(self, parsers):
        flat_parsers = []
        for p in parsers:
            if isinstance(p, type(self)):
                flat_parsers += p.children
            elif isinstance(p, Parser):
                flat_parsers += [p]
            else:
                raise ValueError(type(p))
        self.children = flat_parsers


class or_(Flattens, Parser):
    def __init__(self, *parsers):
        Flattens.__init__(self, parsers)

    def try_parse(self, components):
        for p in self.children:
            r = p.try_parse(components)
            if r.succeeded:
                return r
        if self.children:
            return r
        else:
            return fail([])

    def possible_next(self, components):
        res = []
        for p in self.children:
            for ex in p.possible_next(components):
                if ex not in res:
                    res.append(ex)
        return res


class seq(Flattens, Parser):
    def __init__(self, *parsers, throw_away=None):
        Flattens.__init__(self, parsers)
        self._throw_away = throw_away

    def try_parse(self, components):
        remaining_components = list(components)
        values = []
        for p in self.children:
            r = p.try_parse(remaining_components)
            if not r.succeeded:
                return ParseResult(False, [v for v in values if v != self._throw_away], r.rest, r.expects)
            values.append(r.value)
            remaining_components = r.rest
        return ParseResult(True, [v for v in values if v != self._throw_away], remaining_components, [])

    def possible_next(self, components):
        res = []
        remaining_components = list(components)
        for i,p in enumerate(self.children):
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
    def __init__(self, fake_name=None):
        self._fake_name = fake_name if fake_name is not None else ''

    def try_parse(self, components):
        if components:
            return ParseResult(True, components[0], components[1:], [])
        return fail([self._fake_name], components)

class not_(Parser):  # Fails if p succeeds, and succeeds (consuming n tokens) if p fails and there are at least n tokens left
    def __init__(self, p, n=0, fake_name=None):
        self._n = n
        self._p = p
        self._fake_name = fake_name if fake_name is not None else ''

    def try_parse(self, components):
        if self._p.try_parse(components).succeeded or len(components) < self._n:
            return fail([self._fake_name], components)
        else:
            return ParseResult(True, components[:self._n], components[self._n:], [])

def not1(p, fake_name=None):
    return exl(not_(p, n=1, fake_name=fake_name))


# Note that not_(not_(p)) is not equivalent to p, but rather will test for p and not consume any input.




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

