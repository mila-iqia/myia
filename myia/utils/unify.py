"""Unification module."""

import operator
from functools import reduce
from typing import \
    Any, Callable, Dict, Iterable, List, Set, TypeVar, Union

from . import overload

T = TypeVar('T')

EquivT = Dict['Var', Any]
FnFiltT = Callable[[Any], bool]
FilterT = Union[Iterable, FnFiltT]


class UnificationError(Exception):
    """Exception raised for errors in unification."""


_ID = 0


def _get_next_tag():
    global _ID
    _ID += 1
    return f"_{_ID}"


class Var:
    """Basic universal variable type."""

    def __init__(self, tag: str = None) -> None:
        """Optionally set a tag."""
        if tag is not None:
            self.tag = tag

    def matches(self, value) -> bool:
        """Return True if the variable matches the value given.

        Note that this relation is transitive, but not associative.
        """
        return True

    def intersection(self, v):
        """Return the intersection with the given variable.

        Returns:
            * A variable that matches only the values both self and
              v match.
            * `False` if it can be proven that self and v are mutually
              exclusive.
            * NotImplemented, in which case one may try
              `v.intersection(self)`.

        """
        return NotImplemented

    def ensure_tag(self) -> None:
        """Make sure that tag is set."""
        if not hasattr(self, 'tag'):
            self.tag = _get_next_tag()

    def __str__(self) -> str:
        self.ensure_tag()
        return self.tag

    def __repr__(self) -> str:
        self.ensure_tag()
        return f"Var({self.tag})"


class Seq(tuple):
    """Class to mark sequence of values matched by an SVar."""

    def __repr__(self) -> str:
        return "Seq" + super().__repr__()


class SVar(Var):
    """Variable to represent a variable length of values."""

    __slots__ = ('subtype',)

    def __init__(self, subtype: Var = None) -> None:
        """Create an SVar."""
        if subtype is None:
            subtype = Var()
        self.subtype = subtype

    def matches(self, value) -> bool:
        """Check if the provided value matches the SVar."""
        if isinstance(value, SVar):
            return self.subtype.matches(value.subtype)
        elif isinstance(value, Seq):
            return all(self.subtype.matches(v) for v in value)
        else:
            return False

    def __str__(self) -> str:
        self.ensure_tag()
        return f"*{self.tag}"

    def __repr__(self) -> str:
        self.ensure_tag()
        return f"SVar({self.tag})"


class UnionVar(Var):
    """Variable for a possible set of values."""

    __slots__ = ('values',)

    def __init__(self, values: Iterable) -> None:
        """Create a UnionVar."""
        self.values = set(values)

    def matches(self, value) -> bool:
        """Engine bypasses this."""
        raise UnificationError("matches called on a UnionVar")

    def __repr__(self) -> str:
        self.ensure_tag()
        return f"UnionVar({self.tag}, {self.values})"


class RestrictedVar(Var):
    """Variable restricted to a set of values."""

    __slots__ = ('legal_values',)

    def __init__(self, legal_values: Iterable) -> None:
        """Create a RestrictedVar."""
        self.legal_values = tuple(legal_values)

    def matches(self, value) -> bool:
        """Return True if the variable matches the value."""
        if isinstance(value, RestrictedVar):
            return all(v in self.legal_values for v in value.legal_values)
        return value in self.legal_values

    def intersection(self, v):
        """Return the intersection of two RestrictedVars.

        The resulting variable's legal values are the intersection
        of self and v's legal values.
        """
        if isinstance(v, RestrictedVar):
            lv = set(self.legal_values)
            lv2 = set(v.legal_values)
            common = lv & lv2
            if common == lv:
                return self
            elif common == lv2:
                return v
            elif common:
                return RestrictedVar(common)
            else:
                return False
        else:
            return NotImplemented

    def __repr__(self) -> str:
        self.ensure_tag()
        return f"RestrictedVar({self.tag}, {self.legal_values})"


class PredicateSet:
    """Set of predicates.

    Attributes:
        predicates: A set of callable predicates.

    """

    def __init__(self, *predicates):
        """Initialize a PredicateSet."""
        self.predicates: Set = set()
        for p in predicates:
            if isinstance(p, PredicateSet):
                self.predicates |= p.predicates
            else:
                self.predicates.add(p)

    def __eq__(self, v):
        return isinstance(v, PredicateSet) \
            and self.predicates == v.predicates

    def __call__(self, x):
        """Returns the conjunction of all predicates."""
        return all(p(x) for p in self.predicates)

    def __str__(self):
        def _str(x):
            if hasattr(x, '__name__'):
                return x.__name__
            else:
                return str(x)  # pragma: no cover
        return '&'.join(map(_str, self.predicates))


class FilterVar(Var):
    """Variable restricted to values that pass a filter function."""

    def __init__(self, filter: FnFiltT) -> None:
        """Create a FilterVar."""
        self.filter = filter

    def matches(self, value) -> bool:
        """Return True if the variable matches the value."""
        if isinstance(value, RestrictedVar):
            return all(self.filter(v) for v in value.legal_values)
        if isinstance(value, FilterVar):
            return self.filter == value.filter
        return self.filter(value)

    def intersection(self, v):
        """Return the intersection of two FilterVars.

        The resulting variable tests that both self and v's filters
        return true.
        """
        if isinstance(v, FilterVar):
            if self.filter == v.filter:
                return self
            return FilterVar(PredicateSet(self.filter, v.filter))
        else:
            return NotImplemented

    def __repr__(self) -> str:
        self.ensure_tag()
        return f"FilterVar({self.tag}, {PredicateSet(self.filter)})"


def var(filter: FilterT = None)-> Var:
    """Create a variable for unification purposes.

    Arguments:
        tag: An identifier for the variable. Two variables with the
            same filter and identifier will return the same object.
        filter: A predicate, or a set of values the variable is
            allowed to take.
    """
    if callable(filter):
        return FilterVar(filter)
    elif filter is not None:
        return RestrictedVar(filter)
    else:
        return Var()


def svar(subtype: Var = None) -> SVar:
    """Create an SVar (can match 0 or more items).

    Items must match the subtype.
    """
    return SVar(subtype)


def uvar(values: Iterable) -> UnionVar:
    """Create a UnionVar (represents multiple possibilities)."""
    return UnionVar(values)


def expandlist(lst: Iterable[T]) -> List[T]:
    """Flatten the Seq instances in a sequence."""
    lst = list(lst)
    off = 0
    for i, e in enumerate(list(lst)):
        if isinstance(e, Seq):
            lst[off + i:off + i + 1] = e
            off += len(e) - 1
    return lst


def noseq(fn: Callable[[T], T], u: T) -> T:
    """Make sure that there are no Seq in the value."""
    um = fn(u)
    if isinstance(um, Seq):
        raise TypeError("Multiple values in single-value position")
    return um


@overload(fallback_method='__visit__')
def default_visit(value: (list, tuple), fn):
    xs = expandlist(fn(x) for x in value)
    return type(value)(xs)


@overload  # noqa: F811
def default_visit(value: dict, fn):
    return {k: fn(v) for k, v in sorted(value.items())}


class VisitError(Exception):
    """Report unvisitable object."""


class Unification:
    """Unification engine."""

    def __init__(self, visitors=None, eq=operator.eq):
        """Create a unification engine.

        New visitors can be added using the decorator
        `@unif.visitors.register`
        """
        self.visitors = visitors or default_visit.variant()
        self.eq = eq

    def visit(self, fn: Callable[[T], T], value: T) -> T:
        """Apply `fn` to each element of `value` and return the result."""
        try:
            visit = self.visitors.map[type(value)]
        except KeyError as e:
            raise VisitError
        return visit(value, fn)

    def clone(self, v: T, copy_map: Dict = None) -> T:
        """Return a copy of a templated type structure.

        Type are passed through without modification, variables are
        duplicated with a new id.

        This preserves relationships between variables like this::

            clone(Tuple(v1, v1)) -> Tuple(v2, v2)

        Arguments:
            v: expression
            copy_map: Dictionary of variable mappings

        """
        if copy_map is None:
            copy_map = {}
        return self._clone(v, copy_map)

    def _clone(self, v: T, copy_map: Dict) -> T:
        if v in copy_map:
            return copy_map[v]

        if isinstance(v, Var):
            if isinstance(v, SVar):
                copy_map[v] = SVar()
            elif isinstance(v, UnionVar):
                copy_map[v] = UnionVar(self._clone(v, copy_map)
                                       for v in v.values)
            elif isinstance(v, FilterVar):
                copy_map[v] = FilterVar(v.filter)
            elif isinstance(v, RestrictedVar):
                copy_map[v] = RestrictedVar(v.legal_values)
            else:
                copy_map[v] = Var()

        elif isinstance(v, Seq):
            copy_map[v] = Seq(self._clone(val, copy_map) for val in v)

        else:
            try:
                copy_map[v] = self.visit(lambda v: self._clone(v, copy_map), v)
            except VisitError:
                return v

        return copy_map[v]

    def unify_union(self, w: UnionVar, v, equiv: EquivT) -> EquivT:
        """Perform UnionVar unification.

        This is called as required from `unify_raw`, but can also be
        called directly.
        """
        ok: Dict[Any, EquivT] = dict()
        for vw in w.values:
            equiv_copy = dict(equiv)
            # try each of the possible alternatives
            try:
                ok[vw] = self.unify_raw(vw, v, equiv_copy)
            except UnificationError:
                pass
        if len(ok) == 0:
            # if none match we fail
            raise UnificationError("All values unmatched for UnionVar")
        elif len(ok) == 1:
            # if there is a single match, we record it as the union value
            equiv.update(ok.popitem()[1])
            return equiv
        else:
            # if there were multiple matches, we try to find the
            # differences between them and see if we can find
            # common ground.

            # Essentially we are looking for match differences and
            # those differences should at most be for a single variable.

            # get the set of keys in common between all matches
            initial_keys = set(equiv.keys())
            ok_values = [v for v in ok.values()]
            comm_keys = [set(en.keys()) - initial_keys for en in ok_values]
            common_ground = reduce(lambda x, v: x & v, comm_keys)

            # check if common keys have the same values
            rej = set()
            for k in common_ground:
                if any(okv[k] != ok_values[0][k] for okv in ok_values):
                    rej.add(k)
            common_ground = common_ground - rej

            # get the difference between matches
            common_diffs = map(lambda x: x - common_ground, comm_keys)
            common_diff = reduce(lambda x, v: x & v, common_diffs)
            if len(common_diff) != 1:
                raise UnificationError("More than one match difference "
                                       "for UnionVar")
            diff = common_diff.pop()
            assert not isinstance(diff, UnionVar)

            # lift the UnionVar
            equiv[diff] = UnionVar(set(okv[diff] for okv in ok_values))
            return equiv

    def _getvar(self, v):
        return getattr(v, '__var__', v)

    def unify_raw(self, w, v, equiv: EquivT) -> EquivT:
        """'raw' interface for unification.

        The `equiv` argument is modified in-place.

        Arguments:
            w: An expression
            v: An expression
            equiv: A dictionary of variable equivalences.

        Returns:
            The equivalence dictionary

        Raises:
            UnificationError
                If the expressions are not compatible.  The dictionary may
                contain partial matching in this case and should no longer
                be used for further unification.

        Note:
            There must not be loops in the equivalence relationships
            described by `equiv` or this function might never return.

        """
        w = self._getvar(w)
        v = self._getvar(v)

        while isinstance(w, Var) and w in equiv:
            w = equiv[w]
        while isinstance(v, Var) and v in equiv:
            v = equiv[v]

        if self.eq(w, v):
            return equiv

        if isinstance(w, UnionVar):
            return self.unify_union(w, v, equiv)

        if isinstance(v, UnionVar):
            return self.unify_union(v, w, equiv)

        if isinstance(v, Var) and isinstance(w, Var):
            u = v.intersection(w)
            if u is NotImplemented:
                u = w.intersection(v)
            if u is False:
                raise UnificationError("Incompatible variables")
            if u is not NotImplemented:
                assert isinstance(u, Var)
                if u is not v:
                    equiv[v] = u
                if u is not w:
                    equiv[w] = u
                return equiv

        if isinstance(w, Var):
            if w.matches(v):
                equiv[w] = v
                return equiv

        if isinstance(v, Var):
            if v.matches(w):
                equiv[v] = w
                return equiv

        if type(v) != type(w):
            raise UnificationError("Type match error")

        if isinstance(v, Seq) and isinstance(w, Seq):
            values_v = list(v)
            values_w = list(w)
        else:
            def appender(l):
                def fn(u):
                    l.append(self._getvar(u))
                    return u
                return fn
            try:
                values_v = []
                self.visit(appender(values_v), v)
                values_w = []
                self.visit(appender(values_w), w)
            except VisitError:
                raise UnificationError("Cannot visit elements")

        sv = -1
        sw = -1

        for i, vv in enumerate(values_v):
            if isinstance(vv, SVar):
                if sv != -1:
                    raise UnificationError("Multiple SVars in sequence")
                sv = i

        for i, vw in enumerate(values_w):
            if isinstance(vw, SVar):
                if sw != -1:
                    raise UnificationError("Multiple SVars in sequence")
                sw = i

        if sv != -1 and sw != -1:
            if len(values_v) == len(values_w) and sv == sw:
                self.unify_raw(values_w[sw], values_v[sv], equiv)
                values_v.pop(sv)
                values_w.pop(sw)
            else:
                raise UnificationError("SVars in both sides of the match")

        if sv != -1 and len(values_w) >= len(values_v) - 1:
            wb = values_w[:sv]
            diff = len(values_w) - len(values_v) + 1
            wm = Seq(values_w[sv:sv+diff])
            we = values_w[sv+diff:]
            values_w = wb + [wm] + we

        if sw != -1 and len(values_v) >= len(values_w) - 1:
            vb = values_v[:sw]
            diff = len(values_v) - len(values_w) + 1
            vm = Seq(values_v[sw:sw+diff])
            ve = values_v[sw+diff:]
            values_v = vb + [vm] + ve

        if len(values_w) != len(values_v):
            raise UnificationError("Structures of differing size")

        for wi, vi in zip(values_w, values_v):
            equiv = self.unify_raw(wi, vi, equiv)

        return equiv

    def unify(self, w, v, equiv: EquivT = None) -> EquivT:
        """Unify two expressions.

        After a match is found, this will post-process the dictionary to
        set all the equivalences to their transitive values.

        Arguments:
            w: expression
            v: expression
            equiv: Dictionary of pre-existing equivalences.

        Returns:
            The equivalence dictionary if a match is found,
            None otherwise.

        Note:
            There must not be loops in the equivalence relationships
            described by `equiv` or this function will never return.

        """
        if equiv is None:
            equiv = {}
        try:
            equiv = self.unify_raw(w, v, equiv)
        except UnificationError:
            return None

        # Set all keys to their transitive values
        ks = set(equiv.keys())
        for k in ks:
            init_k = k
            while k in equiv:
                k = equiv[k]
                equiv[init_k] = k

        return equiv

    def reify(self, v, equiv: EquivT) -> Any:
        """Fill in a expression according to the equivalences given.

        Arguments:
            v: expression
            equiv: equivalence relationships (transitively mapped)

        Note:
            This expects the dictionary of equivalences to be transitively
            transformed to work properly.  `unify` does this automatically
            on its return value, but if you use unify_raw directly, you
            have to take care of this.

        """
        v = self._getvar(v)
        if v in equiv:
            return equiv[v]

        try:
            return self.visit(lambda u: self.reify(u, equiv), v)
        except VisitError:
            return v
