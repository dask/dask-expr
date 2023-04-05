import functools
import numbers
import operator
from collections import defaultdict
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import pandas as pd
import toolz
from dask.base import normalize_token, tokenize
from dask.dataframe import methods
from dask.optimization import SubgraphCallable
from dask.utils import M, apply, funcname
from matchpy import (
    Arity,
    CustomConstraint,
    Operation,
    Pattern,
    ReplacementRule,
    Wildcard,
    replace_all,
)
from matchpy.expressions.expressions import _OperationMeta

replacement_rules = []


class _ExprMeta(_OperationMeta):
    """Metaclass to determine Operation behavior

    Matchpy overrides `__call__` so that `__init__` doesn't behave as expected.
    This is gross, but has some logic behind it.  We need to enforce that
    expressions can be easily replayed.  Eliminating `__init__` is one way to
    do this.  It forces us to compute things lazily rather than at
    initialization.

    We motidify Matchpy's implementation so that we can handle keywords and
    default values more cleanly.

    We also collect replacement rules here.
    """

    seen = set()

    def __call__(cls, *args, variable_name=None, **kwargs):
        # Collect optimization rules for new classes
        if cls not in _ExprMeta.seen:
            _ExprMeta.seen.add(cls)
            for rule in cls._replacement_rules():
                replacement_rules.append(rule)

        # Grab keywords and manage default values
        operands = list(args)
        for parameter in cls._parameters[len(operands) :]:
            operands.append(kwargs.pop(parameter, cls._defaults[parameter]))
        assert not kwargs

        # Defer up to matchpy
        return super().__call__(*operands, variable_name=None)


_defer_to_matchpy = False


class Expr(Operation, metaclass=_ExprMeta):
    """Primary class for all Expressions

    This mostly includes Dask protocols and various Pandas-like method
    definitions to make us look more like a DataFrame.
    """

    commutative = False
    associative = False
    _parameters = []
    _defaults = {}

    @classmethod
    def _replacement_rules(cls) -> Iterator[ReplacementRule]:
        """Rules associated to this class that are useful for optimization

        See also:
            optimize_expr
            _ExprMeta
        """
        yield from []

    def __str__(self):
        s = ", ".join(
            str(param) + "=" + str(operand)
            for param, operand in zip(self._parameters, self.operands)
            if operand != self._defaults.get(param)
        )
        return f"{type(self).__name__}({s})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self._name)

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError as err:
            # Allow operands to be accessed as attributes
            # as long as the keys are not already reserved
            # by existing methods/properties
            _parameters = type(self)._parameters
            if key in _parameters:
                idx = _parameters.index(key)
                return self.operands[idx]
            raise err

    def operand(self, key):
        # Access an operand unambiguously
        # (e.g. if the key is reserved by a method/property)
        return self.operands[type(self)._parameters.index(key)]

    @property
    def index(self):
        return ProjectIndex(self)

    @property
    def size(self):
        return Size(self)

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)

    def __getitem__(self, other):
        if isinstance(other, Expr):
            return Filter(self, other)  # df[df.x > 1]
        else:
            return Projection(self, other)  # df[["a", "b", "c"]]

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __truediv__(self, other):
        return Div(self, other)

    def __rtruediv__(self, other):
        return Div(other, self)

    def __lt__(self, other):
        return LT(self, other)

    def __rlt__(self, other):
        return LT(other, self)

    def __gt__(self, other):
        return GT(self, other)

    def __rgt__(self, other):
        return GT(other, self)

    def __le__(self, other):
        return LE(self, other)

    def __rle__(self, other):
        return LE(other, self)

    def __ge__(self, other):
        return GE(self, other)

    def __rge__(self, other):
        return GE(other, self)

    def __eq__(self, other):
        if _defer_to_matchpy:  # Defer to matchpy when optimizing
            return Operation.__eq__(self, other)
        else:
            return EQ(other, self)

    def __ne__(self, other):
        if _defer_to_matchpy:  # Defer to matchpy when optimizing
            return Operation.__ne__(self, other)
        else:
            return NE(other, self)

    def sum(self, skipna=True, level=None, numeric_only=None, min_count=0):
        return Sum(self, skipna, level, numeric_only, min_count)

    def mean(self, skipna=True, level=None, numeric_only=None, min_count=0):
        return self.sum(skipna=skipna) / self.count()

    def max(self, skipna=True, level=None, numeric_only=None, min_count=0):
        return Max(self, skipna, level, numeric_only, min_count)

    def mode(self, dropna=True):
        return Mode(self, dropna=dropna)

    def min(self, skipna=True, level=None, numeric_only=None, min_count=0):
        return Min(self, skipna, level, numeric_only, min_count)

    def count(self, numeric_only=None):
        return Count(self, numeric_only)

    def astype(self, dtypes):
        return AsType(self, dtypes)

    def apply(self, function, *args, **kwargs):
        return Apply(self, function, args, kwargs)

    @property
    def divisions(self):
        return tuple(self._divisions())

    def _divisions(self):
        raise NotImplementedError()

    @property
    def known_divisions(self):
        """Whether divisions are already known"""
        return len(self.divisions) > 0 and self.divisions[0] is not None

    @property
    def npartitions(self):
        if "npartitions" in self._parameters:
            idx = self._parameters.index("npartitions")
            return self.operands[idx]
        else:
            return len(self.divisions) - 1

    @property
    def _name(self):
        return funcname(type(self)).lower() + "-" + tokenize(*self.operands)

    @property
    def columns(self):
        return self._meta.columns

    @property
    def dtypes(self):
        return self._meta.dtypes

    @property
    def _meta(self):
        raise NotImplementedError()

    def __dask_graph__(self):
        """Traverse expression tree, collect layers"""
        stack = [self]
        seen = set()
        layers = []
        while stack:
            expr = stack.pop()

            if expr._name in seen:
                continue
            seen.add(expr._name)

            layers.append(expr._layer())
            for operand in expr.operands:
                if isinstance(operand, Expr):
                    stack.append(operand)

        return toolz.merge(layers)

    def __dask_keys__(self):
        return [(self._name, i) for i in range(self.npartitions)]


class Blockwise(Expr):
    """Super-class for block-wise operations

    This is fairly generic, and includes definitions for `_meta`, `divisions`,
    `_layer` that are often (but not always) correct.  Mostly this helps us
    avoid duplication in the future.
    """

    operation = None

    @property
    def _meta(self):
        return self.operation(
            *[arg._meta if isinstance(arg, Expr) else arg for arg in self.operands]
        )

    @property
    def _kwargs(self):
        return {}

    def _divisions(self):
        # This is an issue.  In normal Dask we re-divide everything in a step
        # which combines divisions and graph.
        # We either have to create a new Align layer (ok) or combine divisions
        # and graph into a single operation.
        first = [o for o in self.operands if isinstance(o, Expr)][0]
        assert all(
            arg.divisions == first.divisions
            for arg in self.operands
            if isinstance(arg, Expr)
        )
        return first.divisions

    @property
    def _name(self):
        return funcname(self.operation) + "-" + tokenize(*self.operands)

    def _subgraph_dependencies(self):
        # List `Expr` operands only.
        # This property is required to enable fusion
        return [operand for operand in self.operands if isinstance(operand, Expr)]

    def _block_subgraph(self):
        return {
            self._name: (
                apply,
                self.operation,
                [
                    operand._name if isinstance(operand, Expr) else operand
                    for operand in self.operands
                ],
                self._kwargs,
            )
        }

    def _layer(self):
        return _subgraph_callable_layer(
            self._name,
            self._block_subgraph(),
            self._subgraph_dependencies(),
            self.npartitions,
        )


class Elemwise(Blockwise):
    """
    This doesn't really do anything, but we anticipate that future
    optimizations, like `len` will care about which operations preserve length
    """

    pass


class AsType(Elemwise):
    """A good example of writing a trivial blockwise operation"""

    _parameters = ["frame", "dtypes"]
    operation = M.astype


class Apply(Elemwise):
    """A good example of writing a less-trivial blockwise operation"""

    _parameters = ["frame", "function", "args", "kwargs"]
    _defaults = {"args": (), "kwargs": {}}
    operation = M.apply

    @property
    def _meta(self):
        return self.frame._meta.apply(self.function, *self.args, **self.kwargs)

    def _block_subgraph(self):
        return {
            self._name: (
                apply,
                M.apply,
                [self.frame._name, self.function] + list(self.args),
                self.kwargs,
            )
        }


class Assign(Elemwise):
    """Column Assignment"""

    _parameters = ["frame", "key", "value"]
    operation = methods.assign

    @property
    def _meta(self):
        return self.frame._meta.assign(**{self.key: self.value._meta})

    def _block_subgraph(self):
        return {
            self._name: (
                methods.assign,
                self.frame._name,
                self.key,
                self.value._name,
            )
        }


class Filter(Blockwise):
    _parameters = ["frame", "predicate"]
    operation = operator.getitem

    @classmethod
    def _replacement_rules(self):
        df = Wildcard.dot("df")
        condition = Wildcard.dot("condition")
        columns = Wildcard.dot("columns")

        # Project columns down through dataframe
        # df[df.x > 1].y -> df.y[df.x > 1]
        yield ReplacementRule(
            Pattern(Filter(df, condition)[columns]),
            lambda df, condition, columns: df[columns][condition],
        )


class Projection(Elemwise):
    """Column Selection"""

    _parameters = ["frame", "columns"]
    operation = operator.getitem

    def _divisions(self):
        return self.frame.divisions

    @property
    def columns(self):
        if isinstance(self.operand("columns"), list):
            return pd.Index(self.operand("columns"))
        else:
            return self.operand("columns")

    @property
    def _meta(self):
        return self.frame._meta[self.columns]

    def _block_subgraph(self):
        return {self._name: (operator.getitem, self.frame._name, self.columns)}

    def __str__(self):
        base = str(self.frame)
        if " " in base:
            base = "(" + base + ")"
        return f"{base}[{repr(self.columns)}]"


class ProjectIndex(Elemwise):
    """Column Selection"""

    _parameters = ["frame"]
    operation = getattr

    def _divisions(self):
        return self.frame.divisions

    @property
    def _meta(self):
        return self.frame._meta.index

    def _block_subgraph(self):
        return {self._name: (getattr, self.frame._name, "index")}


class Head(Expr):
    _parameters = ["frame", "n"]
    _defaults = {"n": 5}

    @property
    def _meta(self):
        return self.frame._meta

    def _divisions(self):
        return self.frame.divisions[:2]

    def _layer(self):
        return {
            (self._name, 0): (M.head, (self.frame._name, 0), self.n),
        }


class Binop(Elemwise):
    _parameters = ["left", "right"]
    arity = Arity.binary

    def _block_subgraph(self):
        return {
            self._name: (
                self.operation,
                self.left._name if isinstance(self.left, Expr) else self.left,
                self.right._name if isinstance(self.right, Expr) else self.right,
            )
        }

    def __str__(self):
        return f"{self.left} {self._operator_repr} {self.right}"

    @classmethod
    def _replacement_rules(cls):
        left = Wildcard.dot("left")
        right = Wildcard.dot("right")
        columns = Wildcard.dot("columns")

        # Column Projection
        def transform(left, right, columns, cls=None):
            if isinstance(left, Expr):
                left = left[columns]  # TODO: filter just the correct columns

            if isinstance(right, Expr):
                right = right[columns]

            return cls(left, right)

        # (a + b)[c] -> a[c] + b[c]
        yield ReplacementRule(
            Pattern(cls(left, right)[columns]), functools.partial(transform, cls=cls)
        )


class Add(Binop):
    operation = operator.add
    _operator_repr = "+"

    @classmethod
    def _replacement_rules(cls):
        x = Wildcard.dot("x")
        yield ReplacementRule(
            Pattern(Add(x, x)),
            lambda x: Mul(2, x),
        )

        yield from super()._replacement_rules()


class Sub(Binop):
    operation = operator.sub
    _operator_repr = "-"


class Mul(Binop):
    operation = operator.mul
    _operator_repr = "*"

    @classmethod
    def _replacement_rules(cls):
        a, b, c = map(Wildcard.dot, "abc")
        yield ReplacementRule(
            Pattern(
                Mul(a, Mul(b, c)),
                CustomConstraint(
                    lambda a, b, c: isinstance(a, numbers.Number)
                    and isinstance(b, numbers.Number)
                ),
            ),
            lambda a, b, c: Mul(a * b, c),
        )

        yield from super()._replacement_rules()


class Div(Binop):
    operation = operator.truediv
    _operator_repr = "/"


class LT(Binop):
    operation = operator.lt
    _operator_repr = "<"


class LE(Binop):
    operation = operator.le
    _operator_repr = "<="


class GT(Binop):
    operation = operator.gt
    _operator_repr = ">"


class GE(Binop):
    operation = operator.ge
    _operator_repr = ">="


class EQ(Binop):
    operation = operator.eq
    _operator_repr = "=="


class NE(Binop):
    operation = operator.ne
    _operator_repr = "!="


@normalize_token.register(Expr)
def normalize_expression(expr):
    return expr._name


def optimize_expr(expr, fuse=True):
    """High level query optimization

    Today we just use MatchPy's term rewriting system, leveraging the
    replacement rules found in the `replacement_rules` global list .  We continue
    rewriting until nothing changes.  The `replacement_rules` list can be added
    to by anyone, but is mostly populated by the various `_replacement_rules`
    methods on the Expr subclasses.

    Note: matchpy expects `__eq__` and `__ne__` to work in a certain way during
    matching.  This is a bit of a hack, but we turn off our implementations of
    `__eq__` and `__ne__` when this function is running using the
    `_defer_to_matchpy` global.  Please forgive us our sins, as we forgive
    those who sin against us.
    """
    last = None
    global _defer_to_matchpy

    _defer_to_matchpy = True  # take over ==/!= when optimizing
    try:
        while str(expr) != str(last):
            last = expr
            expr = replace_all(expr, replacement_rules)
    finally:
        _defer_to_matchpy = False

    if fuse:
        # Apply fusion
        expr = _blockwise_fusion(expr)

    return expr


from dask_match.reductions import Count, Max, Min, Mode, Size, Sum


## Utilites for Expr fusion


def replace_nodes(expr, nodes_to_replace: dict):
    """Replace specific nodes in an Expr tree

    `nodes_to_replace` corresponds to a `dict` mapping
    of old `Expr` objects to new `Expr` objects.
    """
    if not nodes_to_replace:
        # Nothing to replace
        return expr
    elif expr in nodes_to_replace:
        # Already at targetted expr
        return nodes_to_replace[expr]

    new_operands = []
    new_count = 0
    for operand in expr.operands:
        if isinstance(operand, Expr) and operand in nodes_to_replace:
            # Replacing this operand
            val = nodes_to_replace[operand]
            new_count += 1
        elif isinstance(operand, Expr):
            # Non-Expr operand - Recursive call
            val = replace_nodes(operand, nodes_to_replace)
            if operand._name != val._name:
                new_count += 1
        else:
            # Non-Expr operand
            val = operand
        new_operands.append(val)

    if new_count:
        # Only return new object if something changed
        return type(expr)(*new_operands)
    return expr


def _blockwise_fusion(expr):
    """Traverse the expression graph and apply fusion"""

    def _fusion_pass(expr):
        # Full pass to find global dependencies
        seen = set()
        stack = [expr]
        dependents = defaultdict(set)
        dependencies = {}
        while stack:
            next = stack.pop()

            if next._name in seen:
                continue
            seen.add(next._name)

            if isinstance(next, Fusable):
                dependencies[next] = set()
                if next not in dependents:
                    dependents[next] = set()

            for operand in next.operands:
                if isinstance(operand, Expr):
                    stack.append(operand)
                    if isinstance(operand, Fusable):
                        if next in dependencies:
                            dependencies[next].add(operand)
                        dependents[operand].add(next)

        # Traverse each "root" until we find a fusable sub-group.
        # Here we use root to refer to a Fusable Expr node that
        # has no Fusable dependents
        roots = [
            k
            for k, v in dependents.items()
            if v == set() or all(not isinstance(_expr, Fusable) for _expr in v)
        ]
        while roots:
            root = roots.pop()
            seen = set()
            stack = [root]
            group = []
            while stack:
                next = stack.pop()

                if next._name in seen:
                    continue
                seen.add(next._name)

                group.append(next)
                for dep in dependencies[next]:
                    if not (dependents[dep] - set(stack) - set(group)):
                        # All of deps dependents are contained
                        # in the local group (or the local stack
                        # of expr nodes that we know we will be
                        # adding to the local group)
                        stack.append(dep)
                    elif dep not in roots and dependencies[dep]:
                        # Couldn't fuse dep, but we may be able to
                        # use it as a new root on the next pass
                        roots.append(dep)

            # Replace fusable sub-group
            if len(group) > 1:
                group_deps = []
                local_names = [_expr._name for _expr in group]
                for _expr in group:
                    group_deps += [
                        operand
                        for operand in _expr._subgraph_dependencies()
                        if operand._name not in local_names
                    ]
                to_replace = {group[0]: FusedExpr(group, *group_deps)}
                return replace_nodes(expr, to_replace), not roots

        # Return original expr if no fusable sub-groups were found
        return expr, True

    while True:
        original_name = expr._name
        expr, done = _fusion_pass(expr)
        if done or expr._name == original_name:
            break

    return expr


@runtime_checkable
class Fusable(Protocol):
    """Fusable Expr Protocol
    An `Expr` with these methods will be treated as "fusable"
    """

    def _subgraph_dependencies(self):
        """List of `Expr` operands or `MappedArg`
        dependencies. Since `_block_subgraph`
        cannot include mapped arguments explicitly,
        IO expressions must represent these task
        arguments as `MappedArg` dependencies.
        """
        raise NotImplementedError

    def _block_subgraph(self):
        """Return the subgraph for an abstract 'block'.
        Used to initialize a `SubgraphCallabe` object.
        """
        raise NotImplementedError


class FusedExpr(Expr):
    @property
    def exprs(self):
        return self.operands[0]

    @property
    def _meta(self):
        return self.exprs[0]._meta

    def __str__(self):
        descr = "-".join([expr._name.split("-")[0] for expr in self.exprs])
        return f"Fused-{descr}"

    @property
    def _name(self):
        return f"{str(self)}-{tokenize(self.exprs)}"

    def _divisions(self):
        return self.exprs[0]._divisions()

    def _subgraph_dependencies(self):
        return self.operands[1:]

    def _block_subgraph(self):
        block = {self._name: self.exprs[0]._name}
        for _expr in self.exprs:
            for k, v in _expr._block_subgraph().items():
                block[k] = v
        return block

    def _layer(self):
        return _subgraph_callable_layer(
            self._name,
            self._block_subgraph(),
            self._subgraph_dependencies(),
            self.npartitions,
        )


class MappedArg:
    """Indexable Expr dependency

    NOTE: This class is used by IO expressions
    to map path-like arguments over output partitions
    in a fusion-compatible way.
    """

    def __init__(self, lookup):
        assert callable(lookup) or isinstance(lookup, (list, dict))
        self.lookup = lookup

    @property
    def _name(self):
        return f"mapped-arg-{tokenize(self.lookup)}"

    def __getitem__(self, index):
        if callable(self.lookup):
            return self.lookup(index)
        return self.lookup[index]


def _subgraph_callable_layer(name, block, dependencies, npartitions, funcname=None):
    """Construct a low-level blockwise layer"""

    # Convert `_block` logic to SubgraphCallable
    func = SubgraphCallable(
        block,
        name,
        [dep._name for dep in dependencies],
        funcname or name,
    )

    # Tasks depend on operand keys only
    return {
        (name, i): (
            func,
            *[
                dep[i] if isinstance(dep, MappedArg) else (dep._name, i)
                for dep in dependencies
            ],
        )
        for i in range(npartitions)
    }
