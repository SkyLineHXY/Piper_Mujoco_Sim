from typing import overload, List, Union

from spatialmath import SO3, SE3
from spatialmath.base import SO3Array


class SO3Impl(SO3):
    @overload
    def __init__(self):
        ...

    @overload
    def __init__(self, arg: SO3, *, check=True):
        ...

    @overload
    def __init__(self, arg: SE3, *, check=True):
        ...

    @overload
    def __init__(self, arg: SO3Array, *, check=True):
        ...

    @overload
    def __init__(self, arg: List[SO3Array], *, check=True):
        ...

    @overload
    def __init__(self, arg: List[Union[SO3, SO3Array]], *, check=True):
        ...

    def __init__(self, arg=None, *, check=True):
        super().__init__(arg, check=check)

    def __add__(left, right):
        if isinstance(right, SO3):
            return right * left
        return super().__add__(right)

    def __sub__(left, right):
        if isinstance(right, SO3):
            return left * SO3Impl(right.inv().R)
        return super().__sub__(right)

    def __mul__(left, right):
        if isinstance(right, SO3):
            return SO3Impl(super().__mul__(right.R))
        elif isinstance(right, (float, int)):
            return SO3Impl((SO3.Exp(left.log() * right)).R)
        return super().__mul__(right)

    def __rmul__(right, left):
        if isinstance(left, (SO3, float, int)):
            return right * left
        return super().__rmul__(left)
