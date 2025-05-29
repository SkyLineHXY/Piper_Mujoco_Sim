from typing import overload, Union, List

from spatialmath import SE3, SO3, SE2
from spatialmath.base import ArrayLike3, SE3Array

from .SO3Impl import SO3Impl


class SE3Impl(SE3):
    @overload
    def __init__(self):
        ...

    @overload
    def __init__(self, x: Union[SE3, SO3, SE2], *, check=True):  # copy/promote
        ...

    @overload
    def __init__(self, x: List[SE3], *, check=True):  # import list of SE3
        ...

    @overload
    def __init__(self, x: float, y: float, z: float, *, check=True):  # pure translation
        ...

    @overload
    def __init__(self, x: ArrayLike3, *, check=True):  # pure translation
        ...

    @overload
    def __init__(self, x: SE3Array, *, check=True):  # import native array
        ...

    @overload
    def __init__(self, x: List[SE3Array], *, check=True):  # import native arrays
        ...

    def __init__(self, x=None, y=None, z=None, *, check=True):
        super().__init__(x, y, z, check=check)

    def __add__(left, right):
        if isinstance(right, SE3):
            R = SO3Impl(left.R) + SO3Impl(right.R)
            t = left.t + right.t
            T = SE3(t) * SE3(R)
            return SE3Impl(T.A)
        return super().__add__(right)

    def __sub__(left, right):
        if isinstance(right, SE3):
            R = SO3Impl(left.R) - SO3Impl(right.R)
            t = left.t - right.t
            T = SE3(t) * SE3(R)
            return SE3Impl(T.A)
        return super().__sub__(right)

    def __mul__(left, right):
        if isinstance(right, SE3):
            return SE3Impl(super().__mul__(right.A))
        elif isinstance(right, (float, int)):
            R = SO3Impl(left.R) * right
            t = left.t * right
            T = SE3(t) * SE3(R)
            return SE3Impl(T.A)
        return super().__mul__(right)

    def __rmul__(right, left):
        if isinstance(left, SE3):
            return SE3Impl(super().__rmul__(left.A))
        elif isinstance(left, (float, int)):
            R = SO3Impl(right.R) * left
            t = right.t * left
            T = SE3(t) * SE3(R)
            return SE3Impl(T.A)
        return super().__rmul__(left)
