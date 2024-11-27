"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, Optional

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiplies two numbers and returns the result.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: Product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: Unchanged input number.

    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers and returns the result.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: Addition of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negates the given number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: Negated input number.

    """
    return float(-x)


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: Returns true if x is less than y, false otherwise.

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if one number is equal to another.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: Returns true if x is equal to y, false otherwise.

    """
    return x == y


def max(x: float, y: float) -> float:
    """Returns the maximum of two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: Maximum of x and y.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close to each other.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: Returns true if the absolute difference between x and y is less than 1e-2, false otherwise.

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Calculates the sigmoid of the given number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: Sigmoid of the input number.

    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def sigmoid_back(a: float, d_output: float) -> float:
    """Compute the derivative of the sigmoid function."""
    s = sigmoid(a)
    return d_output * s * (1 - s)


def relu(x: float) -> float:
    """Calculates the rectified linear unit of the given number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: Rectified linear unit of the input number.

    """
    return max(0.0, x)


def log(x: float) -> float:
    """Calculates the natural logarithm of the given number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: Natural logarithm of the input number.

    """
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential of the given number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: Exponential of the input number.

    """
    return math.exp(x)


def exp_back(x: float, y: float) -> float:
    """Computes the derivative of exp times a second arg.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: Derivative of exp times y.

    """
    return y * exp(x)


def inv(x: float) -> float:
    """Calculates the inverse of the given number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: Inverse of the input number.

    """
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: Derivative of log times y.

    """
    return y / x


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of inv times a second arg.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: Derivative of inv times y.

    """
    return -y / (x * x)


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of relu times a second arg.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: Derivative of relu times y.

    """
    return y if x > 0.0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(func: Callable, iter: Iterable) -> Iterable:
    """Applies given function to each element of given iterable.

    Args:
    ----
        func (Callable): Given function to apply.
        iter (Iterable): Given iterable to apply function.

    Returns:
    -------
        Iterable: Result of applying given function to each element of given iterable.

    """
    return [func(x) for x in iter]


def zipWith(func: Callable, first_iter: Iterable, second_iter: Iterable) -> Iterable:
    """Applies given function to each element of given two iterables and combines them.

    Args:
    ----
        func (Callable): Function to apply.
        first_iter (Iterable): First iterable to apply function and combine.
        second_iter (Iterable): Second iterable to apply function and combine.

    Returns:
    -------
        Iterable: Result of applying given function to each element of given two iterables and combining them.

    """
    return [func(x, y) for x, y in zip(first_iter, second_iter)]


def reduce(func: Callable, lis: Iterable, initial: Optional[float] = None) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function, starting with an optional initial value.

    Args:
    ----
        func (Callable): Function to reduce iterable.
        lis (Iterable): Iterable to reduce.
        initial: Initial value to start reduction.

    Returns:
    -------
        Reduced value of iterable.

    """
    it = iter(lis)
    if initial is None:
        try:
            acc = next(it)
        except StopIteration:
            raise TypeError("reduce() of empty sequence with no initial value")
    else:
        acc = initial
    for x in it:
        acc = func(acc, x)
    return acc


def addLists(first_iter: Iterable, second_iter: Iterable) -> Iterable:
    """Adds two lists together.

    Args:
    ----
        first_iter (Iterable): First list to add.
        second_iter (Iterable): Second list to add.

    Returns:
    -------
        Iterable: Sum of two lists.

    """
    return zipWith(add, first_iter, second_iter)


def negList(iter: Iterable) -> Iterable:
    """Negates all element in a list.

    Args:
    ----
        iter (Iterable): List to negate.

    Returns:
    -------
        Iterable: Negated elements.

    """
    return map(neg, iter)


def sum(iter: Iterable) -> float:
    """Sum all elements in a list using reduce.

    Args:
    ----
        iter (Iterable): List to calculate sum.

    Returns:
    -------
        float: Summation of all elements.

    """
    return reduce(add, iter, 0.0)


def prod(iter: Iterable) -> float:
    """Take the product of all elements in a list using reduce.

    Args:
    ----
        iter (Iterable): List to calculate product.

    Returns:
    -------
        float: Product of all elements.

    """
    return reduce(mul, iter, 1.0)
