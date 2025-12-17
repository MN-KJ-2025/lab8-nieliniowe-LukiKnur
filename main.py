# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np
from typing import Callable


def func(x: int | float | np.ndarray) -> int | float | np.ndarray:
    
    """Funkcja wyliczająca wartości funkcji f(x).
    f(x) = e^(-2x) + x^2 - 1

    Args:
        x (int | float | np.ndarray): Argumenty funkcji.

    Returns:
        (int | float | np.ndarray): Wartości funkcji f(x).

    Raises:
        TypeError: Jeśli argument x nie jest typu `np.ndarray`, `float` lub 
            `int`.
        ValueError: Jeśli argument x nie jest jednowymiarowy.
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return np.exp(-2 * x) + x**2 - 1


def dfunc(x: np.ndarray) -> np.ndarray:
    """Funkcja wyliczająca wartości pierwszej pochodnej (df(x)) funkcji f(x).
    f(x) = e^(-2x) + x^2 - 1
    df(x) = -2 * e^(-2x) + 2x

    Args:
        x (int | float | np.ndarray): Argumenty funkcji.

    Returns:
        (int | float | np.ndarray): Wartości funkcji f(x).

    Raises:
        TypeError: Jeśli argument x nie jest typu `np.ndarray`, `float` lub 
            `int`.
        ValueError: Jeśli argument x nie jest jednowymiarowy.
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return -2 * np.exp(-2 * x) + 2 * x


def ddfunc(x: np.ndarray) -> np.ndarray:
    """Funkcja wyliczająca wartości drugiej pochodnej (ddf(x)) funkcji f(x).
    f(x) = e^(-2x) + x^2 - 1
    ddf(x) = 4 * e^(-2x) + 2

    Args:
        x (int | float | np.ndarray): Argumenty funkcji.

    Returns:
        (int | float | np.ndarray): Wartości funkcji f(x).

    Raises:
        TypeError: Jeśli argument x nie jest typu `np.ndarray`, `float` lub 
            `int`.
        ValueError: Jeśli argument x nie jest jednowymiarowy.
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return 4 * np.exp(-2 * x) + 2


def bisection(
    a: int | float,
    b: int | float,
    f: Callable[[float], float],
    epsilon: float,
    max_iter: int,
) -> tuple[float, int] | None:
    
    if a >= b:
        return None
    
    try:
        f_a = f(float(a))
        f_b = f(float(b))
    except Exception as e:
        return None
    
    if f_a * f_b > 0:
        return None

    if abs(f_a) < epsilon:
        return (float(a), 0)
    if abs(f_b) < epsilon:
        return (float(b), 0)
    
    iter_count = 0
    current_a = float(a)
    current_b = float(b)

    for i in range(1, max_iter + 1):
        iter_count = i
        
        c = (current_a + current_b) / 2.0
        
        try:
            
            f_c = f(c)
        except Exception as e:
            return None

        if abs(f_c) < epsilon:
            return (c, iter_count)
        
        if f_a * f_c < 0:
            current_b = c

        elif f_c * f_b < 0:
            current_a = c

        else:
            return (c, iter_count)
        
    c_final = (current_a + current_b) / 2.0
    print(f"Maksymalna liczba iteracji ({max_iter}). Wynik może nie być idealnie dokładny.")
    return (c_final, max_iter)        


def secant(
    a: int | float,
    b: int | float,
    f: Callable[[float], float],
    epsilon: float,
    max_iters: int,
) -> tuple[float, int] | None:
    
    fa = f(a)
    fb = f(b)

    if fa * fb >= 0:
        return None

    i = 0
    for _ in range(max_iters):
        i += 1

        if fb == fa:
            return None

        
        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)

       
        if abs(fc) < epsilon or abs(c - b) < epsilon:
            return c, i

        
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return c, i


def difference_quotient(
    f: Callable[[float], float], x: int | float, h: int | float
) -> float | None:
    
    if h == 0:
        return None

    return (f(x + h) - f(x)) / h


def newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    ddf: Callable[[float], float],
    a: int | float,
    b: int | float,
    epsilon: float,
    max_iter: int,
) -> tuple[float, int] | None:
    
    if f(a) * ddf(a) > 0:
        x_curr = a
    elif f(b) * ddf(b) > 0:
        x_curr = b
    else:
        x_curr = (a + b) / 2

    iteration = 0
    for _ in range(max_iter):
        iteration += 1

        deriv_val = df(x_curr)
        if deriv_val == 0:
            return None

        x_new = x_curr - f(x_curr) / deriv_val

        if abs(x_new - x_curr) < epsilon or abs(f(x_new)) < epsilon:
            return x_new, iteration

        x_curr = x_new

    return x_curr, iteration
