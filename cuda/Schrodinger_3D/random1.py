import numpy as np
import matplotlib.pyplot as plt

def quadratic(a, b, c):
    """
    Restituisce una funzione che calcola ax^2 + bx + c
    per un valore x dato.
    """
    return lambda x: a*x**2 + b*x + c

# Coefficienti dell'equazione di secondo grado
a = 2
b = 7
c = 3

# Creo la funzione quadratica f(x) = ax^2 + bx + c
f = quadratic(a, b, c)

# Definisco l'intervallo di x
x = np.linspace(-10, 10, 300)
y = f(x)

# Creo il grafico di base
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=f"{a}x^2 + {b}x + {c}")

# Aggiungo assi orizzontale e verticale (solo per riferimenti visivi)
plt.axhline(y=0, color='black', linewidth=0.8)
plt.axvline(x=0, color='black', linewidth=0.8)

# 1) Intersezione con x = 0 (asse y)
#    Qui x=0 => y = c
plt.plot(0, c, 'ro', label=f"Intersezione con x=0: (0, {c})")

# 2) Calcolo le soluzioni dell'equazione a*x^2 + b*x + c = 0
#    (intersezione con l'asse x, dove f(x) = 0)
delta = b**2 - 4*a*c

if delta > 0:
    # Due soluzioni reali distinte
    x1 = (-b + np.sqrt(delta)) / (2*a)
    x2 = (-b - np.sqrt(delta)) / (2*a)
    plt.plot(x1, 0, 'go', label=f"Soluzione 1: x={x1:.2f}")
    plt.plot(x2, 0, 'go', label=f"Soluzione 2: x={x2:.2f}")
elif delta == 0:
    # Una soluzione reale (doppia)
    x0 = -b / (2*a)
    plt.plot(x0, 0, 'go', label=f"Soluzione: x={x0:.2f}")
else:
    # Nessuna soluzione reale
    # Non tracciamo nulla, ma potremmo mostrare un messaggio
    pass

# Stile del grafico
plt.title("Grafico di una funzione di secondo grado e le sue intersezioni")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()
