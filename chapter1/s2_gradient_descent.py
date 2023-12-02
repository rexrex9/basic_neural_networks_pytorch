import sympy

x1 = sympy.symbols("x1")
w1 = sympy.symbols("w1")
x2 = sympy.symbols("x2")
w2 = sympy.symbols("w2")
b = sympy.symbols("b")
y = sympy.symbols("y")

y_pred = x1*w1+ x2*w2 + b
l = (y_pred-y)**2

dify = sympy.diff(l,w1)
