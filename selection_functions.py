def kirkpatrick_selection(x_A, d=1):
    return d * x_A / (1 + (d - 1) * x_A)

def seger_selection(x_A, d=0):
    return x_A * (1 + d * (1 - x_A))
