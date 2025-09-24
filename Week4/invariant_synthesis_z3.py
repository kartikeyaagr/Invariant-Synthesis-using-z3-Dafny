# invariant_synthesis_z3.py
from z3 import *

def synthesize_linear_invariants(n=10, coeff_bound=6, max_sol=10):
    # search integer coefficients a,b,c in [-coeff_bound, coeff_bound]
    a = Int('a')
    b = Int('b')
    c = Int('c')

    s = Solver()

    # restrict coefficients to small range to make search finite
    s.add(a >= -coeff_bound, a <= coeff_bound)
    s.add(b >= -coeff_bound, b <= coeff_bound)
    s.add(c >= -coeff_bound, c <= coeff_bound)

    # For each reachable i in 0..n, sum == i*(i-1)/2 must satisfy a*i + b*sum + c <= 0
    constraints = []
    for i_val in range(0, n+1):
        sum_expr = (i_val * (i_val - 1)) // 2
        constraints.append(a * i_val + b * sum_expr + c <= 0)

    s.add(constraints)

    solutions = []
    while s.check() == sat and len(solutions) < max_sol:
        m = s.model()
        aval, bval, cval = m[a].as_long(), m[b].as_long(), m[c].as_long()
        solutions.append((aval, bval, cval))
        # block this solution to get others
        s.add(Or(a != aval, b != bval, c != cval))
    return solutions

def check_implication(n, coeff1, coeff2):
    # check if coeff1 implies coeff2 on reachable states i in 0..n
    a1,b1,c1 = coeff1
    a2,b2,c2 = coeff2
    s = Solver()
    # look for a reachable i violating implication: a1*i + b1*sum + c1 <= 0 AND NOT(a2*i + b2*sum + c2 <= 0)
    found = False
    for i in range(0, n+1):
        sum_expr = (i * (i - 1)) // 2
        s.push()
        s.add(a1 * i + b1 * sum_expr + c1 <= 0)
        s.add(Not(a2 * i + b2 * sum_expr + c2 <= 0))
        if s.check() == sat:
            found = True
        s.pop()
        if found:
            break
    return not found  # True => coeff1 implies coeff2 over reachable states

if __name__ == "__main__":
    n = 10
    sols = synthesize_linear_invariants(n=n, coeff_bound=6, max_sol=20)
    print(f"Found {len(sols)} linear invariants for n={n} (a,b,c):")
    for s in sols:
        print("  ", s)

    # demonstrate comparing two solutions
    if len(sols) >= 2:
        s1 = sols[0]; s2 = sols[1]
        print()
        print("Check if s1 implies s2 on reachable states:", check_implication(n, s1, s2))
