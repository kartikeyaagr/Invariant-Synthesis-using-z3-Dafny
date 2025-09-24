# nonlinear_solutions_z3.py
from z3 import *

def find_all_solutions():
    x = Int('x')
    y = Int('y')
    s = Solver()
    s.add(x*x + y*y == 25)
    s.add(x + y == 7)
    s.add(x > 0, y > 0)

    solutions = []
    while s.check() == sat:
        m = s.model()
        xv = m[x].as_long()
        yv = m[y].as_long()
        solutions.append((xv, yv))
        # block found solution
        s.add(Or(x != xv, y != yv))
    return solutions

if __name__ == "__main__":
    sols = find_all_solutions()
    print("Integer solutions (x,y) with x^2 + y^2 = 25, x+y=7, x>0,y>0:")
    for sol in sols:
        print(sol)
