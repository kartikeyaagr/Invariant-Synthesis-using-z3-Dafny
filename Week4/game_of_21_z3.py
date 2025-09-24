# game_of_21_z3.py
from z3 import *

def solve_game_of_21(total = 21, max_take = 3):
    # Bool variables Win0..Win_total
    Win = [Bool(f"Win_{p}") for p in range(total + 1)]
    s = Solver()

    # Base case: no stones -> losing for player to move
    s.add(Win[0] == False)

    # Recurrence for p > 0
    for p in range(1, total + 1):
        moves = []
        for m in range(1, max_take + 1):
            if p - m >= 0:
                moves.append(Not(Win[p - m]))  # take m then opponent must be losing
        # if any move makes opponent lose, current is winning
        s.add(Win[p] == Or(*moves) if moves else Win[p] == False)

    if s.check() != sat:
        print("Unsat or unknown result while building board.")
        return

    # get model
    m = s.model()
    wins = {p: is_true(m.eval(Win[p])) for p in range(total + 1)}
    return wins

if __name__ == "__main__":
    wins = solve_game_of_21(21, 3)
    # print strategy
    for p in range(22):
        print(f"Pile {p:2d}: {'Winning' if wins[p] else 'Losing'}")
    print()
    print("Conclusion: First player with 21 stones can", "win" if wins[21] else "not always win")
