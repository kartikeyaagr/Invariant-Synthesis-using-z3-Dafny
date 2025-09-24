# Week 4: Z3 Problem Solving – Explanations

## Problem 1: Game of 21 (Nim Game)

The game starts with 21 objects. Players alternate removing 1–3 objects, and the player who removes the last object wins.  
The key is to avoid leaving the opponent a multiple of 4. By always taking enough objects to force the pile to be 20, 16, 12, 8, or 4 after the opponent’s move, the first player guarantees victory.  

**Conclusion:** The first player can always win by following the mod-4 strategy.

---

## Problem 2: Non-Linear Constraint Solving

We need integer solutions to:

- x² + y² = 25  
- x + y = 7, with x, y > 0

From the second equation: y = 7 - x. Substituting into the first gives x² + (7-x)² = 25. Simplifying leads to integer solutions (x=3, y=4) or (x=4, y=3).  

**Conclusion:** There are two valid integer solutions.

---

## Problem 3: Invariant Synthesis Practice

We analyze this loop:

```
i = 0; sum = 0
while (i < n):
    sum = sum + i
    i = i + 1
```

We want linear invariants of the form a·i + b·sum + c ≤ 0.  

Key observations:  
- i ≥ 0 always holds.  
- sum ≥ 0 always holds.  
- A stronger invariant: 2·sum ≥ i·(i - 1), since sum accumulates triangular numbers.  

**Conclusion:** Linear invariants like i ≥ 0, sum ≥ 0, and bounds related to triangular growth characterize the loop.

---
