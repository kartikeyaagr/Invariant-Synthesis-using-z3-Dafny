// Benchmark 5: Two-Phase Loop
// Phase 1: x increases, y stays 0
// Phase 2: x stays at mid, y increases
// Disjunctive invariant: (y == 0 && x <= mid) || (x == mid && y >= 0)

method TwoPhaseLoop(n: int) returns (x: int, y: int)
  requires n >= 0
  ensures x == n
  ensures y == n
{
  x := 0;
  y := 0;
  var mid := n;
  
  // Phase 1
  while x < mid
    // invariant x >= 0
    // invariant y == 0
    // invariant x <= mid
  {
    x := x + 1;
  }
  
  // Phase 2
  while y < n
    // invariant x == mid
    // invariant y >= 0
    // invariant y <= n
  {
    y := y + 1;
  }
}
