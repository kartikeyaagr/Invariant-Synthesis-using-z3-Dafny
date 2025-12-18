// Benchmark 5: Quadratic Relationship
// Two variables with y = x² relationship
// Invariant: y == x * x

method QuadraticGrowth(n: int) returns (x: int, y: int)
  requires n >= 0
  ensures x == n
  ensures y == n * n
{
  x := 0;
  y := 0;
  
  while x < n
    // y grows quadratically with x
    // Invariant: y == x * x
    invariant 0 <= x <= n
    invariant y == x * x
  {
    // y = (x+1)² = x² + 2x + 1 = y + 2x + 1
    y := y + 2 * x + 1;
    x := x + 1;
  }
}
