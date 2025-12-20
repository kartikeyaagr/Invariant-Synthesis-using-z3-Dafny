// Benchmark 1: Two-variable linear update
// Requires conjunction: (x >= 0) && (y >= 0) && (y == 2*x)

method TwoVarLinear(n: int) returns (x: int, y: int)
  requires n >= 0
  ensures x == n
  ensures y == 2 * n
{
  x := 0;
  y := 0;
  
  while x < n
      invariant 2 * x - y == 0
      invariant x - n <= 0
    // Invariants to be synthesized:
    // invariant x >= 0
    // invariant y >= 0
    // invariant y == 2 * x
  {
    x := x + 1;
    y := y + 2;
  }
}
