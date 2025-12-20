// Benchmark 2: Conditional Update Pattern
// Requires disjunction: (x % 2 == 0 && y >= x/2) || (x % 2 == 1 && y >= (x-1)/2)
// Simplified for linear synthesis: bounds on x and y

method ConditionalUpdate(n: int) returns (x: int, y: int)
  requires n >= 0
  ensures x == n
{
  x := 0;
  y := 0;
  
  while x < n
      invariant -n + x <= 0
    // Invariants to be synthesized:
    // invariant x >= 0
    // invariant y >= 0  
    // invariant x <= n
  {
    if x % 2 == 0 {
      y := y + 1;
    }
    x := x + 1;
  }
}
