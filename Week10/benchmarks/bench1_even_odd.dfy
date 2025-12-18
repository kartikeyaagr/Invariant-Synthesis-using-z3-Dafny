// Benchmark 1: Simple If-Then-Else
// Different updates based on condition
// Requires disjunctive invariant: (i % 2 == 0 && ...) || (i % 2 == 1 && ...)

method EvenOddIncrement(n: int) returns (x: int)
  requires n >= 0
  ensures x >= 0
{
  var i := 0;
  x := 0;
  
  while i < n
    // Expected disjunctive invariant:
    // (i % 2 == 0 ==> x >= i/2) && (i % 2 == 1 ==> x >= (i-1)/2)
    // Or simpler: x >= 0 && i >= 0
  {
    if i % 2 == 0 {
      x := x + 2;
    } else {
      x := x + 1;
    }
    i := i + 1;
  }
}
