// Benchmark 2: Threshold-Based Conditional
// Different behavior before and after reaching a threshold
// Requires: (i < mid && inv1) || (i >= mid && inv2)

method ThresholdLoop(n: int) returns (x: int, y: int)
  requires n >= 0
  ensures x >= 0
  ensures y >= 0
{
  var i := 0;
  x := 0;
  y := 0;
  var mid := n / 2;
  
  while i < n
    // Expected path-sensitive invariant:
    // (i < mid ==> x == i) && (i >= mid ==> y == i - mid)
    // Simpler: x >= 0 && y >= 0 && i >= 0
  {
    if i < mid {
      x := x + 1;
    } else {
      y := y + 1;
    }
    i := i + 1;
  }
}
