// Benchmark 6: Flag-Based State
// Behavior changes based on a flag that gets set
// Requires: (flag == false && inv1) || (flag == true && inv2)

method FlagBasedLoop(n: int) returns (x: int, y: int)
  requires n >= 2
  ensures x >= 0 && y >= 0
{
  var i := 0;
  x := 0;
  y := 0;
  var flag := false;
  
  while i < n
    // Before flag is set: x increments
    // After flag is set: y increments
    // Disjunctive: (!flag && x == i) || (flag && y > 0)
  {
    if !flag {
      x := x + 1;
      if x >= n / 2 {
        flag := true;
      }
    } else {
      y := y + 1;
    }
    i := i + 1;
  }
}
