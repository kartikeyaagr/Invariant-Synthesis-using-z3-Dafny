// Benchmark 4: Find Maximum Element
// Find the maximum value in an array
// Invariant: max is greatest among a[0..i]

method FindMax(a: array<int>) returns (max: int)
  requires a.Length > 0
  ensures forall k :: 0 <= k < a.Length ==> a[k] <= max
  ensures exists k :: 0 <= k < a.Length && a[k] == max
{
  var i := 1;
  max := a[0];
  
  while i < a.Length
    invariant 1 <= i <= a.Length
    invariant forall k :: 0 <= k < i ==> a[k] <= max
    invariant exists k :: 0 <= k < i && a[k] == max
  {
    if a[i] > max {
      max := a[i];
    }
    i := i + 1;
  }
}
