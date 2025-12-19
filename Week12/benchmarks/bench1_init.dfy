// Benchmark 1: Array Initialization
// Initialize all elements to zero
// Invariant: forall k :: 0 <= k < i ==> a[k] == 0

method InitArray(a: array<int>)
  modifies a
  ensures forall k :: 0 <= k < a.Length ==> a[k] == 0
{
  var i := 0;
  
  while i < a.Length
    invariant 0 <= i <= a.Length
    invariant forall k :: 0 <= k < i ==> a[k] == 0
  {
    a[i] := 0;
    i := i + 1;
  }
}
