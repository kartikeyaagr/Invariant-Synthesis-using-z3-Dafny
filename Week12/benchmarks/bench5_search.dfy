// Benchmark 5: Linear Search
// Search for a value in an array
// Invariant: value not found in a[0..i]

method LinearSearch(a: array<int>, key: int) returns (found: bool, idx: int)
  ensures found ==> 0 <= idx < a.Length && a[idx] == key
  ensures !found ==> forall k :: 0 <= k < a.Length ==> a[k] != key
{
  var i := 0;
  
  while i < a.Length
    invariant 0 <= i <= a.Length
    invariant forall k :: 0 <= k < i ==> a[k] != key
  {
    if a[i] == key {
      return true, i;
    }
    i := i + 1;
  }
  
  return false, -1;
}
