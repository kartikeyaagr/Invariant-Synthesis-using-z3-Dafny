// Benchmark 8: Check All Positive
// Check if all array elements are positive
// Invariant: all elements in a[0..i] are positive

method AllPositive(a: array<int>) returns (result: bool)
  ensures result <==> forall k :: 0 <= k < a.Length ==> a[k] > 0
{
  var i := 0;
  
  while i < a.Length
    invariant 0 <= i <= a.Length
    invariant forall k :: 0 <= k < i ==> a[k] > 0
  {
    if a[i] <= 0 {
      return false;
    }
    i := i + 1;
  }
  
  return true;
}
