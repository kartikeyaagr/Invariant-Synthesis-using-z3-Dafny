// Benchmark 6: Array Reverse
// Reverse an array in place
// Invariants track swapped elements

method Reverse(a: array<int>)
  modifies a
  ensures forall k :: 0 <= k < a.Length ==> a[k] == old(a[a.Length - 1 - k])
{
  var lo := 0;
  var hi := a.Length - 1;
  
  while lo < hi
    invariant 0 <= lo <= hi + 1 <= a.Length
    invariant forall k :: 0 <= k < lo ==> a[k] == old(a[a.Length - 1 - k])
    invariant forall k :: hi < k < a.Length ==> a[k] == old(a[a.Length - 1 - k])
    invariant forall k :: lo <= k <= hi ==> a[k] == old(a[k])
  {
    var tmp := a[lo];
    a[lo] := a[hi];
    a[hi] := tmp;
    lo := lo + 1;
    hi := hi - 1;
  }
}
