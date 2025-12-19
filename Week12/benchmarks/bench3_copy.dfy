// Benchmark 3: Array Copy
// Copy elements from src to dst
// Invariant: forall k :: 0 <= k < i ==> dst[k] == src[k]

method CopyArray(src: array<int>, dst: array<int>)
  requires src.Length == dst.Length
  modifies dst
  ensures forall k :: 0 <= k < dst.Length ==> dst[k] == src[k]
{
  var i := 0;
  
  while i < src.Length
    invariant 0 <= i <= src.Length
    invariant forall k :: 0 <= k < i ==> dst[k] == src[k]
  {
    dst[i] := src[i];
    i := i + 1;
  }
}
