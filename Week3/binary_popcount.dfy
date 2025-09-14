// binary_popcount.dfy
// Count number of 1-bits in the binary representation (popcount).

function PopCount(n: nat): nat
  decreases n
{
  if n == 0 then 0 else (n % 2) + PopCount(n / 2)
}

// Small lemma to help Dafny unfold the PopCount definition during proofs
lemma PopCount_unfold(m: nat)
  ensures PopCount(m) == (if m == 0 then 0 else (m % 2) + PopCount(m / 2))
  decreases m
{
  if m == 0 { }
  else { PopCount_unfold(m / 2); }
}

method ComputePopCount(n: nat) returns (c: nat)
  ensures c == PopCount(n)
{
  var m := n;
  c := 0;
  while m > 0
    invariant c >= 0
    invariant c + PopCount(m) == PopCount(n)
    decreases m
  {
    // help Dafny reason about PopCount(m)
    PopCount_unfold(m);
    c := c + (m % 2);
    m := m / 2;
  }
}

method Main() {
  var n := 29; // binary 11101 -> popcount = 4
  var c := ComputePopCount(n);
  print "Popcount(", n, ") = ", c, "\n";
}
