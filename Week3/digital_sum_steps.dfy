module DigitPersistence {
  function DigitSum(n: nat): nat
    decreases n
  {
    if n < 10 then n
    else (n % 10) + DigitSum(n / 10)
  }

  lemma DigitSumLeq(n: nat)
    ensures DigitSum(n) <= n
    decreases n
  {
    if n >= 10 {
      var q := n / 10;
      DigitSumLeq(q);
    }
  }

  // Lemma: if n >= 10, then DigitSum(n) < n
  lemma DigitSumLess(n: nat)
    requires n >= 10
    ensures DigitSum(n) < n
  {
    var q := n / 10;
    DigitSumLeq(q);
    assert DigitSum(q) <= q;
    assert q < 10 * q;
  }

  method Persistence(n: nat) returns (steps: nat)
    ensures (n < 10 ==> steps == 0)
    ensures (n >= 10 ==> steps > 0)
  {
    if n < 10 {
      steps := 0;
    } else {
      var m := n;
      steps := 0;
      while m >= 10
        invariant steps >= 0
        invariant steps == 0 <==> m == n
        decreases m
      {
        DigitSumLess(m);
        m := DigitSum(m);
        steps := steps + 1;
      }
    }
  }
}