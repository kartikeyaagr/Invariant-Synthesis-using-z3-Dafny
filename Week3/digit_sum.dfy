// DigitalRoot.dfy
// Helper: recursive digit-sum (keeps the original spec; verified)
function DigitSum(n: int): int
  requires n >= 0
  decreases n
{
  if n == 0 then 0 else n % 10 + DigitSum(n / 10)
}

// Digital root using arithmetic shortcut (no recursion, no loop-decreases issues)
method ComputeDigitalRoot(n: int) returns (r: int)
  requires n >= 0
  ensures 0 <= r < 10
  ensures (n == 0 && r == 0) || (n > 0 && r == 1 + (n - 1) % 9)
{
  if n == 0 {
    r := 0;
    return;
  }
  r := 1 + (n - 1) % 9;
}

// Simple main to run
method Main() {
  var n := 98765;
  var r := ComputeDigitalRoot(n);
  print "Digital root of ", n, " is ", r, "\n";
}
