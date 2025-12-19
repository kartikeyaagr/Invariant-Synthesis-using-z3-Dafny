// Problem 5: Reversing a Number
// This version is updated to be fully verifiable by Dafny.

// Helper function that uses an accumulator, mirroring the loop's logic.
// This is the key change to make the proof straightforward.
function ReverseAcc(n: int, acc: int): int
    requires n >= 0
{
    if n == 0 then acc
    else ReverseAcc(n / 10, acc * 10 + (n % 10))
}

// The main specification function now uses the accumulator-style helper.
function ReverseDigits(n: int): int
    requires n >= 0
{
    ReverseAcc(n, 0)
}

method ReverseNumber(n: int) returns (rev: int)
    requires n >= 0
    ensures rev == ReverseDigits(n)
{
    rev := 0;
    var num := n;
    
    while num > 0
        invariant num >= 0
        // The new invariant directly states that the final result is equal
        // to the result of running the algorithm on the remaining 'num' and 'rev'.
        // Since ReverseAcc is defined identically to the loop's operation,
        // this invariant is easily proven.
        invariant ReverseDigits(n) == ReverseAcc(num, rev)
        decreases num
    {
        var digit := num % 10;
        rev := rev * 10 + digit;
        num := num / 10;
    }
}

// The Power and NumDigits helper functions are no longer needed for verification
// because the new ReverseDigits function doesn't depend on them.

