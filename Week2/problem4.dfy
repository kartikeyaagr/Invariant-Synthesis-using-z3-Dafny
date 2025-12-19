// Helper function
function Power(base: int, exp: int): int
    requires exp >= 0
{
    if exp == 0 then 1 else base * Power(base, exp - 1)
}

// Lemma to prove Power(base*base, exp) == Power(base, 2*exp).
// The verifier needs this lemma to prove that the loop invariant in FastPower is maintained.
lemma PowerOfSquare(base: int, exp: int)
    requires exp >= 0
    ensures Power(base * base, exp) == Power(base, 2 * exp)
{
    if exp > 0 {
        // Inductive proof for exp > 0
        calc {
            Power(base, 2 * exp);
            base * Power(base, 2 * exp - 1);
            base * (base * Power(base, 2 * exp - 2));
            (base * base) * Power(base, 2 * (exp - 1));
            { PowerOfSquare(base, exp - 1); } // Inductive hypothesis
            (base * base) * Power(base * base, exp - 1);
            Power(base * base, exp);
        }
    }
    // The base case (exp == 0) is proven automatically by Dafny.
}


// Problem 4: Fast Power (Exponentiation by Squaring)
method FastPower(base: int, exp: int) returns (result: int)
    requires exp >= 0
    ensures result == Power(base, exp)
{
    var x := base;
    var n := exp;
    result := 1;
    
    while n > 0
        invariant n >= 0
        invariant result * Power(x, n) == Power(base, exp)
        decreases n
    {
        // We call the lemma to make its 'ensures' clause (the property we just proved)
        // available to the verifier for the rest of this loop iteration.
        PowerOfSquare(x, n / 2);

        if n % 2 == 1 {
            result := result * x;
        }
        x := x * x;
        n := n / 2;
    }
}

