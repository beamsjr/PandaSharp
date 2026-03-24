namespace PandaSharp.Column;

/// <summary>
/// Boolean mask operations for combining filter conditions.
/// Usage: var mask = col1.Gt(30).And(col2.Lt(100));
/// </summary>
public static class MaskExtensions
{
    public static bool[] And(this bool[] left, bool[] right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Mask lengths must match.");
        var result = new bool[left.Length];
        for (int i = 0; i < left.Length; i++)
            result[i] = left[i] && right[i];
        return result;
    }

    public static bool[] Or(this bool[] left, bool[] right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Mask lengths must match.");
        var result = new bool[left.Length];
        for (int i = 0; i < left.Length; i++)
            result[i] = left[i] || right[i];
        return result;
    }

    public static bool[] Not(this bool[] mask)
    {
        var result = new bool[mask.Length];
        for (int i = 0; i < mask.Length; i++)
            result[i] = !mask[i];
        return result;
    }

    public static bool[] Xor(this bool[] left, bool[] right)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Mask lengths must match.");
        var result = new bool[left.Length];
        for (int i = 0; i < left.Length; i++)
            result[i] = left[i] ^ right[i];
        return result;
    }

    /// <summary>
    /// Count the number of true values in the mask.
    /// </summary>
    public static int CountTrue(this bool[] mask)
    {
        int count = 0;
        for (int i = 0; i < mask.Length; i++)
            if (mask[i]) count++;
        return count;
    }

    /// <summary>
    /// Returns true if any value in the mask is true.
    /// </summary>
    public static bool Any(this bool[] mask)
    {
        for (int i = 0; i < mask.Length; i++)
            if (mask[i]) return true;
        return false;
    }

    /// <summary>
    /// Returns true if all values in the mask are true.
    /// </summary>
    public static bool All(this bool[] mask)
    {
        for (int i = 0; i < mask.Length; i++)
            if (!mask[i]) return false;
        return true;
    }
}
