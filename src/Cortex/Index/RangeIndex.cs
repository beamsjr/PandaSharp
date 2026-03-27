namespace Cortex.Index;

/// <summary>
/// Default 0-based integer index (0, 1, 2, ..., N-1).
/// </summary>
public class RangeIndex : IIndex
{
    private readonly int _start;

    public int Length { get; }

    public RangeIndex(int length, int start = 0)
    {
        Length = length;
        _start = start;
    }

    public object GetLabel(int position) => _start + position;

    public IIndex Slice(int offset, int length) => new RangeIndex(length);

    public IIndex TakeRows(ReadOnlySpan<int> indices) => new RangeIndex(indices.Length);

    public IIndex Filter(ReadOnlySpan<bool> mask)
    {
        int count = 0;
        for (int i = 0; i < mask.Length; i++) if (mask[i]) count++;
        return new RangeIndex(count);
    }
}
