namespace Cortex.Index;

public interface IIndex
{
    int Length { get; }
    object? GetLabel(int position);
    IIndex Slice(int offset, int length);
    IIndex TakeRows(ReadOnlySpan<int> indices);
    IIndex Filter(ReadOnlySpan<bool> mask);
}
