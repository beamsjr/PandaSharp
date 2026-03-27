namespace Cortex.Storage;

public enum MutationMode
{
    /// <summary>
    /// Default: all operations return new DataFrames. Original is never modified.
    /// </summary>
    Immutable,

    /// <summary>
    /// Buffers are reference-counted. Mutation copies only when shared.
    /// </summary>
    CopyOnWrite
}
