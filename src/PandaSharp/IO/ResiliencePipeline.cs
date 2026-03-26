using System.Net;

namespace PandaSharp.IO;

/// <summary>
/// Configuration options for cloud storage resilience (retry + circuit breaker).
/// </summary>
public class CloudStorageOptions
{
    public int MaxRetries { get; set; } = 3;
    public TimeSpan InitialDelay { get; set; } = TimeSpan.FromMilliseconds(200);
    public double BackoffMultiplier { get; set; } = 2.0;
    public double JitterFactor { get; set; } = 0.1;
    public int CircuitBreakerThreshold { get; set; } = 5;
    public TimeSpan CircuitBreakerDuration { get; set; } = TimeSpan.FromSeconds(30);
}

/// <summary>
/// Lightweight resilience wrapper providing exponential backoff retry and circuit breaker
/// for cloud storage operations. No external dependencies (no Polly).
/// Thread-safe via Interlocked operations.
/// </summary>
public class ResiliencePipeline
{
    private readonly CloudStorageOptions _options;

    // Circuit breaker state: 0 = Closed, 1 = Open, 2 = HalfOpen
    private const int StateClosed = 0;
    private const int StateOpen = 1;
    private const int StateHalfOpen = 2;

    private int _state = StateClosed;
    private int _consecutiveFailures;
    private long _openedAtTicks;

    public ResiliencePipeline(CloudStorageOptions? options = null)
    {
        _options = options ?? new CloudStorageOptions();
    }

    /// <summary>
    /// True when the circuit breaker is in the Open state (rejecting calls).
    /// </summary>
    public bool IsCircuitOpen => Volatile.Read(ref _state) == StateOpen
        && !HasDurationElapsed();

    /// <summary>
    /// Reset the circuit breaker to closed state.
    /// </summary>
    public void Reset()
    {
        Interlocked.Exchange(ref _state, StateClosed);
        Interlocked.Exchange(ref _consecutiveFailures, 0);
    }

    /// <summary>
    /// Execute an async function with retry and circuit breaker protection.
    /// </summary>
    public async Task<T> ExecuteAsync<T>(Func<CancellationToken, Task<T>> action, CancellationToken ct = default)
    {
        EnsureCircuitAllows();

        int attempt = 0;
        while (true)
        {
            ct.ThrowIfCancellationRequested();
            try
            {
                var result = await action(ct).ConfigureAwait(false);
                OnSuccess();
                return result;
            }
            catch (Exception ex) when (IsTransient(ex) && attempt < _options.MaxRetries)
            {
                attempt++;
                var delay = ComputeDelay(attempt);
                await Task.Delay(delay, ct).ConfigureAwait(false);

                // Re-check circuit after delay — another thread may have opened it
                EnsureCircuitAllows();
            }
            catch (OperationCanceledException)
            {
                // Cancellation is not a fault — do not count toward circuit breaker
                throw;
            }
            catch
            {
                OnFailure();
                throw;
            }
        }
    }

    /// <summary>
    /// Execute an async action with retry and circuit breaker protection.
    /// </summary>
    public async Task ExecuteAsync(Func<CancellationToken, Task> action, CancellationToken ct = default)
    {
        await ExecuteAsync<object?>(async token =>
        {
            await action(token).ConfigureAwait(false);
            return null;
        }, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Compute delay for a given attempt using exponential backoff with jitter.
    /// delay = initialDelay * (backoffMultiplier ^ attempt) * (1 + random * jitterFactor)
    /// </summary>
    internal TimeSpan ComputeDelay(int attempt)
    {
        double baseDelay = _options.InitialDelay.TotalMilliseconds
            * Math.Pow(_options.BackoffMultiplier, attempt);
        double jitter = 1.0 + Random.Shared.NextDouble() * _options.JitterFactor;
        return TimeSpan.FromMilliseconds(baseDelay * jitter);
    }

    private static bool IsTransient(Exception ex)
    {
        return ex is IOException
            || ex is HttpRequestException
            || (ex is TaskCanceledException tce && tce.InnerException is TimeoutException);
    }

    private bool HasDurationElapsed()
    {
        long openedAt = Interlocked.Read(ref _openedAtTicks);
        return (DateTime.UtcNow.Ticks - openedAt) >= _options.CircuitBreakerDuration.Ticks;
    }

    private void EnsureCircuitAllows()
    {
        int currentState = Volatile.Read(ref _state);

        if (currentState == StateOpen)
        {
            if (HasDurationElapsed())
            {
                // Transition to half-open: allow exactly one probe request
                Interlocked.CompareExchange(ref _state, StateHalfOpen, StateOpen);
            }
            else
            {
                throw new InvalidOperationException("Circuit breaker is open. Requests are not allowed.");
            }
        }
        // HalfOpen and Closed both allow the request through
    }

    private void OnSuccess()
    {
        Interlocked.Exchange(ref _consecutiveFailures, 0);
        Interlocked.Exchange(ref _state, StateClosed);
    }

    private void OnFailure()
    {
        int failures = Interlocked.Increment(ref _consecutiveFailures);
        if (failures >= _options.CircuitBreakerThreshold)
        {
            Interlocked.Exchange(ref _openedAtTicks, DateTime.UtcNow.Ticks);
            Interlocked.Exchange(ref _state, StateOpen);
        }
    }
}
