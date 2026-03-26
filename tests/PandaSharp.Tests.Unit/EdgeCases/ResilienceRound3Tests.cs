using FluentAssertions;
using PandaSharp.IO;
using Xunit;

namespace PandaSharp.Tests.Unit.EdgeCases;

public class ResilienceRound3Tests
{
    // ═══ Bug: Already-cancelled CancellationToken counts toward circuit breaker failures ═══

    [Fact]
    public async Task AlreadyCancelledToken_DoesNotCountAsCircuitBreakerFailure()
    {
        // When ExecuteAsync is called with an already-cancelled CancellationToken,
        // the OperationCanceledException falls through to the catch-all handler
        // which calls OnFailure(), incrementing the circuit breaker failure counter.
        // Cancellation is not a transient fault and should not trip the circuit breaker.
        var options = new CloudStorageOptions
        {
            MaxRetries = 0,
            CircuitBreakerThreshold = 2,
            CircuitBreakerDuration = TimeSpan.FromMinutes(10)
        };
        var pipeline = new ResiliencePipeline(options);
        var cts = new CancellationTokenSource();
        cts.Cancel(); // already cancelled

        // Call twice with cancelled token
        for (int i = 0; i < 3; i++)
        {
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
                pipeline.ExecuteAsync<int>(async _ => 42, cts.Token));
        }

        // The circuit should NOT be open — cancellation is not a failure
        pipeline.IsCircuitOpen.Should().BeFalse(
            "cancellation should not count toward circuit breaker failures");

        // A valid call should still succeed
        var result = await pipeline.ExecuteAsync<int>(async _ => 99);
        result.Should().Be(99);
    }

    [Fact]
    public async Task CancellationDuringRetryDelay_DoesNotCountAsCircuitBreakerFailure()
    {
        // When cancellation happens during the retry delay (Task.Delay throws),
        // this should not count as a circuit breaker failure either.
        var options = new CloudStorageOptions
        {
            MaxRetries = 3,
            InitialDelay = TimeSpan.FromSeconds(10), // long delay so cancellation hits during delay
            CircuitBreakerThreshold = 2,
            CircuitBreakerDuration = TimeSpan.FromMinutes(10),
            JitterFactor = 0.0
        };
        var pipeline = new ResiliencePipeline(options);
        var cts = new CancellationTokenSource();

        int attempts = 0;
        var task = pipeline.ExecuteAsync<int>(async ct =>
        {
            attempts++;
            if (attempts == 1)
            {
                // Cancel while waiting for retry delay
                _ = Task.Delay(10).ContinueWith(_ => cts.Cancel());
                throw new IOException("transient");
            }
            return 42;
        }, cts.Token);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => task);

        // Circuit should not be open
        pipeline.IsCircuitOpen.Should().BeFalse(
            "cancellation during retry delay should not trip circuit breaker");
    }

    // ═══ Bug: OperationCanceledException from action counts toward circuit breaker ═══

    [Fact]
    public async Task OperationCanceledException_FromAction_DoesNotTripCircuitBreaker()
    {
        // When the action itself throws OperationCanceledException (e.g., HttpClient
        // cancelled), it falls into the catch-all handler which calls OnFailure().
        // This incorrectly increments the circuit breaker failure counter.
        // Cancellation is a deliberate operation, not a fault.
        var options = new CloudStorageOptions
        {
            MaxRetries = 0,
            CircuitBreakerThreshold = 2,
            CircuitBreakerDuration = TimeSpan.FromMinutes(10)
        };
        var pipeline = new ResiliencePipeline(options);

        // Simulate 3 cancelled operations
        for (int i = 0; i < 3; i++)
        {
            await Assert.ThrowsAsync<OperationCanceledException>(() =>
                pipeline.ExecuteAsync<int>(async _ =>
                    throw new OperationCanceledException("cancelled by user")));
        }

        // Circuit should NOT be open — these were cancellations, not failures
        pipeline.IsCircuitOpen.Should().BeFalse(
            "OperationCanceledException should not count toward circuit breaker failures");
    }
}
