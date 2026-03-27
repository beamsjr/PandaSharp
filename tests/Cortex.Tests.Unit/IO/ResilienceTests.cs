using System.Diagnostics;
using FluentAssertions;
using Cortex.IO;
using Xunit;

namespace Cortex.Tests.Unit.IO;

public class ResilienceTests
{
    [Fact]
    public async Task Retry_SucceedsOnThirdAttempt()
    {
        var options = new CloudStorageOptions
        {
            MaxRetries = 3,
            InitialDelay = TimeSpan.FromMilliseconds(10),
            BackoffMultiplier = 2.0,
            JitterFactor = 0.0 // No jitter for deterministic test
        };
        var pipeline = new ResiliencePipeline(options);
        int attempts = 0;

        var result = await pipeline.ExecuteAsync<int>(async _ =>
        {
            attempts++;
            if (attempts < 3)
                throw new IOException("transient failure");
            return 42;
        });

        result.Should().Be(42);
        attempts.Should().Be(3);
    }

    [Fact]
    public async Task Retry_DelayPattern_IsExponential()
    {
        var options = new CloudStorageOptions
        {
            MaxRetries = 3,
            InitialDelay = TimeSpan.FromMilliseconds(50),
            BackoffMultiplier = 2.0,
            JitterFactor = 0.0
        };
        var pipeline = new ResiliencePipeline(options);
        var timestamps = new List<long>();

        var sw = Stopwatch.StartNew();
        int attempts = 0;
        await pipeline.ExecuteAsync<int>(async _ =>
        {
            timestamps.Add(sw.ElapsedMilliseconds);
            attempts++;
            if (attempts < 4)
                throw new IOException("transient");
            return 1;
        });

        // Delay between attempt 1->2 should be ~100ms (50*2^1), 2->3 ~200ms (50*2^2)
        // Allow generous tolerance for CI
        var delay1 = timestamps[1] - timestamps[0];
        var delay2 = timestamps[2] - timestamps[1];
        delay1.Should().BeGreaterThanOrEqualTo(50); // at least 50ms (base * 2^1 = 100, allow variance)
        delay2.Should().BeGreaterThan(delay1 - 20); // second delay should be roughly >= first
    }

    [Fact]
    public async Task CircuitBreaker_OpensAfterThresholdFailures()
    {
        var options = new CloudStorageOptions
        {
            MaxRetries = 0, // No retries, so each call fails immediately
            CircuitBreakerThreshold = 3,
            CircuitBreakerDuration = TimeSpan.FromSeconds(30)
        };
        var pipeline = new ResiliencePipeline(options);

        // Cause N failures to trip the circuit
        for (int i = 0; i < 3; i++)
        {
            await Assert.ThrowsAsync<IOException>(() =>
                pipeline.ExecuteAsync<int>(async _ => throw new IOException("fail")));
        }

        pipeline.IsCircuitOpen.Should().BeTrue();

        // Next call should be rejected by circuit breaker
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            pipeline.ExecuteAsync<int>(async _ => 1));
        ex.Message.Should().Contain("Circuit breaker is open");
    }

    [Fact]
    public async Task CircuitBreaker_RecoversThroughHalfOpen()
    {
        var options = new CloudStorageOptions
        {
            MaxRetries = 0,
            CircuitBreakerThreshold = 2,
            CircuitBreakerDuration = TimeSpan.FromMilliseconds(100) // Very short for testing
        };
        var pipeline = new ResiliencePipeline(options);

        // Trip the circuit
        for (int i = 0; i < 2; i++)
        {
            await Assert.ThrowsAsync<IOException>(() =>
                pipeline.ExecuteAsync<int>(async _ => throw new IOException("fail")));
        }

        pipeline.IsCircuitOpen.Should().BeTrue();

        // Wait for circuit breaker duration to elapse
        await Task.Delay(150);

        // Circuit should now be half-open — allow one request through
        var result = await pipeline.ExecuteAsync<int>(async _ => 99);
        result.Should().Be(99);
        pipeline.IsCircuitOpen.Should().BeFalse();
    }

    [Fact]
    public void Jitter_ProducesDifferentDelays()
    {
        var options = new CloudStorageOptions
        {
            InitialDelay = TimeSpan.FromMilliseconds(100),
            BackoffMultiplier = 2.0,
            JitterFactor = 0.5 // Large jitter to make randomness visible
        };
        var pipeline = new ResiliencePipeline(options);

        var delays = Enumerable.Range(0, 20)
            .Select(_ => pipeline.ComputeDelay(1).TotalMilliseconds)
            .ToList();

        // With jitter, not all delays should be exactly the same
        delays.Distinct().Count().Should().BeGreaterThan(1,
            "jitter should produce varying delay values");

        // Base delay for attempt 1 = 100 * 2^1 = 200, with jitter factor 0.5: [200, 300]
        foreach (var d in delays)
        {
            d.Should().BeGreaterThanOrEqualTo(200);
            d.Should().BeLessThanOrEqualTo(300.1); // small tolerance
        }
    }

    [Fact]
    public async Task NonTransientException_DoesNotRetry()
    {
        var options = new CloudStorageOptions { MaxRetries = 3 };
        var pipeline = new ResiliencePipeline(options);
        int attempts = 0;

        await Assert.ThrowsAsync<ArgumentException>(() =>
            pipeline.ExecuteAsync<int>(async _ =>
            {
                attempts++;
                throw new ArgumentException("non-transient");
            }));

        attempts.Should().Be(1, "non-transient exceptions should not trigger retries");
    }

    [Fact]
    public async Task TransientExceptions_AllRetried()
    {
        var options = new CloudStorageOptions
        {
            MaxRetries = 1,
            InitialDelay = TimeSpan.FromMilliseconds(1),
            JitterFactor = 0.0
        };

        // IOException
        var pipeline1 = new ResiliencePipeline(options);
        int a1 = 0;
        await pipeline1.ExecuteAsync<int>(async _ =>
        {
            if (++a1 == 1) throw new IOException("io");
            return 1;
        });
        a1.Should().Be(2);

        // HttpRequestException
        var pipeline2 = new ResiliencePipeline(options);
        int a2 = 0;
        await pipeline2.ExecuteAsync<int>(async _ =>
        {
            if (++a2 == 1) throw new HttpRequestException("http");
            return 1;
        });
        a2.Should().Be(2);

        // TaskCanceledException with inner TimeoutException
        var pipeline3 = new ResiliencePipeline(options);
        int a3 = 0;
        await pipeline3.ExecuteAsync<int>(async _ =>
        {
            if (++a3 == 1) throw new TaskCanceledException("timeout", new TimeoutException());
            return 1;
        });
        a3.Should().Be(2);
    }

    [Fact]
    public async Task TaskCanceledException_WithoutTimeoutInner_DoesNotRetry()
    {
        var options = new CloudStorageOptions
        {
            MaxRetries = 3,
            InitialDelay = TimeSpan.FromMilliseconds(1)
        };
        var pipeline = new ResiliencePipeline(options);
        int attempts = 0;

        await Assert.ThrowsAsync<TaskCanceledException>(() =>
            pipeline.ExecuteAsync<int>(async _ =>
            {
                attempts++;
                throw new TaskCanceledException("cancelled without timeout");
            }));

        attempts.Should().Be(1);
    }

    [Fact]
    public async Task Reset_ClosesCircuit()
    {
        var options = new CloudStorageOptions
        {
            MaxRetries = 0,
            CircuitBreakerThreshold = 2,
            CircuitBreakerDuration = TimeSpan.FromMinutes(10)
        };
        var pipeline = new ResiliencePipeline(options);

        for (int i = 0; i < 2; i++)
        {
            await Assert.ThrowsAsync<IOException>(() =>
                pipeline.ExecuteAsync<int>(async _ => throw new IOException("fail")));
        }

        pipeline.IsCircuitOpen.Should().BeTrue();
        pipeline.Reset();
        pipeline.IsCircuitOpen.Should().BeFalse();

        // Should now accept requests
        var result = await pipeline.ExecuteAsync<int>(async _ => 7);
        result.Should().Be(7);
    }

    [Fact]
    public async Task ExecuteAsync_VoidOverload_Works()
    {
        var options = new CloudStorageOptions
        {
            MaxRetries = 1,
            InitialDelay = TimeSpan.FromMilliseconds(1),
            JitterFactor = 0.0
        };
        var pipeline = new ResiliencePipeline(options);
        int calls = 0;

        await pipeline.ExecuteAsync(async _ =>
        {
            if (++calls == 1) throw new IOException("fail");
        });

        calls.Should().Be(2);
    }
}
