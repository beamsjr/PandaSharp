using BenchmarkDotNet.Running;
using Cortex.Tests.Benchmarks;

BenchmarkSwitcher.FromAssembly(typeof(DataFrameBenchmarks).Assembly).Run(args);
