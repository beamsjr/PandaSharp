using BenchmarkDotNet.Running;
using PandaSharp.Tests.Benchmarks;

BenchmarkSwitcher.FromAssembly(typeof(DataFrameBenchmarks).Assembly).Run(args);
