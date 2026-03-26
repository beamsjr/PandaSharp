using FluentAssertions;
using PandaSharp.ML.Torch;
using Xunit;

namespace PandaSharp.ML.Torch.Tests;

public class TorchDeviceTests
{
    [Fact]
    public void DeviceInfo_ReturnsNonEmptyString()
    {
        var info = TorchDevice.DeviceInfo();

        info.Should().NotBeNullOrEmpty();
        info.Should().Contain("Best:");
    }

    [Fact]
    public void Best_ReturnsADevice()
    {
        var device = TorchDevice.Best;

        device.Should().NotBeNull();
        // On CI / CPU-only systems, this should be CPU
        device.type.Should().BeOneOf(TorchSharp.DeviceType.CPU, TorchSharp.DeviceType.CUDA, TorchSharp.DeviceType.MPS);
    }

    [Fact]
    public void Resolve_Auto_ReturnsBest()
    {
        var resolved = TorchDevice.Resolve("auto");

        resolved.Should().Be(TorchDevice.Best);
    }

    [Fact]
    public void Resolve_Cpu_ReturnsCpu()
    {
        var resolved = TorchDevice.Resolve("cpu");

        resolved.type.Should().Be(TorchSharp.DeviceType.CPU);
    }
}
