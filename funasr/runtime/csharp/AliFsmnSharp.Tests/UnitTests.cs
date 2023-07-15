using System.Diagnostics;
using NAudio.Wave;

namespace AliFsmnSharp.Tests;

public class UnitTests {
    [Test]
    public void TestParaformer() {
        const string audioFilePath =
            @"G:\Source\CSharp\EasyPathology\EasyPathology.Inference\Resources\Paraformer-large\asr_example.wav"; // 替换为您的音频文件路径

        var paraformer =
            new Paraformer(@"G:\Source\CSharp\EasyPathology\EasyPathology.Inference\Resources\Paraformer-large");

        using var reader = new AudioFileReader(audioFilePath);
        Assert.Multiple(() => {
            Assert.That(reader.WaveFormat.Channels, Is.EqualTo(1));
            Assert.That(reader.WaveFormat.SampleRate, Is.EqualTo(paraformer.SampleRate));
            Assert.That(reader.WaveFormat.BitsPerSample, Is.EqualTo(32));
        });

        var bytesNeeded = (int)reader.Length;
        var buffer = new byte[bytesNeeded];
        var readCount = reader.Read(buffer, 0, bytesNeeded);
        if (readCount != bytesNeeded) {
            throw new Exception("Could not read entire wave file");
        }

        // 将字节数据转换成浮点数数组
        var floatArray = new float[bytesNeeded / 4];
        Buffer.BlockCopy(buffer, 0, floatArray, 0, bytesNeeded);

        var result = paraformer.Inference(new[] { floatArray }).First();
        Console.WriteLine(result);
        Assert.That(result, Is.EqualTo("欢迎大家来体验达摩院推出的语音识别模型"));
    }

    [Test]
    public void TestFsmnVad() {
        var sw = Stopwatch.StartNew();
        const string audioFilePath =
            @"G:\Source\CSharp\EasyPathology\EasyPathology.Inference\Resources\Vad-16k\肾水样变性.wav";

        var vad = new FsmnVad(
            @"G:\Source\CSharp\EasyPathology\EasyPathology.Inference\Resources\Vad-16k",
            4);
        using var reader = new AudioFileReader(audioFilePath);
        Assert.Multiple(() => {
            Assert.That(reader.WaveFormat.Channels, Is.EqualTo(1));
            Assert.That(reader.WaveFormat.SampleRate, Is.EqualTo(vad.SampleRate));
            Assert.That(reader.WaveFormat.BitsPerSample, Is.EqualTo(32));
        });

        var bytesNeeded = (int)reader.Length;
        var buffer = new byte[bytesNeeded];
        var readCount = reader.Read(buffer, 0, bytesNeeded);
        if (readCount != bytesNeeded) {
            throw new Exception("Could not read entire wave file");
        }

        // 将字节数据转换成浮点数数组
        var floatArray = new float[bytesNeeded / 4];
        Buffer.BlockCopy(buffer, 0, floatArray, 0, bytesNeeded);
        Console.WriteLine($"Read wav costs {sw.ElapsedMilliseconds}ms");
        sw.Restart();
        
        var result = vad.Inference(floatArray).ToArray();
        Console.WriteLine($"Vad costs {sw.ElapsedMilliseconds}ms");
        sw.Restart();
        
        var sampleRate = reader.WaveFormat.SampleRate;
        var channels = reader.WaveFormat.Channels;
        var segments = new float[result.Length][];
        for (var i = 0; i < result.Length; i++) {
            var window = result[i];

            // 计算开始和结束的样本索引
            var startSampleIndex = window.BeginTime.TotalSeconds * sampleRate;
            var endSampleIndex = window.EndTime.TotalSeconds * sampleRate;

            // 计算开始和结束的浮点数索引
            var startFloatIndex = (int)(startSampleIndex * channels);
            var endFloatIndex = (int)(endSampleIndex * channels);

            // 切割对应的浮点数数组
            var length = endFloatIndex - startFloatIndex;
            var segment = new float[length];
            Array.Copy(floatArray, startFloatIndex, segment, 0, length);
            segments[i] = segment;
        }
        Console.WriteLine($"Split wav costs {sw.ElapsedMilliseconds}ms");
        sw.Restart();
        
        var paraformer =
            new Paraformer(
                @"G:\Source\CSharp\EasyPathology\EasyPathology.Inference\Resources\Paraformer-large",
                batchSize: 1,
                intraOpNumThreads: 4);
        var results = paraformer.Inference(segments).ToArray();
        Console.WriteLine($"Inference costs {sw.ElapsedMilliseconds}ms\n");
        Console.WriteLine(string.Join("\n", results));
    }
}