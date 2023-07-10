using NAudio.Wave;

namespace AliFsmnSharp.Tests;

public class ParaformerTests {
    [Test]
    public void Test()
    {
        const string audioFilePath = @"G:\Source\CSharp\EasyPathology\EasyPathology.Inference\Resources\Paraformer-large\asr_example.wav"; // 替换为您的音频文件路径

        var paraformer = new Paraformer(@"G:\Source\CSharp\EasyPathology\EasyPathology.Inference\Resources\Paraformer-large");
        
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
}