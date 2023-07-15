using AliFsmnSharp.Model;
using AliFsmnSharp.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AliFsmnSharp;

public sealed class FsmnVad : IDisposable {
    private readonly InferenceSession session;
    private readonly WavFrontend frontend;
    private readonly int maxEndSil;
    private readonly EncoderConfEntity encoderConfig;
    private readonly E2EVadModel vadScorer;

    public int SampleRate { get; }

    public FsmnVad(string modelFolderPath, int intraOpNumThreads = 4) {
        var options = new SessionOptions();
        options.AppendExecutionProvider_CPU(0);
        options.InterOpNumThreads = intraOpNumThreads;
        session = new InferenceSession(Path.Combine(modelFolderPath, "model.onnx"), options);

        var vadYamlEntity = YamlHelper.ReadYaml<VadYamlEntity>(
            Path.Combine(modelFolderPath, "vad.yaml")) ?? throw new InvalidDataException("Invalid config");
        SampleRate = vadYamlEntity.frontend_conf.fs;
        frontend = new WavFrontend(
            Path.Combine(modelFolderPath, "vad.mvn"), vadYamlEntity.frontend_conf);
        vadScorer = new E2EVadModel(vadYamlEntity.vad_post_conf);
        maxEndSil = vadYamlEntity.vad_post_conf.max_end_silence_time;
        encoderConfig = vadYamlEntity.encoder_conf;
    }

    private IEnumerable<TimeWindow[]> GetSegments(float[] samples) {
        var (feats, featsCount) = frontend.ExtractFeat(new[] { samples });
        var inCaches = PrepareCache();
        var i = 0;
        var cacheTensors = inCaches.Select(cache => new {
                cache, cacheDim = new[] { 1, 128, cache.Length / 128 / 1, 1 }
            })
            .Select(p => new DenseTensor<float>(p.cache, p.cacheDim))
            .Select(cacheTensor => NamedOnnxValue.CreateFromTensor($"in_cache{i++}", cacheTensor))
            .ToList();

        var featsLen = featsCount[0];
        var step = Math.Min(featsCount.Max(), 6000);
        for (var tOffset = 0; tOffset < featsLen; tOffset += Math.Min(step, featsLen - tOffset)) {
            bool isFinal;
            if (tOffset + step >= featsLen - 1) {
                step = featsLen - tOffset;
                isFinal = true;
            } else {
                isFinal = false;
            }

            var inputStep = GetFeatPackage(feats, featsLen, 400, tOffset, step);
            var speechDim = session.InputMetadata["speech"].Dimensions;
            speechDim[1] = inputStep.Length / speechDim[2];

            var container = new List<NamedOnnxValue>();
            var tensor = new DenseTensor<float>(inputStep, speechDim);
            cacheTensors.Add(NamedOnnxValue.CreateFromTensor("speech", tensor));
            container.AddRange(cacheTensors);

            DisposableNamedOnnxValue[] results;
            try {
                results = session.Run(container).ToArray();
            } catch (OnnxRuntimeException) {
                yield break;
            }

            var logits = results[0].AsTensor<float>().ToDenseTensor();
            var scores = DimOneToThree(logits.Buffer.Span, 1, logits.Dimensions[1]);
            var vadOutputEntity = new VadOutputEntity(scores);
            foreach (var result in results[1..]) {
                vadOutputEntity.OutCaches.Add(result.AsEnumerable<float>().ToArray());
            }

            var waveform = GetWaveformPackage(samples, tOffset, step);
            var segments = vadScorer.DefaultCall(scores, waveform, isFinal, maxEndSil);
            if (segments.Length == 0) {
                continue;
            }
            
            yield return segments;
        }
    }
    
    public IEnumerable<TimeWindow> Inference(float[] samples) {
        using var enumerator = GetSegments(samples).GetEnumerator();
        while (enumerator.MoveNext()) {
            foreach (var timeWindow in enumerator.Current) {
                yield return timeWindow;
            }
        }
    }

    public async IAsyncEnumerable<TimeWindow> InferenceAsync(float[] samples) {
        using var enumerator = GetSegments(samples).GetEnumerator();
        while (await Task.Run(enumerator.MoveNext)) {
            foreach (var timeWindow in enumerator.Current) {
                yield return timeWindow;
            }
        }
    }

    private List<float[]> PrepareCache() {
        var inCache = new List<float[]>();

        var fsmnLayers = encoderConfig.fsmn_layers;
        var projDim = encoderConfig.proj_dim;
        var lOrder = encoderConfig.lorder;

        for (var i = 0; i < fsmnLayers; i++) {
            var cache = new float[1 * projDim * (lOrder - 1) * 1];
            inCache.Add(cache);
        }

        return inCache;
    }

    private float[] GetFeatPackage(float[] feats, int x, int y, int tOffset, int step) {
        var start = tOffset * y;
        var end = Math.Min(tOffset + step, x) * y;
        var length = end - start;
        var featsPackage = new float[length];

        Buffer.BlockCopy(feats, start * sizeof(float), featsPackage, 0, length * sizeof(float));

        return featsPackage;
    }

    private static float[] GetWaveformPackage(float[] samplesBatch, int tOffset, int step) {
        var start = tOffset * 160;
        var end = Math.Min(samplesBatch.Length, (tOffset + step - 1) * 160 + 400);

        // Check if end index goes beyond array length
        if (end > samplesBatch.Length) {
            end = samplesBatch.Length;
        }

        var waveformPackage = new float[end - start];
        Array.Copy(samplesBatch, start, waveformPackage, 0, end - start);

        return waveformPackage;
    }


    /// <summary>
    /// 一维数组转3维数组
    /// </summary>
    /// <param name="obj"></param>
    /// <param name="len">一维长</param>
    /// <param name="wid">二维长</param>
    /// <returns></returns>
    private static float[,,] DimOneToThree(Span<float> obj, int len, int wid) {
        if (obj.Length % (len * wid) != 0) {
            throw new Exception("数组长度不符合要求");
        }

        var height = obj.Length / (len * wid);
        var threeDimObj = new float[len, wid, height];

        for (var i = 0; i < obj.Length; i++) {
            threeDimObj[i / (wid * height), i / height % wid, i % height] = obj[i];
        }

        return threeDimObj;
    }

    public void Dispose() {
        session.Dispose();
    }
}