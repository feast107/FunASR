using AliFsmnSharp.Model;
using AliFsmnSharp.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AliFsmnSharp;

/// <summary>
/// Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
/// </summary>
/// <remarks>
/// 输入的音频需要是<see cref="SampleRate"/>采样率的单声道32位浮点型PCM音频
/// </remarks>
public sealed class Paraformer : IDisposable {
    private readonly AsrYamlEntity config;
    private readonly WavFrontend frontend;
    private readonly TokenIdConverter converter;
    private readonly InferenceSession session;
    private readonly int batchSize;

    public int SampleRate => config.frontend_conf.fs;

    public Paraformer(
        string modelFolderPath,
        bool isQuantized = true,
        int batchSize = 1,
        int intraOpNumThreads = 4) {
        config = YamlHelper.ReadYaml<AsrYamlEntity>(Path.Combine(modelFolderPath, "config.yaml")) ??
                 throw new InvalidDataException("Invalid config");
        converter = new TokenIdConverter(config.token_list);
        frontend = new WavFrontend(Path.Combine(modelFolderPath, "am.mvn"), config.frontend_conf);
        session = new InferenceSession(
            Path.Combine(modelFolderPath, isQuantized ? "model_quant.onnx" : "model.onnx"),
            new SessionOptions {
                IntraOpNumThreads = intraOpNumThreads
            });

        this.batchSize = batchSize;
    }

    /// <summary>
    /// 推理，生成字幕
    /// </summary>
    /// <param name="samplesList"></param>
    public IEnumerable<string> Inference(IEnumerable<float[]> samplesList) {
        foreach (var samplesBatch in samplesList.Chunk(batchSize)) {
            var (feats, featsCount) = frontend.ExtractFeat(samplesBatch);

            var speechDim = session.InputMetadata["speech"].Dimensions;
            speechDim[0] = samplesBatch.Length;
            speechDim[1] = feats.Length / samplesBatch.Length / speechDim[2];

            var speechLengthsDim = session.InputMetadata["speech_lengths"].Dimensions;
            speechLengthsDim[0] = featsCount.Length;

            var outputs = session.Run(new[] {
                NamedOnnxValue.CreateFromTensor("speech",
                    new DenseTensor<float>(feats, speechDim)),
                NamedOnnxValue.CreateFromTensor("speech_lengths",
                    new DenseTensor<int>(featsCount, speechLengthsDim)),
            }).ToList();

            var amScores = outputs[0].AsTensor<float>().ToDenseTensor();
            var predicts = Decode(
                amScores.Buffer, amScores.Dimensions,
                outputs[1].AsTensor<int>().ToArray());

            foreach (var predict in predicts) {
                yield return string.Join(null, predict);
            }
        }
    }

    private List<string[]> Decode(
        Memory<float> amScores, 
        ReadOnlySpan<int> amScoreDimensions,
        int[] validTokenLengths) {
        var results = new List<string[]>();

        // amScores: [batchSize, x, y]
        for (var i = 0; i < amScoreDimensions[0]; i++) {
            var sliceLength = amScoreDimensions[1] * amScoreDimensions[2];
            results.Add(DecodeOne(
                amScores.Slice(i * sliceLength, sliceLength),
                amScoreDimensions[1..],
                validTokenLengths[i]));
        }

        return results;
    }

    private string[] DecodeOne(Memory<float> amScore, ReadOnlySpan<int> amScoreDimensions, int validTokenNum) {
        var ySeq = GetMaxIndexes(amScore, amScoreDimensions);

        // pad with mask tokens to ensure compatibility with sos/eos tokens
        // asr_model.sos:1  asr_model.eos:2
        var ySeqPadded = new int[ySeq.Length + 2];
        ySeqPadded[0] = 1;
        Array.Copy(ySeq, 0, ySeqPadded, 1, ySeq.Length);
        ySeqPadded[^1] = 2;

        // remove sos/eos and get results
        var tokenInt = new int[ySeqPadded.Length - 2];
        Array.Copy(ySeqPadded, 1, tokenInt, 0, tokenInt.Length);

        // remove blank symbol id, which is assumed to be 0
        tokenInt = tokenInt.Where(x => x != 0 && x != 2).ToArray();

        // Change integer-ids to tokens
        var tokens = converter.Ids2Tokens(tokenInt); // You'll need to implement this
        tokens = tokens[..(validTokenNum - config.model_conf.predictor_bias)];

        return tokens;
    }

    private static int[] GetMaxIndexes(Memory<float> arrays, ReadOnlySpan<int> dimensions) {
        var x = dimensions[0];
        var y = dimensions[1];

        var result = new int[x];
        for (var i = 0; i < x; i++) {
            var slice = arrays.Slice(i * y, y);

            var max = slice.Span[0];
            var maxIndex = 0;
            for (var j = 1; j < y; j++) {
                if (!(slice.Span[j] > max)) continue;
                max = slice.Span[j];
                maxIndex = j;
            }

            result[i] = maxIndex;
        }

        return result;
    }

    public void Dispose() {
        session.Dispose();
    }
}