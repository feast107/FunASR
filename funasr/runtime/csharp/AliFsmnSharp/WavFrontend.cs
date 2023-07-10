using System.Runtime.InteropServices;
using AliFsmnSharp.Interop;
using AliFsmnSharp.Model;

namespace AliFsmnSharp; 

internal class WavFrontend {
    private readonly FrontendConfEntity _frontendConfEntity;
    private readonly IntPtr _opts;
    private readonly CmvnEntity? _cmvnEntity;

    public WavFrontend(string? mvnFilePath, FrontendConfEntity frontendConfEntity) {
        _frontendConfEntity = frontendConfEntity;
        _opts = KaldiNativeFbank.GetFbankOptions(
            dither: _frontendConfEntity.dither,
            snip_edges: true,
            sample_rate: _frontendConfEntity.fs,
            num_bins: _frontendConfEntity.n_mels
        );

        if (mvnFilePath != null) {
            _cmvnEntity = LoadCmvn(mvnFilePath);
        }
    }

    public float[][] GetFbank(float[] samples) {
        float sampleRate = _frontendConfEntity.fs;
        samples = samples.Select(x => x * 32768f).ToArray();
        var knfOnlineFbank = KaldiNativeFbank.GetOnlineFbank(_opts);
        KaldiNativeFbank.AcceptWaveform(knfOnlineFbank, sampleRate, samples, samples.Length);
        KaldiNativeFbank.InputFinished(knfOnlineFbank);
        var framesNum = KaldiNativeFbank.GetNumFramesReady(knfOnlineFbank);
        var fbank = new float[framesNum][];
        for (var i = 0; i < framesNum; i++) {
            var fbankData = new FbankData();
            KaldiNativeFbank.GetFbank(knfOnlineFbank, i, ref fbankData);
            fbank[i] = new float[_frontendConfEntity.n_mels];
            Marshal.Copy(fbankData.data, fbank[i], 0,
                Math.Min(fbankData.data_length / sizeof(float), fbank[i].Length));
            fbankData.data = IntPtr.Zero;
        }

        return fbank;
    }

    public float[][] LfrCmvn(float[][] features) {
        if (_frontendConfEntity.lfr_m != 1 || _frontendConfEntity.lfr_n != 1) {
            features = ApplyLfr(features, _frontendConfEntity.lfr_m, _frontendConfEntity.lfr_n);
        }

        features = ApplyCmvn(features);
        return features;
    }

    private static float[][] ApplyLfr(float[][] inputs, int lfrM, int lfrN) {
        var lfrInputs = new List<float[]>();
        var t = inputs.Length;
        var tLfr = (int)Math.Ceiling((double)t / lfrN);
        var leftPadding = Enumerable.Repeat(inputs[0], (lfrM - 1) / 2).ToArray();
        inputs = leftPadding.Concat(inputs).ToArray();
        t += (lfrM - 1) / 2;
        for (var i = 0; i < tLfr; i++) {
            if (lfrM <= t - i * lfrN) {
                var tempArray = inputs.Skip(i * lfrN).Take(lfrM).SelectMany(x => x).ToArray();
                lfrInputs.Add(tempArray);
            } else {
                // process last LFR frame
                var numPadding = lfrM - (t - i * lfrN);
                var frame = inputs.Skip(i * lfrN).Take(t - i * lfrN).SelectMany(x => x).ToArray();
                for (var _ = 0; _ < numPadding; _++) {
                    frame = frame.Concat(inputs[^1]).ToArray();
                }

                lfrInputs.Add(frame);
            }
        }

        return lfrInputs.ToArray();
    }


    private float[][] ApplyCmvn(float[][] inputs) {
        if (_cmvnEntity == null) {
            return inputs;
        }

        var frame = inputs.Length;
        var dim = inputs[0].Length;
        var means = Enumerable.Range(0, frame).Select(_ => _cmvnEntity.Means.Take(dim).ToArray()).ToArray();
        var vars = Enumerable.Range(0, frame).Select(_ => _cmvnEntity.Vars.Take(dim).ToArray()).ToArray();

        for (var i = 0; i < frame; i++) {
            for (var j = 0; j < dim; j++) {
                inputs[i][j] = (inputs[i][j] + means[i][j]) * vars[i][j];
            }
        }

        return inputs;
    }


    private CmvnEntity LoadCmvn(string mvnFilePath) {
        var meansList = new List<float>();
        var varsList = new List<float>();
        var options = new FileStreamOptions();
        options.Access = FileAccess.Read;
        options.Mode = FileMode.Open;
        var srtReader = new StreamReader(mvnFilePath, options);
        var i = 0;
        while (!srtReader.EndOfStream) {
            var strLine = srtReader.ReadLine();
            if (string.IsNullOrEmpty(strLine)) continue;

            if (strLine.StartsWith("<AddShift>")) {
                i = 1;
                continue;
            }

            if (strLine.StartsWith("<Rescale>")) {
                i = 2;
                continue;
            }

            if (strLine.StartsWith("<LearnRateCoef>") && i == 1) {
                var addShiftLine = strLine.Substring(
                    strLine.IndexOf("[", StringComparison.Ordinal) + 1,
                    strLine.LastIndexOf("]", StringComparison.Ordinal) -
                    strLine.IndexOf("[", StringComparison.Ordinal) - 1).Split(" ");
                meansList = addShiftLine
                    .Where(x => !string.IsNullOrEmpty(x))
                    .Select(x => float.Parse(x.Trim()))
                    .ToList();
                continue;
            }

            if (!strLine.StartsWith("<LearnRateCoef>") || i != 2) continue;

            var rescaleLine = strLine.Substring(
                strLine.IndexOf("[", StringComparison.Ordinal) + 1,
                strLine.LastIndexOf("]", StringComparison.Ordinal) -
                strLine.IndexOf("[", StringComparison.Ordinal) - 1).Split(" ");
            varsList = rescaleLine
                .Where(x => !string.IsNullOrEmpty(x))
                .Select(x => float.Parse(x.Trim()))
                .ToList();
        }

        var cmvnEntity = new CmvnEntity {
            Means = meansList,
            Vars = varsList
        };
        return cmvnEntity;
    }
}