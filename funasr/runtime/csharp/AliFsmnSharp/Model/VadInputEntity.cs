namespace AliFsmnSharp.Model;

internal class VadInputEntity {
    public float[] Speech { get; set; } = Array.Empty<float>();
    public int SpeechLength { get; set; }
    public List<float[]> InCaches { get; set; } = new();
    public float[]? Waveform { get; set; }
}