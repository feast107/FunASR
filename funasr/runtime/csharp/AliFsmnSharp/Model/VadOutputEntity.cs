namespace AliFsmnSharp.Model; 

internal record VadOutputEntity(float[,,] Scores) {
    public List<float[]> OutCaches { get; } = new();
}