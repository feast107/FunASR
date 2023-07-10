namespace AliFsmnSharp.Model; 

public class AsrYamlEntity
{
    public int input_size { get; set; }
    public string frontend { get; set; } = "wav_frontend";
    public string model { get; set; } = "paraformer";
    public string encoder { get; set; } = "sanm";
    public FrontendConfEntity frontend_conf { get; set; } = new();
    public AsrModelConfEntity model_conf { get; set; } = new();
    public List<string> token_list { get; set; } = new();
}