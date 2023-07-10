namespace AliFsmnSharp.Model; 

public class AsrModelConfEntity {
    public float ctc_weight { get; set; }
    public float lsm_weight { get; set; } = 0.1f;
    public bool length_normalized_loss { get; set; } = true;
    public float predictor_weight { get; set; } = 1.0f;
    public int predictor_bias { get; set; } = 1;
    public float sampling_ratio { get; set; } = 0.75f;
}