using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YamlDotNet.Serialization;

namespace AliFsmnSharp.Model; 

public class FrontendConfEntity
{
    public int fs { get; set; } = 16000;
    public string window { get; set; } = "hamming";
    public int n_mels { get; set; } = 80;
    public int frame_length { get; set; } = 25;
    public int frame_shift { get; set; } = 10;
    public float dither { get; set; } = 0.0F;
    public int lfr_m { get; set; } = 5;
    public int lfr_n { get; set; } = 1;
}