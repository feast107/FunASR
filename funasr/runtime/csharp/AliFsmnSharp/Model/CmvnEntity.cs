using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliFsmnSharp.Model; 

internal class CmvnEntity
{
    public List<float> Means { get; set; } = new();
    public List<float> Vars { get; set; } = new();
}