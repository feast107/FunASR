using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliFsmnSharp.Model;

public class SegmentEntity {
    public List<TimeWindow> TimeWindows { get; } = new();
}