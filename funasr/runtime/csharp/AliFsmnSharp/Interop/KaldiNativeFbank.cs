using System.Reflection;
using System.Runtime.InteropServices;

namespace AliFsmnSharp.Interop; 

public static class KaldiNativeFbank
{
        
#if LINUX
        private const string dllName = "kaldi-native-fbank-dll.so.0";
        private const string LibRelativePath = @"runtimes\linux-x64\native\";
#elif OSX
        private const string dllName = "kaldi-native-fbank-dll.dylib";
        private const string LibRelativePath = @"runtimes\osx-x64\native\";
#else
    private const string LibFbank = "kaldi-native-fbank-dll.dll";
    private const string LibRelativePath = @"runtimes\win-x64\native\";
#endif

    static KaldiNativeFbank() {
        NativeLibrary.SetDllImportResolver(typeof(KaldiNativeFbank).Assembly, ImportResolver);
    }

    private static IntPtr ImportResolver(string libraryName, Assembly assembly, DllImportSearchPath? searchPath) {
        return NativeLibrary.Load(Path.Combine(Path.GetDirectoryName(assembly.Location)!, LibRelativePath, libraryName));
    }

    [DllImport(LibFbank, EntryPoint = "GetFbankOptions", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr GetFbankOptions(float dither, bool snip_edges, float sample_rate, int num_bins, float frame_shift = 10.0f, float frame_length = 25.0f, float energy_floor = 0.0f, bool debug_mel = false, string window_type = "hamming");

    [DllImport(LibFbank, EntryPoint = "GetOnlineFbank", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    internal static extern KnfOnlineFbank GetOnlineFbank(IntPtr opts);

    [DllImport(LibFbank, EntryPoint = "AcceptWaveform", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void AcceptWaveform(KnfOnlineFbank knfOnlineFbank, float sample_rate, float[] samples, int samples_size);

    [DllImport(LibFbank, EntryPoint = "InputFinished", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void InputFinished(KnfOnlineFbank knfOnlineFbank);

    [DllImport(LibFbank, EntryPoint = "GetNumFramesReady", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int GetNumFramesReady(KnfOnlineFbank knfOnlineFbank);

    [DllImport(LibFbank, EntryPoint = "AcceptWaveformxxx", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    internal static extern FbankDatas AcceptWaveformxxx(KnfOnlineFbank knfOnlineFbank, float sample_rate, float[] samples, int samples_size);

    [DllImport(LibFbank, EntryPoint = "GetFbank", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void GetFbank(KnfOnlineFbank knfOnlineFbank, int frame, ref FbankData pData);

    [DllImport(LibFbank, EntryPoint = "GetFbanks", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void GetFbanks(KnfOnlineFbank knfOnlineFbank, int framesNum, ref FbankDatas fbankDatas);
        
}