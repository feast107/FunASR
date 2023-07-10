namespace AliFsmnSharp.Interop; 

internal struct FbankData
{
    public IntPtr data;
    public int data_length;
};

internal struct FbankDatas
{
    public IntPtr data;
    public int data_length;
};

internal struct KnfOnlineFbank
{
    public IntPtr impl;
};