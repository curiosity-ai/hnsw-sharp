namespace HNSW.Net
{
    public interface IProgressReporter
    {
        void Progress(int current, int total);
    }
}
