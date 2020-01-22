namespace HNSW.Net
{
    public interface IProgressReporter
    {
        void Progress(int step, int current, int total);
    }
}
