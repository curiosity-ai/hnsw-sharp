using System;
using System.Diagnostics;

namespace HNSW.Net.SiftBenchmark
{
    public class ConsoleProgress : IProgressReporter
    {
        private Stopwatch _sw;
        private int _lastValue;
        public void Progress(int current, int total)
        {
            if (_sw is null) _sw = Stopwatch.StartNew();
            if (_sw.Elapsed.TotalSeconds > 5)
            {
                Console.WriteLine($"At {current:n0} of {total:n0} or {100f * current / total:n1}% or {(current-_lastValue)/(_sw.Elapsed.TotalSeconds):n0}/s");
                _sw.Restart();
                _lastValue = current;
            }
        }
    }
}
