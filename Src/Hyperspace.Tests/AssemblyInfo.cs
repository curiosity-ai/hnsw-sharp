using Microsoft.VisualStudio.TestTools.UnitTesting;

// The tests build indexes that use every core; running them one at a time keeps timings and memory predictable.
[assembly: DoNotParallelize]
