// Copyright (c) 2025 Relatude.DB - Proventus AS
// Licensed under the MIT License. https://github.com/Relatude/Relatude.DB/blob/main/LICENSE.txt

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;

namespace HNSW.Net
{
    public class SmallWorldTurboQuant
    {
        private SmallWorld<EncodedVector, float> _innerGraph;
        private TurboQuant _quantizer;
        private TurboQuantDistance _distance;
        public TurboQuant Quantizer => _quantizer;
        public SmallWorldParameters Parameters => _innerGraph.Parameters;
        public SmallWorldTurboQuant(TurboQuant quantizer, IProvideRandomValues generator, SmallWorldParameters parameters, bool threadSafe = true)
        {
            _quantizer = quantizer ?? throw new ArgumentNullException(nameof(quantizer));
            _distance = new TurboQuantDistance(_quantizer);
            _innerGraph = new SmallWorld<EncodedVector, float>(_distance.GetDistance, generator, parameters, threadSafe);
        }

        private SmallWorldTurboQuant(TurboQuant quantizer, SmallWorld<EncodedVector, float> innerGraph, IProvideRandomValues generator, bool threadSafe)
        {
            _quantizer = quantizer;
            _innerGraph = innerGraph;
            _distance = new TurboQuantDistance(_quantizer);
        }

        public IReadOnlyList<int> AddItems(IReadOnlyList<float[]> items, IProgressReporter progressReporter = null)
        {
            var encodedItems = new List<EncodedVector>(items.Count);
            foreach (var item in items)
            {
                encodedItems.Add(_quantizer.Encode(item));
            }
            return _innerGraph.AddItems(encodedItems, progressReporter);
        }

        public IList<SmallWorld<EncodedVector, float>.KNNSearchResult> KNNSearch(float[] item, int k, Func<EncodedVector, bool> filterItem = null, CancellationToken cancellationToken = default)
        {
            var encodedItem = _quantizer.Encode(item);
            return _innerGraph.KNNSearch(encodedItem, k, filterItem, cancellationToken);
        }

        public EncodedVector GetItem(int index)
        {
            return _innerGraph.GetItem(index);
        }

        public void SerializeGraph(Stream stream)
        {
            var quantizerBytes = _quantizer.ToByteArray();
            using (var bw = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
            {
                bw.Write(quantizerBytes.Length);
                bw.Write(quantizerBytes);
                bw.Flush();
            }
            _innerGraph.SerializeGraph(stream);
        }

        public static (SmallWorldTurboQuant Graph, EncodedVector[] ItemsNotInGraph) DeserializeGraph(IReadOnlyList<float[]> items, IProvideRandomValues generator, Stream stream, bool threadSafe = true)
        {
            byte[] quantizerBytes;
            using (var br = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true))
            {
                int len = br.ReadInt32();
                quantizerBytes = br.ReadBytes(len);
            }
            var quantizer = TurboQuant.FromByteArray(quantizerBytes);
            var distance = new TurboQuantDistance(quantizer);

            var encodedItems = new List<EncodedVector>(items.Count);
            foreach (var item in items)
            {
                encodedItems.Add(quantizer.Encode(item));
            }

            var result = SmallWorld<EncodedVector, float>.DeserializeGraph(encodedItems, distance.GetDistance, generator, stream, threadSafe);
            var wrapper = new SmallWorldTurboQuant(quantizer, result.Graph, generator, threadSafe);

            return (wrapper, result.ItemsNotInGraph);
        }

        public string Print()
        {
            return _innerGraph.Print();
        }

        public void OptimizeIfNeeded(bool force = false)
        {
            _innerGraph.OptimizeIfNeeded(force);
        }

        public void DisableDistanceCache()
        {
            _innerGraph.DisableDistanceCache();
        }

        public void ResizeDistanceCache(int newSize)
        {
            _innerGraph.ResizeDistanceCache(newSize);
        }
    }
}
