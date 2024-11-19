import { run, bench, group, compact, summary } from 'mitata';
import { getCosineSimilarity, getEuclideanDistance, uniformRandomEmbedding } from "./index";

// Dimensions to test
const dimsToTest = [2, 3, 4, 128, 384, 768, 1024, 1536];

compact(() => {
  bench('unif rand embed ($dimension-d)', function* (state: { get: (key: string) => number }) {
    const dim = state.get('dimension');
    yield () => uniformRandomEmbedding(dim);
  }).args('dimension', dimsToTest);
});

// getCosineSimilarity
compact(() => {
  bench('getCosineSimilarity($dimension-d)', function* (state: { get: (key: string) => number }) {
    const dim = state.get('dimension');
    const vecA = uniformRandomEmbedding(dim);
    const vecB = uniformRandomEmbedding(dim);
    yield () => getCosineSimilarity(vecA, vecB);
  }).args('dimension', dimsToTest);
});

// getEuclideanDistance
compact(() => {
  bench('getEuclideanDistance($dimension-d)', function* (state: { get: (key: string) => number }) {
    const dim = state.get('dimension');
    const vecA = uniformRandomEmbedding(dim);
    const vecB = uniformRandomEmbedding(dim);
    yield () => getEuclideanDistance(vecA, vecB);
  }).args('dimension', dimsToTest);
});

// Run benchmarks
await run({ throw: false, colors: true, format: 'mitata' });
