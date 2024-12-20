
/**
* Type representing a vector embedding as an array of numbers
*/
export type Embedding = number[];

/**
 * Generates a uniformly random embedding vector with values between -1 and 1
 *
 * Note via Claude: Random distributions may not be ideal for simulating embeddings:
 * - Real embeddings often lie on/near lower-dimensional manifolds, not uniformly distributed
 * - Most random distributions (uniform, Gaussian, etc) produce roughly orthogonal vectors in high dimensions
 * - Random vectors don't preserve semantic relationships like real embeddings do
 * Consider using domain-specific techniques if trying to simulate realistic embeddings
 */
export function uniformRandomEmbedding(dimension: number): Embedding {
  return Array.from({ length: dimension }, () => Math.random() * 2 - 1);
}

/**
* Calculates the cosine similarity between two embeddings
* @param a First embedding vector
* @param b Second embedding vector
* @returns A value between -1 and 1, where 1 means most similar
* @throws Error if embeddings have different lengths
*/
export function getCosineSimilarity(a: Embedding, b: Embedding): number {
  if (a.length !== b.length) {
    throw new Error('Embeddings must have the same length');
  }

  // Calculate dot product and magnitudes
  let dotProduct = 0;
  let magnitudeA = 0;
  let magnitudeB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    magnitudeA += a[i] * a[i];
    magnitudeB += b[i] * b[i];
  }

  magnitudeA = Math.sqrt(magnitudeA);
  magnitudeB = Math.sqrt(magnitudeB);

  // Prevent division by zero
  if (magnitudeA === 0 || magnitudeB === 0) {
    return 0;
  }

  return dotProduct / (magnitudeA * magnitudeB);
}

/**
* Calculates the Euclidean distance between two embeddings
* @param a First embedding vector
* @param b Second embedding vector
* @returns The Euclidean distance (always non-negative)
* @throws Error if embeddings have different lengths
*/
export function getEuclideanDistance(a: Embedding, b: Embedding): number {
  if (a.length !== b.length) {
    throw new Error('Embeddings must have the same length');
  }

  let sumSquaredDifferences = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sumSquaredDifferences += diff * diff;
  }

  return Math.sqrt(sumSquaredDifferences);
}
