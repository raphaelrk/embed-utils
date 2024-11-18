# embed-utils

Created to test out publishing to npm / using `bun publish`.

Simple utilities for working with vector embeddings in TypeScript/JavaScript.

## Installation

```bash
bun install embed-utils
# or use another package manager:
# - `npm install embed-utils`
# - `pnpm install embed-utils`
# - `yarn add embed-utils`
```

## Usage

```typescript
import { getCosineSimilarity, getEuclideanDistance, type Embedding } from 'embed-utils';

const embedding1: Embedding = [0.5, 0.2, 0.1];
const embedding2: Embedding = [0.4, 0.3, 0.2];

// Get cosine similarity (angle-based similarity)
const similarity = getCosineSimilarity(embedding1, embedding2);
console.log(similarity); // Outputs a number between -1 and 1

// Get Euclidean distance (straight-line distance)
const distance = getEuclideanDistance(embedding1, embedding2);
console.log(distance); // Outputs a non-negative number
```

## API

### Types

- `Embedding`: A type alias for `number[]`, representing a vector embedding.

### Functions

- `getCosineSimilarity(a: Embedding, b: Embedding): number`
  - Calculates the cosine similarity between two embeddings
  - Returns a value between -1 and 1, where:
    - 1 means the vectors are identical in direction
    - 0 means the vectors are orthogonal (perpendicular)
    - -1 means the vectors are opposite in direction
  - Throws an error if the embeddings have different lengths

- `getEuclideanDistance(a: Embedding, b: Embedding): number`
  - Calculates the Euclidean (straight-line) distance between two embeddings
  - Returns a non-negative number where:
    - 0 means the vectors are identical
    - Larger values indicate greater distance between the vectors
  - Throws an error if the embeddings have different lengths

## License

MIT
