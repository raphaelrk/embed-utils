# embedding-helpers

Simple utilities for working with vector embeddings in TypeScript/JavaScript.

## Installation

```bash
bun install embedding-helpers
# or use another package manager:
# - `npm install embedding-helpers`
# - `pnpm install embedding-helpers`
# - `yarn add embedding-helpers`
```

## Usage

```typescript
import { getCosineSimilarity, type Embedding } from 'embedding-helpers';

const embedding1: Embedding = [0.5, 0.2, 0.1];
const embedding2: Embedding = [0.4, 0.3, 0.2];

const similarity = getCosineSimilarity(embedding1, embedding2);
console.log(similarity); // Outputs a number between -1 and 1
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

## License

MIT
