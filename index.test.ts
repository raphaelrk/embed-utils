import { expect, test, describe } from "bun:test";
import { getCosineSimilarity, getEuclideanDistance, type Embedding } from "./index";

describe("getCosineSimilarity", () => {
  test("returns 1 for identical vectors", () => {
    const vec: Embedding = [1, 2, 3];
    expect(getCosineSimilarity(vec, vec)).toBe(1);
  });
  
  test("returns -1 for opposite vectors", () => {
    const vecA: Embedding = [1, 2, 3];
    const vecB: Embedding = [-1, -2, -3];
    expect(getCosineSimilarity(vecA, vecB)).toBe(-1);
  });
  
  test("returns 0 for orthogonal vectors", () => {
    const vecA: Embedding = [1, 0, 0];
    const vecB: Embedding = [0, 1, 0];
    expect(getCosineSimilarity(vecA, vecB)).toBe(0);
  });
  
  test("returns 0 for zero vectors", () => {
    const vecA: Embedding = [0, 0, 0];
    const vecB: Embedding = [1, 2, 3];
    expect(getCosineSimilarity(vecA, vecB)).toBe(0);
  });
  
  test("throws error for different length vectors", () => {
    const vecA: Embedding = [1, 2, 3];
    const vecB: Embedding = [1, 2];
    expect(() => getCosineSimilarity(vecA, vecB)).toThrow();
  });
  
  test("handles floating point numbers", () => {
    const vecA: Embedding = [0.5, 0.25, 0.1];
    const vecB: Embedding = [0.2, 0.4, 0.8];
    expect(getCosineSimilarity(vecA, vecB)).toBeCloseTo(0.537964389857286, 9);
  });
  
  test("handles very small numbers correctly", () => {
    const vecA: Embedding = [1e-10, 2e-10, 3e-10];
    const vecB: Embedding = [2e-10, 4e-10, 6e-10];
    expect(getCosineSimilarity(vecA, vecB)).toBe(1); // Should be parallel vectors
  });
  
  test("handles very large numbers correctly", () => {
    const vecA: Embedding = [1e10, 2e10, 3e10];
    const vecB: Embedding = [2e10, 4e10, 6e10];
    expect(getCosineSimilarity(vecA, vecB)).toBeCloseTo(1, 9); // Should be parallel vectors
  });
  
  test("handles mixed positive/negative numbers", () => {
    const vecA: Embedding = [1, -2, 3];
    const vecB: Embedding = [-1, 2, -3];
    expect(getCosineSimilarity(vecA, vecB)).toBe(-1);
  });
  
  test("handles longer vectors", () => {
    const vecA: Embedding = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const vecB: Embedding = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    expect(getCosineSimilarity(vecA, vecB)).toBeCloseTo(1, 9);
  });
  
  test("handles 2D vectors", () => {
    // Test with vectors that are parallel
    const vecA: Embedding = [3, 4];
    const vecB: Embedding = [6, 8];
    expect(getCosineSimilarity(vecA, vecB)).toBe(1); // cos(0°) = 1
    
    // Test with vectors that are rotated 90 degrees from each other
    const vecC: Embedding = [1, 0];
    const vecD: Embedding = [0, 1];
    expect(getCosineSimilarity(vecC, vecD)).toBe(0); // cos(90°) = 0
    
    // Test with vectors at 45 degrees
    const vecE: Embedding = [1, 1];
    const vecF: Embedding = [1, 0];
    expect(getCosineSimilarity(vecE, vecF)).toBeCloseTo(1 / Math.sqrt(2), 6); // cos(45°) ≈ 0.707
    
    // Test with vectors at 30 degrees
    const vecK: Embedding = [1, 0];
    const vecL: Embedding = [Math.sqrt(3)/2, 0.5];
    expect(getCosineSimilarity(vecK, vecL)).toBeCloseTo(Math.sqrt(3)/2, 6); // cos(30°) ≈ 0.866
    
    // Test with vectors at 120 degrees
    const vecG: Embedding = [1, 0];
    const vecH: Embedding = [-0.5, Math.sqrt(3)/2];
    expect(getCosineSimilarity(vecG, vecH)).toBeCloseTo(-0.5, 6); // cos(120°) ≈ -0.5
    
    // Test with vectors at 60 degrees
    const vecI: Embedding = [1, 0];
    const vecJ: Embedding = [0.5, Math.sqrt(3)/2];
    expect(getCosineSimilarity(vecI, vecJ)).toBeCloseTo(0.5, 6); // cos(60°) ≈ 0.5
  });
});

describe("getEuclideanDistance", () => {
  test("returns 0 for identical vectors", () => {
    const vec: Embedding = [1, 2, 3];
    expect(getEuclideanDistance(vec, vec)).toBe(0);
  });
  
  test("calculates distance correctly for simple cases", () => {
    const vecA: Embedding = [0, 0, 0];
    const vecB: Embedding = [1, 0, 0];
    expect(getEuclideanDistance(vecA, vecB)).toBe(1);
    
    const vecC: Embedding = [0, 0];
    const vecD: Embedding = [3, 4];
    expect(getEuclideanDistance(vecC, vecD)).toBe(5); // Pythagorean triple
  });
  
  test("throws error for different length vectors", () => {
    const vecA: Embedding = [1, 2, 3];
    const vecB: Embedding = [1, 2];
    expect(() => getEuclideanDistance(vecA, vecB)).toThrow();
  });
  
  test("handles floating point numbers", () => {
    const vecA: Embedding = [0.5, 0.25, 0.1];
    const vecB: Embedding = [0.2, 0.4, 0.8];
    expect(getEuclideanDistance(vecA, vecB)).toBeCloseTo(0.7762087348, 6);
  });
  
  test("handles very small numbers correctly", () => {
    const vecA: Embedding = [1e-10, 2e-10, 3e-10];
    const vecB: Embedding = [2e-10, 4e-10, 6e-10];
    expect(getEuclideanDistance(vecA, vecB)).toBeCloseTo(0.000000000374166, 6);
  });
  
  test("handles very large numbers correctly", () => {
    const vecA: Embedding = [1e10, 2e10, 3e10];
    const vecB: Embedding = [2e10, 4e10, 6e10];
    expect(getEuclideanDistance(vecA, vecB)).toBeCloseTo(37416573867.7394138558, 6);
  });
  
  test("handles higher dimensional vectors (5D)", () => {
    const vecA: Embedding = [1, 2, 3, 4, 5];
    const vecB: Embedding = [2, 3, 4, 5, 6];
    // Distance should be sqrt((1)^2 + (1)^2 + (1)^2 + (1)^2 + (1)^2) = sqrt(5)
    expect(getEuclideanDistance(vecA, vecB)).toBeCloseTo(Math.sqrt(5), 6);
  });
  
  test("is symmetric", () => {
    // Test symmetry across a variety of vector types
    const testVectors: Embedding[] = [
      [1, 2, 3],           // Regular integers
      [-4, -5, -6],        // Negative integers
      [0.1, 0.2, 0.3],     // Small decimals
      [1e5, 2e5, 3e5],     // Large numbers
      [1e-5, 2e-5, 3e-5],  // Very small numbers
      [0, 0, 0],           // Zero vector
      [Infinity, 2, 3],    // Special values
    ];
    
    // Test every combination of vectors
    for (let i = 0; i < testVectors.length; i++) {
      for (let j = i + 1; j < testVectors.length; j++) {
        const vecA = testVectors[i];
        const vecB = testVectors[j];
        expect(getEuclideanDistance(vecA, vecB)).toBe(getEuclideanDistance(vecB, vecA));
      }
    }
  });
  
  test("satisfies triangle inequality", () => {
    // Test triangle inequality across a variety of vector types
    const testVectors: Embedding[] = [
      [0, 0, 0],           // Origin
      [1, 1, 1],           // Unit vector
      [-1, -1, -1],        // Negative unit vector
      [0.5, 0.2, 0.1],     // Decimal values
      [2, 3, 4],           // Different components
      [-2, 1, -3],         // Mixed signs
      [10, 20, 30],        // Larger values
      [0.01, 0.02, 0.03],  // Small values
    ];
    
    // Test triangle inequality for all possible triplets
    for (let i = 0; i < testVectors.length; i++) {
      for (let j = 0; j < testVectors.length; j++) {
        for (let k = 0; k < testVectors.length; k++) {
          if (i !== j && j !== k && i !== k) {
            const vecA = testVectors[i];
            const vecB = testVectors[j];
            const vecC = testVectors[k];
            
            const distAB = getEuclideanDistance(vecA, vecB);
            const distBC = getEuclideanDistance(vecB, vecC);
            const distAC = getEuclideanDistance(vecA, vecC);
            
            const epsilon = 1e-10;
            expect(distAC).toBeLessThanOrEqual(distAB + distBC + epsilon);
          }
        }
      }
    }
  });
});

// References on floating point math:
// - https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
// - https://nhigham.com/2020/05/04/what-is-floating-point-arithmetic/
// - https://en.wikipedia.org/wiki/IEEE_754
// - https://stackoverflow.com/questions/588004/is-floating-point-math-broken
// - https://tc39.es/ecma262/multipage/numbers-and-dates.html#sec-math.sqrt
