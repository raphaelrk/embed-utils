import { expect, test, describe } from "bun:test";
import { getCosineSimilarity, type Embedding } from "./index";

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

// References on floating point math:
// - https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
// - https://nhigham.com/2020/05/04/what-is-floating-point-arithmetic/
// - https://en.wikipedia.org/wiki/IEEE_754
// - https://stackoverflow.com/questions/588004/is-floating-point-math-broken
// - https://tc39.es/ecma262/multipage/numbers-and-dates.html#sec-math.sqrt
