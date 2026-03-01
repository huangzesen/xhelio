import type { MemoryEntry } from '../../api/types';

/** Minimal shape shared by full MemoryEntry reviews and review_summary.reviews items. */
export interface ReviewLike {
  id: string;
  content: string;
  tags: string[];
  version: number;
  created_at: string;
  review_of: string;
}

/**
 * Build a lookup map from target memory ID → list of review MemoryEntries.
 * Reviews are Memory entries with type="review" and review_of set to the target ID.
 * Multiple reviews (from different agents) can exist per memory.
 */
export function buildReviewMap(memories: MemoryEntry[]): Record<string, MemoryEntry[]> {
  const map: Record<string, MemoryEntry[]> = {};
  for (const m of memories) {
    if (m.type === 'review' && m.review_of) {
      if (!map[m.review_of]) map[m.review_of] = [];
      map[m.review_of].push(m);
    }
  }
  return map;
}

/**
 * Parse the star rating from a review memory's content.
 * Content format: "N★ comment text"
 * Returns -1 if no review or unparseable.
 */
export function parseReviewStars(review: ReviewLike | null | undefined): number {
  if (!review) return -1;
  const match = review.content.match(/^(\d)★/);
  return match ? parseInt(match[1], 10) : -1;
}

/**
 * Compute average star rating across multiple reviews.
 * Returns -1 if no reviews.
 */
export function avgReviewStars(reviews: ReviewLike[]): number {
  if (reviews.length === 0) return -1;
  let total = 0;
  let count = 0;
  for (const r of reviews) {
    const s = parseReviewStars(r);
    if (s > 0) {
      total += s;
      count++;
    }
  }
  return count > 0 ? total / count : -1;
}

/**
 * Parse the comment text from a review memory's content.
 * Content format: "N★ comment text"
 * Returns the comment portion after the star prefix.
 */
export function parseReviewComment(review: ReviewLike): string {
  return review.content.replace(/^\d★\s*/, '');
}

/**
 * Parse the agent name from a review memory's tags.
 * Tags include the agent name (e.g., "VizAgent").
 */
export function parseReviewAgent(review: ReviewLike): string {
  for (const tag of review.tags) {
    if (!tag.startsWith('review:') && !tag.startsWith('stars:')) {
      return tag;
    }
  }
  return '';
}
