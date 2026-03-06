import type { Variants, Transition } from 'framer-motion';

// Check for reduced motion preference
const prefersReducedMotion =
  typeof window !== 'undefined' &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const duration = prefersReducedMotion ? 0 : 0.2;
const staggerDelay = prefersReducedMotion ? 0 : 0.05;

export const transition: Transition = {
  duration,
  ease: [0.25, 0.1, 0.25, 1],
};

export const fadeSlideIn: Variants = {
  hidden: { opacity: 0, y: prefersReducedMotion ? 0 : 8 },
  visible: { opacity: 1, y: 0, transition },
  exit: { opacity: 0, y: prefersReducedMotion ? 0 : -4, transition: { duration: duration * 0.75 } },
};

export const fadeSlideInStagger: Variants = {
  hidden: { opacity: 0, y: prefersReducedMotion ? 0 : 6 },
  visible: { 
    opacity: 1, 
    y: 0, 
    transition: {
      duration,
      ease: [0.25, 0.1, 0.25, 1],
    } 
  },
  exit: { opacity: 0, y: prefersReducedMotion ? 0 : -3, transition: { duration: duration * 0.6 } },
};

export const fadeIn: Variants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition },
  exit: { opacity: 0, transition: { duration: duration * 0.75 } },
};

export const scaleIn: Variants = {
  hidden: { opacity: 0, scale: prefersReducedMotion ? 1 : 0.95 },
  visible: { opacity: 1, scale: 1, transition },
  exit: { opacity: 0, scale: prefersReducedMotion ? 1 : 0.95, transition: { duration: duration * 0.75 } },
};

export const stagger = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: staggerDelay,
    },
  },
};

export const staggerFast: Variants = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: prefersReducedMotion ? 0 : 0.03,
      delayChildren: prefersReducedMotion ? 0 : 0.05,
    },
  },
};

export const expandCollapse: Variants = {
  hidden: { height: 0, opacity: 0, overflow: 'hidden' as const },
  visible: { height: 'auto', opacity: 1, overflow: 'hidden' as const, transition: { duration } },
  exit: { height: 0, opacity: 0, overflow: 'hidden' as const, transition: { duration: duration * 0.75 } },
};
