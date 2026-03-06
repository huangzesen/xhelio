import { Star } from 'lucide-react';

export function StarRating({ stars }: { stars: number }) {
  return (
    <span className="inline-flex gap-px">
      {[1, 2, 3, 4, 5].map((i) => (
        <Star
          key={i}
          size={11}
          className={i <= stars ? 'fill-amber-400 text-amber-400' : 'text-text-muted/30'}
        />
      ))}
    </span>
  );
}
