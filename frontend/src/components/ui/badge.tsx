import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const badgeVariants = cva(
  'inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors',
  {
    variants: {
      variant: {
        default: 'bg-badge-gray-bg text-badge-gray-text',
        blue: 'bg-badge-blue-bg text-badge-blue-text',
        orange: 'bg-badge-orange-bg text-badge-orange-text',
        red: 'bg-badge-red-bg text-badge-red-text',
        pink: 'bg-badge-pink-bg text-badge-pink-text',
        green: 'bg-badge-green-bg text-badge-green-text',
        purple: 'bg-badge-purple-bg text-badge-purple-text',
        teal: 'bg-badge-teal-bg text-badge-teal-text',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <span className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
