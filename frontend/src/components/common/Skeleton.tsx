interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className = '' }: SkeletonProps) {
  return (
    <div
      className={`animate-pulse rounded bg-border/50 ${className}`}
    />
  );
}

export function ChatSkeleton() {
  return (
    <div className="space-y-4 p-4 max-w-3xl mx-auto">
      {/* User message skeleton */}
      <div className="flex gap-3 flex-row-reverse">
        <Skeleton className="w-8 h-8 rounded-full shrink-0" />
        <Skeleton className="h-10 w-48 rounded-xl" />
      </div>
      {/* Agent message skeleton */}
      <div className="flex gap-3">
        <Skeleton className="w-8 h-8 rounded-full shrink-0" />
        <div className="space-y-2 flex-1 max-w-[80%]">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-4 w-1/2" />
        </div>
      </div>
      {/* Another user message */}
      <div className="flex gap-3 flex-row-reverse">
        <Skeleton className="w-8 h-8 rounded-full shrink-0" />
        <Skeleton className="h-10 w-36 rounded-xl" />
      </div>
    </div>
  );
}

export function PipelineSkeleton() {
  return (
    <div className="space-y-4">
      {/* DAG card skeleton */}
      <div className="bg-panel rounded-xl border border-border p-4 space-y-3">
        <Skeleton className="h-4 w-28" />
        <Skeleton className="h-[300px] w-full rounded-lg" />
      </div>
      {/* Operations table skeleton */}
      <div className="bg-panel rounded-xl border border-border p-4 space-y-3">
        <Skeleton className="h-4 w-24" />
        <div className="space-y-2">
          <Skeleton className="h-6 w-full" />
          <Skeleton className="h-6 w-full" />
          <Skeleton className="h-6 w-full" />
          <Skeleton className="h-6 w-3/4" />
        </div>
      </div>
    </div>
  );
}
