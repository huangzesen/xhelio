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

export function SavedPipelineListSkeleton() {
  return (
    <div className="space-y-3">
      {/* Table header skeleton */}
      <div className="flex gap-4 pb-2 border-b border-border">
        <Skeleton className="h-3 w-[180px]" />
        <Skeleton className="h-3 flex-1" />
        <Skeleton className="h-3 w-[100px]" />
        <Skeleton className="h-3 w-[40px]" />
        <Skeleton className="h-3 w-[120px]" />
        <Skeleton className="h-3 w-[80px]" />
      </div>
      {/* 5 row skeletons */}
      {Array.from({ length: 5 }, (_, i) => (
        <div key={i} className="flex gap-4 py-2 border-b border-border/50">
          <Skeleton className="h-4 w-[180px]" />
          <Skeleton className="h-4 flex-1" />
          <div className="w-[100px] flex gap-1">
            <Skeleton className="h-4 w-10 rounded" />
            <Skeleton className="h-4 w-12 rounded" />
          </div>
          <Skeleton className="h-4 w-[40px]" />
          <Skeleton className="h-4 w-[120px]" />
          <Skeleton className="h-4 w-[80px]" />
        </div>
      ))}
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
