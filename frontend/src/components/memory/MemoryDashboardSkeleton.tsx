import { Skeleton } from '../common/Skeleton';

export function MemoryDashboardSkeleton() {
  return (
    <div className="space-y-4">
      {/* Stats row */}
      <div className="flex flex-wrap gap-3">
        <Skeleton className="h-16 w-[120px] rounded-lg" />
        <Skeleton className="h-16 flex-1 min-w-[200px] rounded-lg" />
        <Skeleton className="h-16 w-[280px] rounded-lg" />
      </div>

      {/* Filter bar */}
      <Skeleton className="h-10 w-full rounded-lg" />

      {/* Scope chips */}
      <div className="flex gap-2">
        <Skeleton className="h-7 w-16 rounded-full" />
        <Skeleton className="h-7 w-24 rounded-full" />
        <Skeleton className="h-7 w-20 rounded-full" />
      </div>

      {/* Type tabs */}
      <Skeleton className="h-8 w-full rounded" />

      {/* Cards */}
      <div className="space-y-3">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="border border-border rounded-lg p-4 space-y-2">
            <div className="flex items-center gap-2">
              <Skeleton className="h-5 w-20 rounded" />
              <Skeleton className="h-5 w-16 rounded-full" />
            </div>
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
            <div className="flex gap-4 pt-1">
              <Skeleton className="h-3 w-20" />
              <Skeleton className="h-3 w-16" />
              <Skeleton className="h-3 w-24" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
