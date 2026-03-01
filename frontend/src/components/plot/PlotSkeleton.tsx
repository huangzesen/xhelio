export function PlotSkeleton() {
  return (
    <div className="border-b border-border bg-panel p-4">
      <div className="animate-pulse space-y-3">
        <div className="h-4 bg-surface-elevated rounded w-1/4" />
        <div className="h-[460px] bg-surface-elevated rounded-lg" />
        <div className="flex justify-between">
          <div className="h-3 bg-surface-elevated rounded w-1/6" />
          <div className="h-3 bg-surface-elevated rounded w-1/6" />
        </div>
      </div>
    </div>
  );
}
