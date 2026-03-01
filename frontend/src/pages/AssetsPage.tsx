import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import * as Collapsible from '@radix-ui/react-collapsible';
import {
  ArrowLeft,
  ChevronDown,
  Database,
  HardDrive,
  Loader2,
  RefreshCw,
  Satellite,
  Server,
  Trash2,
} from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../components/ui/dialog';
import { useAssetStore } from '../stores/assetStore';
import type { AssetCategory, DirStats } from '../api/types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CATEGORY_META: Record<
  string,
  { label: string; slug: string; icon: typeof Database; color: string; barColor: string }
> = {
  cdf_cache: {
    label: 'CDF Cache',
    slug: 'cdf-cache',
    icon: Database,
    color: 'text-badge-blue-text',
    barColor: 'bg-badge-blue-text',
  },
  ppi_cache: {
    label: 'PPI Cache',
    slug: 'ppi-cache',
    icon: Server,
    color: 'text-badge-purple-text',
    barColor: 'bg-badge-purple-text',
  },
  sessions: {
    label: 'Sessions',
    slug: 'sessions',
    icon: HardDrive,
    color: 'text-badge-teal-text',
    barColor: 'bg-badge-teal-text',
  },
  spice_kernels: {
    label: 'SPICE Kernels',
    slug: 'spice-kernels',
    icon: Satellite,
    color: 'text-badge-orange-text',
    barColor: 'bg-badge-orange-text',
  },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  let v = n;
  for (const unit of ['KB', 'MB', 'GB', 'TB']) {
    v /= 1024;
    if (v < 1024 || unit === 'TB') return `${v.toFixed(1)} ${unit}`;
  }
  return `${v.toFixed(1)} TB`;
}

function formatDate(iso: string | null): string {
  if (!iso) return '\u2014';
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}

// ---------------------------------------------------------------------------
// Usage bar
// ---------------------------------------------------------------------------

function UsageBar({ categories }: { categories: AssetCategory[] }) {
  const total = categories.reduce((s, c) => s + c.total_bytes, 0);
  if (total === 0) return null;

  return (
    <div className="flex h-3 w-full rounded-full overflow-hidden bg-border-subtle">
      {categories.map((cat) => {
        const pct = (cat.total_bytes / total) * 100;
        if (pct < 0.5) return null;
        const meta = CATEGORY_META[cat.name];
        return (
          <div
            key={cat.name}
            className={`${meta?.barColor ?? 'bg-text-muted'} transition-all duration-500 first:rounded-l-full last:rounded-r-full`}
            style={{ width: `${pct}%` }}
            title={`${meta?.label}: ${formatBytes(cat.total_bytes)} (${pct.toFixed(1)}%)`}
          />
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Category card
// ---------------------------------------------------------------------------

function CategoryCard({
  category,
  onClean,
  cleaning,
}: {
  category: AssetCategory;
  onClean: (slug: string, targets: string[], olderThanDays: number | null, emptyOnly: boolean) => void;
  cleaning: string | null;
}) {
  const [open, setOpen] = useState(category.total_bytes > 0);
  const [olderDays, setOlderDays] = useState<string>('');
  const meta = CATEGORY_META[category.name];
  if (!meta) return null;

  const Icon = meta.icon;
  const isBusy = cleaning === meta.slug;
  const isSession = category.name === 'sessions';

  return (
    <Collapsible.Root open={open} onOpenChange={setOpen}>
      <div className="bg-panel rounded-xl border border-border overflow-hidden">
        {/* Header */}
        <Collapsible.Trigger asChild>
          <button className="w-full flex items-center gap-3 px-4 py-3.5 text-left hover:bg-hover-bg transition-colors">
            <Icon size={16} className={meta.color} />
            <span className="font-medium text-text text-sm flex-1">{meta.label}</span>
            <span className="text-xs font-mono text-text-muted">
              {formatBytes(category.total_bytes)}
            </span>
            <span className="text-xs text-text-muted tabular-nums">
              {category.file_count.toLocaleString()} files
            </span>
            <ChevronDown
              size={14}
              className={`text-text-muted transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
            />
          </button>
        </Collapsible.Trigger>

        {/* Content */}
        <Collapsible.Content>
          <div className="border-t border-border">
            {category.subcategories.length === 0 ? (
              <div className="px-4 py-6 text-center text-xs text-text-muted">No data</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border-subtle text-text-muted">
                      <th className="text-left font-normal px-4 py-2">Name</th>
                      <th className="text-right font-normal px-3 py-2">Size</th>
                      <th className="text-right font-normal px-3 py-2">Files</th>
                      <th className="text-right font-normal px-3 py-2 hidden sm:table-cell">Oldest</th>
                      <th className="text-right font-normal px-3 py-2 hidden sm:table-cell">Newest</th>
                      {isSession && (
                        <>
                          <th className="text-right font-normal px-3 py-2">Turns</th>
                          <th className="text-left font-normal px-3 py-2 hidden md:table-cell">Session</th>
                        </>
                      )}
                      <th className="px-3 py-2 w-8" />
                    </tr>
                  </thead>
                  <tbody>
                    {category.subcategories.map((sub) => (
                      <SubRow
                        key={sub.name}
                        sub={sub}
                        slug={meta.slug}
                        isSession={isSession}
                        onClean={onClean}
                        isBusy={isBusy}
                      />
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Bulk actions */}
            {category.subcategories.length > 0 && (
              <div className="flex flex-wrap items-center gap-2 px-4 py-3 border-t border-border-subtle bg-surface/50">
                <button
                  onClick={() => onClean(meta.slug, [], null, false)}
                  disabled={isBusy}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs
                    border border-status-error-border text-status-error-text
                    hover:bg-status-error-bg transition-colors disabled:opacity-40"
                >
                  {isBusy ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />}
                  Clean All
                </button>

                {isSession && (
                  <button
                    onClick={() => onClean(meta.slug, [], null, true)}
                    disabled={isBusy}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs
                      border border-border text-text-muted
                      hover:bg-hover-bg transition-colors disabled:opacity-40"
                  >
                    Clean Empty
                  </button>
                )}

                <div className="flex items-center gap-1.5 ml-auto">
                  <span className="text-xs text-text-muted">Older than</span>
                  <input
                    type="number"
                    min={1}
                    placeholder="N"
                    value={olderDays}
                    onChange={(e) => setOlderDays(e.target.value)}
                    className="w-14 rounded-lg border border-border px-2 py-1 text-xs
                      bg-input-bg text-text text-center tabular-nums
                      focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
                  />
                  <span className="text-xs text-text-muted">days</span>
                  <button
                    onClick={() => {
                      const days = parseInt(olderDays, 10);
                      if (days > 0) onClean(meta.slug, [], days, false);
                    }}
                    disabled={isBusy || !olderDays || parseInt(olderDays, 10) <= 0}
                    className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs
                      border border-border text-text hover:bg-hover-bg transition-colors disabled:opacity-40"
                  >
                    Clean
                  </button>
                </div>
              </div>
            )}
          </div>
        </Collapsible.Content>
      </div>
    </Collapsible.Root>
  );
}

// ---------------------------------------------------------------------------
// Subcategory row
// ---------------------------------------------------------------------------

function SubRow({
  sub,
  slug,
  isSession,
  onClean,
  isBusy,
}: {
  sub: DirStats;
  slug: string;
  isSession: boolean;
  onClean: (slug: string, targets: string[], olderThanDays: number | null, emptyOnly: boolean) => void;
  isBusy: boolean;
}) {
  return (
    <tr className="border-b border-border-subtle last:border-b-0 hover:bg-hover-bg/50 transition-colors">
      <td className="px-4 py-2 font-mono text-text truncate max-w-[180px]" title={sub.name}>
        {sub.name}
      </td>
      <td className="px-3 py-2 text-right tabular-nums text-text-muted">{formatBytes(sub.total_bytes)}</td>
      <td className="px-3 py-2 text-right tabular-nums text-text-muted">{sub.file_count}</td>
      <td className="px-3 py-2 text-right text-text-muted hidden sm:table-cell">{formatDate(sub.oldest_mtime)}</td>
      <td className="px-3 py-2 text-right text-text-muted hidden sm:table-cell">{formatDate(sub.newest_mtime)}</td>
      {isSession && (
        <>
          <td className="px-3 py-2 text-right tabular-nums text-text-muted">
            {sub.round_count != null ? sub.round_count : '\u2014'}
          </td>
          <td className="px-3 py-2 text-left text-text-muted truncate max-w-[140px] hidden md:table-cell" title={sub.session_name ?? ''}>
            {sub.session_name || ''}
          </td>
        </>
      )}
      <td className="px-3 py-2">
        <button
          onClick={() => onClean(slug, [sub.name], null, false)}
          disabled={isBusy}
          className="p-1 rounded hover:bg-hover-danger-bg text-text-muted hover:text-status-error-text
            transition-colors disabled:opacity-30"
          title={`Delete ${sub.name}`}
        >
          <Trash2 size={12} />
        </button>
      </td>
    </tr>
  );
}

// ---------------------------------------------------------------------------
// Confirm dialog
// ---------------------------------------------------------------------------

interface CleanAction {
  slug: string;
  targets: string[];
  olderThanDays: number | null;
  emptyOnly: boolean;
}

function describeAction(action: CleanAction): string {
  const meta = Object.values(CATEGORY_META).find((m) => m.slug === action.slug);
  const label = meta?.label ?? action.slug;
  const parts: string[] = [];

  if (action.targets.length > 0) {
    parts.push(`${action.targets.join(', ')} from ${label}`);
  } else {
    parts.push(`all ${label}`);
  }
  if (action.olderThanDays) {
    parts.push(`older than ${action.olderThanDays} days`);
  }
  if (action.emptyOnly) {
    parts.push('(empty sessions only)');
  }
  return parts.join(' ');
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export function AssetsPage() {
  const { overview, loading, cleaning, cleanResult, error, loadOverview, clean, clearResult } =
    useAssetStore();

  const [pendingAction, setPendingAction] = useState<CleanAction | null>(null);
  const [resultBanner, setResultBanner] = useState<{ text: string; freed: string } | null>(null);

  useEffect(() => {
    loadOverview();
  }, [loadOverview]);

  // Show result banner when clean completes
  useEffect(() => {
    if (cleanResult && !cleanResult.dry_run) {
      setResultBanner({
        text: `Deleted ${cleanResult.deleted_count} files`,
        freed: cleanResult.freed_human,
      });
      clearResult();
      // Refresh data
      loadOverview();
      // Auto-dismiss after 6s
      const t = setTimeout(() => setResultBanner(null), 6000);
      return () => clearTimeout(t);
    }
  }, [cleanResult, clearResult, loadOverview]);

  const handleRequestClean = useCallback(
    (slug: string, targets: string[], olderThanDays: number | null, emptyOnly: boolean) => {
      setPendingAction({ slug, targets, olderThanDays, emptyOnly });
    },
    [],
  );

  const handleConfirmClean = useCallback(() => {
    if (!pendingAction) return;
    clean(pendingAction.slug, {
      targets: pendingAction.targets.length > 0 ? pendingAction.targets : undefined,
      older_than_days: pendingAction.olderThanDays ?? undefined,
      empty_only: pendingAction.emptyOnly,
    });
    setPendingAction(null);
  }, [pendingAction, clean]);

  const totalFiles = useMemo(
    () => overview?.categories.reduce((s, c) => s + c.file_count, 0) ?? 0,
    [overview],
  );

  // Loading state
  if (loading && !overview) {
    return (
      <div className="flex-1 flex items-center justify-center bg-surface">
        <Loader2 size={20} className="animate-spin text-text-muted" />
        <span className="ml-2 text-text-muted text-sm">Scanning disk...</span>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-6 bg-surface">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3">
          <Link
            to="/settings"
            className="inline-flex items-center gap-1 text-xs text-text-muted hover:text-text transition-colors"
          >
            <ArrowLeft size={14} />
            Settings
          </Link>
        </div>

        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-text">Disk Assets</h1>
            {overview && (
              <p className="text-xs text-text-muted mt-1">
                {formatBytes(overview.total_bytes)} across {totalFiles.toLocaleString()} files
                <span className="mx-1.5 opacity-40">&middot;</span>
                scanned in {overview.scan_time_ms} ms
              </p>
            )}
          </div>
          <button
            onClick={loadOverview}
            disabled={loading}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs
              border border-border text-text-muted hover:text-text hover:bg-hover-bg
              transition-colors disabled:opacity-40"
          >
            {loading ? (
              <Loader2 size={12} className="animate-spin" />
            ) : (
              <RefreshCw size={12} />
            )}
            Rescan
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-status-error-bg border border-status-error-border text-status-error-text rounded-lg px-4 py-3 text-xs">
            {error}
          </div>
        )}

        {/* Result banner */}
        {resultBanner && (
          <div className="bg-status-success-bg border border-status-success-border text-status-success-text rounded-lg px-4 py-3 text-sm flex items-center justify-between animate-in fade-in slide-in-from-top-2 duration-300">
            <span>
              {resultBanner.text} &mdash; freed <strong>{resultBanner.freed}</strong>
            </span>
            <button onClick={() => setResultBanner(null)} className="ml-3 opacity-60 hover:opacity-100">
              &times;
            </button>
          </div>
        )}

        {/* Usage bar */}
        {overview && overview.total_bytes > 0 && (
          <div className="space-y-2">
            <UsageBar categories={overview.categories} />
            <div className="flex flex-wrap gap-x-5 gap-y-1">
              {overview.categories.map((cat) => {
                const meta = CATEGORY_META[cat.name];
                if (!meta || cat.total_bytes === 0) return null;
                return (
                  <div key={cat.name} className="flex items-center gap-1.5 text-xs text-text-muted">
                    <div className={`w-2 h-2 rounded-full ${meta.barColor}`} />
                    <span>{meta.label}</span>
                    <span className="font-mono tabular-nums">{formatBytes(cat.total_bytes)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Category cards */}
        {overview && (
          <div className="space-y-3">
            {overview.categories.map((cat) => (
              <CategoryCard
                key={cat.name}
                category={cat}
                onClean={handleRequestClean}
                cleaning={cleaning}
              />
            ))}
          </div>
        )}

        {/* Empty state */}
        {overview && overview.total_bytes === 0 && (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <HardDrive size={48} className="text-text-muted mb-4 opacity-20" />
            <p className="text-text-muted text-sm">No cached data on disk</p>
          </div>
        )}
      </div>

      {/* Confirmation dialog */}
      <Dialog open={pendingAction !== null} onOpenChange={(open) => !open && setPendingAction(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Confirm Cleanup</DialogTitle>
            <DialogDescription>
              This will permanently delete {pendingAction ? describeAction(pendingAction) : ''}. This action
              cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <button
              onClick={() => setPendingAction(null)}
              className="px-4 py-2 rounded-lg text-sm border border-border text-text
                hover:bg-hover-bg transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleConfirmClean}
              className="px-4 py-2 rounded-lg text-sm bg-status-error-text text-white
                hover:opacity-90 transition-opacity"
            >
              Delete
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
