import { useEffect } from 'react';
import { useGalleryStore } from '../stores/galleryStore';
import { PipelineDAG } from '../components/pipeline/PipelineDAG';
import { Image, Trash2, Play, Loader2, ExternalLink, X } from 'lucide-react';

export function GalleryPage() {
  const {
    items,
    selectedItem,
    replaying,
    replayResult,
    loading,
    error,
    loadItems,
    deleteItem,
    replayItem,
    selectItem,
  } = useGalleryStore();

  useEffect(() => {
    loadItems();
  }, [loadItems]);

  if (loading && items.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center bg-surface">
        <Loader2 size={20} className="animate-spin text-text-muted" />
        <span className="ml-2 text-text-muted text-sm">Loading gallery...</span>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 bg-surface">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-xl font-semibold text-text mb-6">Visualization Gallery</h1>

        {error && (
          <div className="text-xs text-status-error-text bg-status-error-bg rounded px-3 py-2 mb-4">{error}</div>
        )}

        {items.length === 0 ? (
          /* Empty state */
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <Image size={48} className="text-text-muted mb-4 opacity-30" />
            <p className="text-text-muted text-sm mb-2">No saved visualizations yet</p>
            <p className="text-text-muted text-xs max-w-sm">
              Save data products from the Pipeline page to build your gallery.
              Select a session, choose a render product, and click "Save to Gallery".
            </p>
          </div>
        ) : (
          <>
            {/* Card grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {items.map((item) => (
                <button
                  key={item.id}
                  onClick={() => selectItem(selectedItem?.id === item.id ? null : item)}
                  className={`relative group rounded-xl border overflow-hidden text-left transition-all
                    ${selectedItem?.id === item.id
                      ? 'border-primary ring-2 ring-primary/20'
                      : 'border-border hover:border-primary/40'
                    } bg-panel`}
                >
                  {/* Thumbnail */}
                  <div className="aspect-[16/10] bg-surface overflow-hidden">
                    <img
                      src={`/api/gallery/${item.id}/thumbnail`}
                      alt={item.name}
                      className="w-full h-full object-contain"
                      loading="lazy"
                    />
                  </div>

                  {/* Info */}
                  <div className="p-3">
                    <div className="text-sm font-medium text-text truncate">{item.name}</div>
                    <div className="text-xs text-text-muted mt-1">
                      {new Date(item.created_at).toLocaleDateString(undefined, {
                        year: 'numeric', month: 'short', day: 'numeric',
                        hour: '2-digit', minute: '2-digit',
                      })}
                    </div>
                  </div>

                  {/* Delete button */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteItem(item.id);
                    }}
                    className="absolute top-2 right-2 p-1.5 rounded-lg
                      bg-surface/80 backdrop-blur opacity-0 group-hover:opacity-100
                      hover:bg-status-error-bg hover:text-status-error-text
                      transition-all text-text-muted"
                    title="Delete"
                  >
                    <Trash2 size={14} />
                  </button>
                </button>
              ))}
            </div>

            {/* Expanded view for selected item */}
            {selectedItem && (
              <div className="mt-6 bg-panel rounded-xl border border-border p-4 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-base font-medium text-text">{selectedItem.name}</h3>
                    <p className="text-xs text-text-muted mt-0.5">
                      Session: {selectedItem.session_id.slice(0, 12)}... | Render: {selectedItem.render_op_id}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => replayItem(selectedItem.id)}
                      disabled={replaying}
                      className="flex items-center gap-2 px-4 py-2 rounded-lg
                        bg-primary text-white hover:bg-primary-dark transition-colors
                        disabled:opacity-50 text-sm"
                    >
                      {replaying ? (
                        <Loader2 size={14} className="animate-spin" />
                      ) : (
                        <Play size={14} />
                      )}
                      Replay
                    </button>
                    <button
                      onClick={() => selectItem(null)}
                      className="p-2 rounded-lg text-text-muted hover:bg-hover-bg transition-colors"
                    >
                      <X size={16} />
                    </button>
                  </div>
                </div>

                {/* Replay status */}
                {replayResult && (
                  <div className="text-xs space-y-1">
                    <div className={replayResult.errors.length > 0 ? 'text-status-error-text' : 'text-status-success-text'}>
                      {replayResult.steps_completed}/{replayResult.steps_total} steps completed
                    </div>
                    {replayResult.errors.map((err, i) => (
                      <div key={i} className="text-status-error-text font-mono">
                        {err.op_id}: {err.error}
                      </div>
                    ))}
                  </div>
                )}

                {/* Inline Plotly figure */}
                {replayResult?.figure && (
                  <div className="overflow-hidden rounded-lg">
                    <PipelineDAG figure={replayResult.figure} />
                  </div>
                )}

                {/* Large figure — open in new tab */}
                {replayResult && !replayResult.figure && replayResult.figure_url && (
                  <div className="flex items-center justify-between p-4 bg-surface rounded-lg">
                    <div>
                      <h4 className="text-sm font-medium text-text">Large figure</h4>
                      <p className="text-xs text-text-muted mt-1">
                        This figure is too large to display inline.
                      </p>
                    </div>
                    <a
                      href={replayResult.figure_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="shrink-0 ml-4 flex items-center gap-2 px-4 py-2 rounded-lg
                        bg-primary text-white text-sm font-medium hover:bg-primary-dark transition-colors"
                    >
                      <ExternalLink size={14} />
                      Open in new tab
                    </a>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
