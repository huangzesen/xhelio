import { useEffect } from 'react';
import { useSavedPipelineStore } from '../../stores/savedPipelineStore';
import { SavedPipelineList } from './SavedPipelineList';
import { SavedPipelineDetail } from './SavedPipelineDetail';
import { SavedPipelineListSkeleton } from '../common/Skeleton';

export function SavedPipelinesTab() {
  const {
    pipelines,
    loading,
    error,
    selectedPipelineId,
    detail,
    dagFigure,
    detailLoading,
    selectedStep,
    executing,
    executeResult,
    loadPipelines,
    selectPipeline,
    clearSelection,
    executePipeline,
    deletePipeline,
    selectStep,
    updatePipeline,
    addFeedback,
  } = useSavedPipelineStore();

  useEffect(() => {
    loadPipelines();
  }, [loadPipelines]);

  // Detail view
  if (selectedPipelineId && detail) {
    return (
      <div className="flex-1 overflow-y-auto p-4 bg-surface">
        <SavedPipelineDetail
          detail={detail}
          dagFigure={dagFigure}
          selectedStep={selectedStep}
          executing={executing}
          executeResult={executeResult}
          error={error}
          onBack={clearSelection}
          onSelectStep={selectStep}
          onExecute={executePipeline}
          onDelete={deletePipeline}
          onUpdate={updatePipeline}
          onFeedback={addFeedback}
        />
      </div>
    );
  }

  // Loading detail
  if (selectedPipelineId && detailLoading) {
    return (
      <div className="flex-1 overflow-y-auto p-4 bg-surface">
        <SavedPipelineListSkeleton />
      </div>
    );
  }

  // List view
  return (
    <div className="flex-1 overflow-y-auto p-4 bg-surface">
      {loading && <SavedPipelineListSkeleton />}

      {!loading && error && (
        <div className="text-sm text-status-error-text bg-status-error-bg border border-status-error-border rounded-lg p-3">
          {error}
        </div>
      )}

      {!loading && !error && (
        <SavedPipelineList pipelines={pipelines} onSelect={selectPipeline} />
      )}
    </div>
  );
}
