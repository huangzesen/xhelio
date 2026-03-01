import { useState } from 'react';
import { ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { expandCollapse } from '../common/MotionPresets';
import type { MissionValidation, DatasetValidation, ValidationRecord } from '../../api/types';

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return iso;
  }
}

// ---- ValidationRecordRow ----

function ValidationRecordRow({ record }: { record: ValidationRecord }) {
  return (
    <div className="flex items-center gap-2 py-1.5 text-xs">
      <span className="px-1.5 py-0.5 rounded bg-badge-gray-bg text-badge-gray-text font-medium">
        v{record.version}
      </span>
      <span className="font-mono text-text-muted truncate flex-1" title={record.source_file}>
        {record.source_file}
      </span>
      <span className="text-text-muted whitespace-nowrap">{formatDate(record.validated_at)}</span>
      <span
        className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
          record.discrepancy_count === 0
            ? 'bg-badge-green-bg text-badge-green-text'
            : 'bg-badge-orange-bg text-badge-orange-text'
        }`}
      >
        {record.discrepancy_count === 0 ? 'Clean' : `${record.discrepancy_count} discrepancies`}
      </span>
      {record.source_url && (
        <a
          href={record.source_url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-text-muted hover:text-primary transition-colors"
          title="View source file"
        >
          <ExternalLink size={12} />
        </a>
      )}
    </div>
  );
}

// ---- DatasetCard ----

function DatasetCard({ dataset }: { dataset: DatasetValidation }) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const hasIssues = dataset.phantom_count > 0 || dataset.undocumented_count > 0;

  return (
    <div className="border border-border/50 rounded-lg bg-surface">
      {/* Dataset header */}
      <div className="flex items-center gap-2 px-3 py-2.5">
        <span className="font-mono text-xs text-text flex-1 truncate" title={dataset.dataset_id}>
          {dataset.dataset_id}
        </span>
        <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-badge-teal-bg text-badge-teal-text">
          {dataset.validation_count} {dataset.validation_count === 1 ? 'validation' : 'validations'}
        </span>
        {dataset.validated && (
          <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-badge-green-bg text-badge-green-text">
            Validated
          </span>
        )}
        {dataset.phantom_count > 0 && (
          <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-badge-red-bg text-badge-red-text">
            {dataset.phantom_count} phantom
          </span>
        )}
        {dataset.undocumented_count > 0 && (
          <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-badge-blue-bg text-badge-blue-text">
            {dataset.undocumented_count} undocumented
          </span>
        )}
        {(hasIssues || dataset.validations.length > 0) && (
          <button
            onClick={() => setDetailsOpen(!detailsOpen)}
            className="text-xs text-text-muted hover:text-text transition-colors"
          >
            {detailsOpen ? 'Hide' : 'Details'}
          </button>
        )}
      </div>

      {/* Expandable details */}
      <AnimatePresence>
        {detailsOpen && (
          <motion.div
            variants={expandCollapse}
            initial="hidden"
            animate="visible"
            exit="hidden"
            className="border-t border-border/50"
          >
            <div className="px-3 py-2.5 space-y-3">
              {/* Phantom params */}
              {dataset.phantom_params.length > 0 && (
                <div>
                  <div className="text-[11px] font-medium text-text-muted mb-1">
                    Phantom parameters (in metadata, not in data)
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {dataset.phantom_params.map((p) => (
                      <span
                        key={p}
                        className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-badge-red-bg text-badge-red-text"
                      >
                        {p}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Undocumented params */}
              {dataset.undocumented_params.length > 0 && (
                <div>
                  <div className="text-[11px] font-medium text-text-muted mb-1">
                    Undocumented parameters (in data, not in metadata)
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {dataset.undocumented_params.map((p) => (
                      <span
                        key={p}
                        className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-badge-blue-bg text-badge-blue-text"
                      >
                        {p}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Validation records */}
              {dataset.validations.length > 0 && (
                <div>
                  <div className="text-[11px] font-medium text-text-muted mb-1">
                    Validation history
                  </div>
                  <div className="divide-y divide-border/30">
                    {dataset.validations.map((v, i) => (
                      <ValidationRecordRow key={i} record={v} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ---- MissionValidationCard ----

interface Props {
  mission: MissionValidation;
}

export function MissionValidationCard({ mission }: Props) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border border-border rounded-lg overflow-hidden">
      {/* Mission header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 bg-surface-elevated hover:bg-hover-bg transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-text">{mission.display_name}</span>
          <span className="text-xs text-text-muted">
            {mission.dataset_count} {mission.dataset_count === 1 ? 'dataset' : 'datasets'}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {mission.issue_count === 0 ? (
            <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-badge-green-bg text-badge-green-text">
              Clean
            </span>
          ) : (
            <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-badge-orange-bg text-badge-orange-text">
              {mission.issue_count} with issues
            </span>
          )}
          {expanded ? (
            <ChevronUp size={16} className="text-text-muted" />
          ) : (
            <ChevronDown size={16} className="text-text-muted" />
          )}
        </div>
      </button>

      {/* Expanded dataset list */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            variants={expandCollapse}
            initial="hidden"
            animate="visible"
            exit="hidden"
            className="border-t border-border"
          >
            <div className="p-3 space-y-2">
              {mission.datasets.map((ds) => (
                <DatasetCard key={ds.dataset_id} dataset={ds} />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
