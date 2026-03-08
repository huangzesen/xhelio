import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Key,
  Eye,
  EyeOff,
  Plus,
  Trash2,
  Check,
  X,
  Loader2,
  ChevronDown,
} from 'lucide-react';
import { listApiKeys, updateApiKey, deleteApiKey } from '../../api/client';
import type { ApiKeyEntry } from '../../api/types';
import { useTranslation } from 'react-i18next';

const KNOWN_PROVIDERS = [
  { id: 'gemini', name: 'Gemini', env: 'GOOGLE_API_KEY' },
  { id: 'openai', name: 'OpenAI', env: 'OPENAI_API_KEY' },
  { id: 'anthropic', name: 'Anthropic', env: 'ANTHROPIC_API_KEY' },
  { id: 'minimax', name: 'MiniMax', env: 'MINIMAX_API_KEY' },
  { id: 'grok', name: 'Grok', env: 'GROK_API_KEY' },
  { id: 'deepseek', name: 'DeepSeek', env: 'DEEPSEEK_API_KEY' },
  { id: 'qwen', name: 'Qwen', env: 'QWEN_API_KEY' },
  { id: 'kimi', name: 'Kimi', env: 'KIMI_API_KEY' },
  { id: 'glm', name: 'GLM', env: 'GLM_API_KEY' },
  { id: 'custom', name: 'CustomBot', env: 'CUSTOM_API_KEY' },
] as const;

const KNOWN_ENV_NAMES = new Set(KNOWN_PROVIDERS.map((p) => p.env));

function findKeyEntry(keys: ApiKeyEntry[], envName: string): ApiKeyEntry | undefined {
  return keys.find((k) => k.name === envName);
}

export function ApiKeysSection() {
  const { t } = useTranslation(['settings', 'common']);
  const [keys, setKeys] = useState<ApiKeyEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveResult, setSaveResult] = useState<{
    valid: boolean;
    error: string | null;
  } | null>(null);
  const [addDropdownOpen, setAddDropdownOpen] = useState(false);
  const [customMode, setCustomMode] = useState(false);
  const [customEnvName, setCustomEnvName] = useState('');

  const dropdownRef = useRef<HTMLDivElement>(null);

  const loadKeys = useCallback(async () => {
    try {
      const resp = await listApiKeys();
      setKeys(resp.keys);
    } catch {
      // silent — keys list will be empty
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadKeys();
  }, [loadKeys]);

  // Close dropdown on outside click
  useEffect(() => {
    if (!addDropdownOpen) return;
    function handleClick(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setAddDropdownOpen(false);
        setCustomMode(false);
        setCustomEnvName('');
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [addDropdownOpen]);

  const customKeys = keys.filter((k) => !KNOWN_ENV_NAMES.has(k.name));

  function toggleExpand(envName: string) {
    if (expandedKey === envName) {
      collapse();
    } else {
      setExpandedKey(envName);
      setEditValue('');
      setShowPassword(false);
      setSaveResult(null);
    }
  }

  function collapse() {
    setExpandedKey(null);
    setEditValue('');
    setShowPassword(false);
    setSaveResult(null);
  }

  const collapseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (collapseTimerRef.current) clearTimeout(collapseTimerRef.current);
    };
  }, []);

  async function handleSave(envName: string) {
    if (!editValue.trim()) return;
    setSaving(true);
    setSaveResult(null);
    try {
      // Find the provider id for known providers
      const known = KNOWN_PROVIDERS.find((p) => p.env === envName);
      const providerId = known ? known.id : 'custom';
      const nameParam = known ? undefined : envName;
      const result = await updateApiKey(providerId, editValue.trim(), nameParam);
      setSaveResult({ valid: result.valid, error: result.error });
      if (result.valid) {
        await loadKeys();
        // Keep expanded briefly to show success, then collapse
        if (collapseTimerRef.current) clearTimeout(collapseTimerRef.current);
        collapseTimerRef.current = setTimeout(() => collapse(), 1500);
      }
    } catch (err) {
      setSaveResult({
        valid: false,
        error: err instanceof Error ? err.message : 'Failed to save',
      });
    } finally {
      setSaving(false);
    }
  }

  async function handleRemove(envName: string) {
    setSaving(true);
    try {
      await deleteApiKey(envName);
      await loadKeys();
      collapse();
    } catch {
      setSaveResult({ valid: false, error: t('apiKeys.failedToRemove') });
    } finally {
      setSaving(false);
    }
  }

  function handleAddSelect(provider: (typeof KNOWN_PROVIDERS)[number]) {
    setAddDropdownOpen(false);
    setCustomMode(false);
    setCustomEnvName('');
    toggleExpand(provider.env);
  }

  function handleAddCustom() {
    if (!customEnvName.trim()) return;
    const envName = customEnvName.trim().toUpperCase().replace(/[^A-Z0-9_]/g, '_');
    setAddDropdownOpen(false);
    setCustomMode(false);
    setCustomEnvName('');
    toggleExpand(envName);
  }

  function renderKeyRow(
    envName: string,
    displayName: string,
    entry: ApiKeyEntry | undefined,
  ) {
    const configured = entry?.configured ?? false;
    const masked = entry?.masked ?? null;
    const isExpanded = expandedKey === envName;

    return (
      <div className="border-b border-border last:border-b-0">
        {/* Row */}
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-3 min-w-0">
            <span
              className={`w-2 h-2 rounded-full shrink-0 ${
                configured ? 'bg-status-success-text' : 'bg-border'
              }`}
            />
            <span className="text-sm text-text truncate">{displayName}</span>
            {displayName !== envName && (
              <span className="text-xs text-text-muted hidden sm:inline">{envName}</span>
            )}
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs text-text-muted">
              {configured && masked ? masked : t('common:status.notConfigured')}
            </span>
            <button
              onClick={() => toggleExpand(envName)}
              className="px-3 py-1.5 rounded-lg text-xs font-medium transition-colors bg-panel border border-border text-text-muted hover:text-text hover:bg-surface"
            >
              {configured ? t('common:actions.edit') : t('common:actions.add')}
            </button>
          </div>
        </div>

        {/* Expanded edit area */}
        <div className={`grid transition-all duration-200 ease-in-out ${isExpanded ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'}`}>
          <div className="overflow-hidden">
            <div className="px-4 pb-4 pt-1 space-y-3">
              {/* Password input */}
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={editValue}
                  onChange={(e) => {
                    setEditValue(e.target.value);
                    setSaveResult(null);
                  }}
                  placeholder={configured ? t('apiKeys.enterNewKey') : t('apiKeys.pasteApiKey')}
                  className="w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text pr-10 focus:outline-none focus:ring-1 focus:ring-primary"
                  autoFocus
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleSave(envName);
                    if (e.key === 'Escape') collapse();
                  }}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-text-muted hover:text-text transition-colors p-1"
                  tabIndex={-1}
                >
                  {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>

              {/* Action buttons */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleSave(envName)}
                  disabled={saving || !editValue.trim()}
                  className="px-3 py-2 rounded-lg bg-primary text-white text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-1.5"
                >
                  {saving ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <Check size={14} />
                  )}
                  {t('common:actions.save')}
                </button>
                {configured && (
                  <button
                    onClick={() => handleRemove(envName)}
                    disabled={saving}
                    className="px-3 py-2 rounded-lg border border-status-error-text/30 text-status-error-text text-sm font-medium hover:bg-status-error-bg disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-1.5"
                  >
                    <Trash2 size={14} />
                    {t('common:actions.remove')}
                  </button>
                )}
                <button
                  onClick={collapse}
                  className="text-text-muted hover:text-text transition-colors p-2"
                >
                  <X size={16} />
                </button>
              </div>

              {/* Save result feedback */}
              {saveResult && (
                <div
                  className={`text-sm px-3 py-2 rounded-lg ${
                    saveResult.valid
                      ? 'text-status-success-text bg-status-success-bg'
                      : 'text-status-error-text bg-status-error-bg'
                  }`}
                >
                  {saveResult.valid
                    ? t('apiKeys.keySaved')
                    : saveResult.error || t('apiKeys.invalidKey')}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 size={24} className="animate-spin text-text-muted" />
      </div>
    );
  }

  return (
    <div className="py-4 space-y-6">
      <div>
        <h2 className="text-lg font-medium text-text mb-1">{t('apiKeys.title')}</h2>
        <p className="text-xs text-text-muted">
          {t('apiKeys.description')}
        </p>
      </div>

      {/* Known providers list */}
      <div className="bg-panel rounded-xl border border-border overflow-hidden">
        {KNOWN_PROVIDERS.map((provider) => {
          const entry = findKeyEntry(keys, provider.env);
          return <div key={provider.env}>{renderKeyRow(provider.env, provider.name, entry)}</div>;
        })}
      </div>

      {/* Custom keys section */}
      {customKeys.length > 0 && (
        <>
          <div className="flex items-center gap-3">
            <div className="border-t border-border flex-1" />
            <span className="text-xs text-text-muted font-medium uppercase tracking-wider">
              {t('apiKeys.customKeys')}
            </span>
            <div className="border-t border-border flex-1" />
          </div>

          <div className="bg-panel rounded-xl border border-border overflow-hidden">
            {customKeys.map((entry) => (
              <div key={entry.name}>{renderKeyRow(entry.name, entry.name, entry)}</div>
            ))}
          </div>
        </>
      )}

      {/* Also show expanded custom key that doesn't exist yet */}
      {expandedKey &&
        !KNOWN_ENV_NAMES.has(expandedKey) &&
        !customKeys.some((k) => k.name === expandedKey) && (
          <>
            {customKeys.length === 0 && (
              <div className="flex items-center gap-3">
                <div className="border-t border-border flex-1" />
                <span className="text-xs text-text-muted font-medium uppercase tracking-wider">
                  {t('apiKeys.customKeys')}
                </span>
                <div className="border-t border-border flex-1" />
              </div>
            )}
            <div className="bg-panel rounded-xl border border-border overflow-hidden">
              {renderKeyRow(expandedKey, expandedKey, undefined)}
            </div>
          </>
        )}

      {/* Add API Key button */}
      <div className="relative" ref={dropdownRef}>
        <button
          onClick={() => {
            setAddDropdownOpen(!addDropdownOpen);
            setCustomMode(false);
            setCustomEnvName('');
          }}
          className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium text-text-muted hover:text-text hover:bg-panel border border-border transition-colors"
        >
          <Plus size={16} />
          {t('apiKeys.addApiKey')}
          <ChevronDown
            size={14}
            className={`transition-transform ${addDropdownOpen ? 'rotate-180' : ''}`}
          />
        </button>

        {addDropdownOpen && (
          <div className="absolute left-0 mt-1 w-64 bg-panel rounded-xl border border-border shadow-lg z-20 overflow-hidden">
            <div className="max-h-80 overflow-y-auto py-1">
              {KNOWN_PROVIDERS.map((provider) => {
                const entry = findKeyEntry(keys, provider.env);
                const configured = entry?.configured ?? false;
                return (
                  <button
                    key={provider.id}
                    onClick={() => handleAddSelect(provider)}
                    disabled={configured}
                    className={`w-full flex items-center gap-3 px-4 py-2.5 text-sm text-left transition-colors ${
                      configured
                        ? 'text-text-muted/50 cursor-not-allowed'
                        : 'text-text hover:bg-surface'
                    }`}
                  >
                    <span
                      className={`w-2 h-2 rounded-full shrink-0 ${
                        configured ? 'bg-status-success-text' : 'bg-border'
                      }`}
                    />
                    <span className="flex-1">{provider.name}</span>
                    {configured && (
                      <Check size={14} className="text-status-success-text" />
                    )}
                  </button>
                );
              })}

              {/* Divider */}
              <div className="border-t border-border my-1" />

              {/* Custom option */}
              {!customMode ? (
                <button
                  onClick={() => setCustomMode(true)}
                  className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-left text-text hover:bg-surface transition-colors"
                >
                  <Key size={14} className="text-text-muted" />
                  <span>{t('apiKeys.custom')}</span>
                </button>
              ) : (
                <div className="px-4 py-2.5 space-y-2">
                  <input
                    type="text"
                    value={customEnvName}
                    onChange={(e) => setCustomEnvName(e.target.value)}
                    placeholder={t('apiKeys.envVarPlaceholder')}
                    className="w-full rounded-lg border border-border px-3 py-1.5 text-sm bg-input-bg text-text focus:outline-none focus:ring-1 focus:ring-primary"
                    autoFocus
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleAddCustom();
                      if (e.key === 'Escape') {
                        setCustomMode(false);
                        setCustomEnvName('');
                      }
                    }}
                  />
                  <button
                    onClick={handleAddCustom}
                    disabled={!customEnvName.trim()}
                    className="w-full px-3 py-1.5 rounded-lg bg-primary text-white text-xs font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {t('common:actions.add')}
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
