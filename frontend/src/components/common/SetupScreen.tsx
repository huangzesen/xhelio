import { useState } from 'react';
import { HelionLogo } from './HelionLogo';
import { updateApiKey } from '../../api/client';
import { Loader2, Eye, EyeOff, ExternalLink, CheckCircle2, XCircle } from 'lucide-react';

interface Props {
  onComplete: () => void;
}

export function SetupScreen({ onComplete }: Props) {
  const [key, setKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [result, setResult] = useState<{ valid: boolean; error: string | null } | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!key.trim()) return;

    setSaving(true);
    setResult(null);

    try {
      const res = await updateApiKey(key.trim());
      setResult({ valid: res.valid, error: res.error });
      if (res.valid) {
        setTimeout(() => onComplete(), 1200);
      }
    } catch (err) {
      setResult({ valid: false, error: (err as Error).message });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div data-testid="setup-screen" className="h-full flex items-center justify-center bg-surface">
      <div className="w-full max-w-md px-6">
        <div className="bg-panel rounded-2xl border border-border p-8 shadow-lg">
          {/* Logo + Welcome */}
          <div className="flex flex-col items-center gap-3 mb-8">
            <HelionLogo size={48} className="text-primary" />
            <h1 className="text-xl font-semibold text-text">Welcome to xhelio</h1>
            <p className="text-sm text-text-muted text-center leading-relaxed">
              To get started, enter your Google Gemini API key.
              <br />
              <a
                href="https://ai.google.dev/"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-primary hover:underline mt-1"
              >
                Get an API key <ExternalLink size={12} />
              </a>
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <label className="block">
              <span className="text-xs text-text-muted">Gemini API Key</span>
              <div className="mt-1 relative">
                <input
                  data-testid="setup-api-key-input"
                  type={showKey ? 'text' : 'password'}
                  value={key}
                  onChange={(e) => { setKey(e.target.value); setResult(null); }}
                  placeholder="AIza..."
                  autoFocus
                  className="block w-full rounded-lg border border-border px-3 py-2.5 pr-10 text-sm
                    bg-input-bg text-text placeholder:text-text-muted/50
                    focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
                />
                <button
                  type="button"
                  onClick={() => setShowKey((v) => !v)}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 text-text-muted hover:text-text"
                  tabIndex={-1}
                >
                  {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </label>

            {/* Validation result */}
            {result && (
              <div className={`flex items-start gap-2 text-sm rounded-lg px-3 py-2.5 ${
                result.valid
                  ? 'bg-status-success-bg text-status-success-text'
                  : 'bg-status-error-bg text-status-error-text'
              }`}>
                {result.valid ? (
                  <>
                    <CheckCircle2 size={16} className="mt-0.5 shrink-0" />
                    <span>API key is valid. Starting up...</span>
                  </>
                ) : (
                  <>
                    <XCircle size={16} className="mt-0.5 shrink-0" />
                    <span>Invalid API key{result.error ? `: ${result.error}` : ''}</span>
                  </>
                )}
              </div>
            )}

            <button
              data-testid="setup-submit"
              type="submit"
              disabled={saving || !key.trim() || (result?.valid ?? false)}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl
                bg-primary text-white font-medium
                hover:bg-primary-dark transition-colors disabled:opacity-50"
            >
              {saving ? (
                <Loader2 size={16} className="animate-spin" />
              ) : result?.valid ? (
                <CheckCircle2 size={16} />
              ) : null}
              {saving ? 'Validating...' : result?.valid ? 'Saved' : 'Save & Continue'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
