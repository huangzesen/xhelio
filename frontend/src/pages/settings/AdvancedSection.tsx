import { useSettingsStore } from '../../stores/settingsStore';
import { useTranslation } from 'react-i18next';
import { supportedLanguages } from '../../i18n';

export function AdvancedSection() {
  const { t, i18n } = useTranslation(['settings', 'common']);
  const { config, updateConfig } = useSettingsStore();
  const descriptions = (config._descriptions as Record<string, string>) ?? {};

  const setField = (key: string, value: unknown) => {
    updateConfig({ [key]: value });
  };

  const setNestedField = (parent: string, key: string, value: unknown) => {
    const current = (config[parent] as Record<string, unknown>) ?? {};
    updateConfig({ [parent]: { ...current, [key]: value } });
  };

  const reasoning = (config.reasoning as Record<string, unknown>) ?? {};

  return (
    <div className="py-4 space-y-8">
      <div>
        <h2 className="text-lg font-medium text-text mb-1">{t('advanced.title')}</h2>
        <p className="text-sm text-text-muted">{t('advanced.description')}</p>
      </div>

      {/* Language subsection */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-text-muted uppercase tracking-wider">
          {t('language.title')}
        </h3>
        <p className="text-xs text-text-muted">{t('language.description')}</p>
        <label className="block">
          <span className="text-xs text-text-muted">{t('language.label')}</span>
          <select
            value={i18n.language}
            onChange={(e) => i18n.changeLanguage(e.target.value)}
            className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
          >
            {supportedLanguages.map((lang) => (
              <option key={lang.code} value={lang.code}>
                {lang.name}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* Memory subsection */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-text-muted uppercase tracking-wider">{t('advanced.memory')}</h3>

        {([
          ['memory_token_budget', t('advanced.memoryTokenBudget'), 100000],
          ['ops_library_max_entries', t('advanced.opsLibraryMaxEntries'), 50],
        ] as const).map(([key, label, def]) => (
          <label key={key} className="block">
            <span className="text-xs text-text-muted">{label}</span>
            <input
              type="number"
              min={1}
              value={(config[key] as number) ?? def}
              onChange={(e) => setField(key, parseInt(e.target.value))}
              className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
            />
            {descriptions[key] && (
              <p className="mt-1.5 text-xs italic text-text-muted/70 leading-relaxed">{descriptions[key]}</p>
            )}
          </label>
        ))}

        <label className="block">
          <span className="text-xs text-text-muted">{t('advanced.memoryExtractionInterval')}</span>
          <input
            type="number"
            min={0}
            value={(config.memory_extraction_interval as number) ?? 2}
            onChange={(e) => setField('memory_extraction_interval', parseInt(e.target.value))}
            className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
          />
          {descriptions['memory_extraction_interval'] && (
            <p className="mt-1.5 text-xs italic text-text-muted/70 leading-relaxed">{descriptions['memory_extraction_interval']}</p>
          )}
        </label>

      </div>

      {/* Eureka subsection */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-text-muted uppercase tracking-wider">
          {t('advanced.eureka', 'Eureka Discovery')}
        </h3>

        <div>
          <label className="flex items-center justify-between cursor-pointer">
            <span className="text-xs text-text-muted">{t('advanced.eurekaEnabled', 'Enable Eureka Discovery')}</span>
            <input
              type="checkbox"
              checked={(config.eureka_enabled as boolean) ?? true}
              onChange={(e) => setField('eureka_enabled', e.target.checked)}
              className="rounded"
            />
          </label>
          {descriptions['eureka_enabled'] && (
            <p className="mt-1 text-xs italic text-text-muted/70 leading-relaxed">{descriptions['eureka_enabled']}</p>
          )}
        </div>

        <div>
          <label className="flex items-center justify-between cursor-pointer">
            <span className="text-xs text-text-muted">{t('advanced.eurekaMode', 'Eureka Mode (auto-execute suggestions)')}</span>
            <input
              type="checkbox"
              checked={(config.eureka_mode as boolean) ?? false}
              onChange={(e) => setField('eureka_mode', e.target.checked)}
              className="rounded"
            />
          </label>
          {descriptions['eureka_mode'] && (
            <p className="mt-1 text-xs italic text-text-muted/70 leading-relaxed">{descriptions['eureka_mode']}</p>
          )}
        </div>

        {(config.eureka_mode as boolean) && (
          <label className="block">
            <span className="text-xs text-text-muted">{t('advanced.eurekaMaxRounds', 'Max consecutive Eureka rounds')}</span>
            <input
              type="number"
              min={1}
              max={50}
              value={(config.eureka_max_rounds as number) ?? 5}
              onChange={(e) => setField('eureka_max_rounds', parseInt(e.target.value) || 5)}
              className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
            />
            {descriptions['eureka_max_rounds'] && (
              <p className="mt-1.5 text-xs italic text-text-muted/70 leading-relaxed">{descriptions['eureka_max_rounds']}</p>
            )}
          </label>
        )}
      </div>

      {/* Reasoning subsection */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-text-muted uppercase tracking-wider">{t('advanced.reasoning')}</h3>

        {([
          ['observation_summaries', t('advanced.observationSummaries'), true],
          ['self_reflection', t('advanced.selfReflection'), true],
          ['show_thinking', t('advanced.showThinking'), false],
          ['insight_feedback', t('advanced.insightFeedback'), false],
          ['async_delegation', t('advanced.asyncDelegation'), false],
        ] as const).map(([key, label, def]) => {
          const val = (reasoning[key] as boolean) ?? def;
          const desc = descriptions[`reasoning.${key}`];
          return (
            <div key={key}>
              <label className="flex items-center justify-between cursor-pointer">
                <span className="text-xs text-text-muted">{label}</span>
                <input
                  type="checkbox"
                  checked={val}
                  onChange={(e) => setNestedField('reasoning', key, e.target.checked)}
                  className="rounded"
                />
              </label>
              {desc && (
                <p className="mt-1 text-xs italic text-text-muted/70 leading-relaxed">{desc}</p>
              )}
            </div>
          );
        })}

        {(reasoning.insight_feedback as boolean) && (
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-text-muted">{t('advanced.insightFeedbackMaxIterations')}</span>
              <span className="text-xs text-text-muted">{(reasoning.insight_feedback_max_iterations as number) ?? 2}</span>
            </div>
            <input
              type="range"
              min={1}
              max={5}
              value={(reasoning.insight_feedback_max_iterations as number) ?? 2}
              onChange={(e) => setNestedField('reasoning', 'insight_feedback_max_iterations', parseInt(e.target.value))}
              className="w-full mt-1"
            />
            {descriptions['reasoning.insight_feedback_max_iterations'] && (
              <p className="mt-1 text-xs italic text-text-muted/70 leading-relaxed">{descriptions['reasoning.insight_feedback_max_iterations']}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
