import { useState, useEffect, useCallback } from 'react';
import { type ExportFormat } from '../utils/exportSession';

interface ExportSettings {
  format: ExportFormat;
  localPath: string;
}

const STORAGE_KEY = 'xhelio_export_settings';

const DEFAULT_SETTINGS: ExportSettings = {
  format: 'base64',
  localPath: 'export-images',
};

export function useExportSettings() {
  const [settings, setSettings] = useState<ExportSettings>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? { ...DEFAULT_SETTINGS, ...JSON.parse(stored) } : DEFAULT_SETTINGS;
    } catch {
      return DEFAULT_SETTINGS;
    }
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  }, [settings]);

  const setFormat = useCallback((format: ExportFormat) => {
    setSettings((s) => ({ ...s, format }));
  }, []);

  const setLocalPath = useCallback((path: string) => {
    setSettings((s) => ({ ...s, localPath: path }));
  }, []);

  return { settings, setFormat, setLocalPath };
}
