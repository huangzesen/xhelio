import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

// English
import commonEn from './locales/en/common.json';
import settingsEn from './locales/en/settings.json';
import setupEn from './locales/en/setup.json';

// Chinese (Simplified)
import commonZhCN from './locales/zh-CN/common.json';
import settingsZhCN from './locales/zh-CN/settings.json';
import setupZhCN from './locales/zh-CN/setup.json';

export const supportedLanguages = [
  { code: 'en', name: 'English' },
  { code: 'zh-CN', name: '简体中文' },
] as const;

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: {
      en: {
        common: commonEn,
        settings: settingsEn,
        setup: setupEn,
      },
      'zh-CN': {
        common: commonZhCN,
        settings: settingsZhCN,
        setup: setupZhCN,
      },
    },
    supportedLngs: ['en', 'zh-CN'],
    fallbackLng: 'en',
    defaultNS: 'common',
    ns: ['common', 'settings', 'setup'],
    interpolation: {
      escapeValue: false, // React already escapes
    },
    detection: {
      order: ['localStorage', 'navigator'],
      lookupLocalStorage: 'xhelio-language',
      caches: ['localStorage'],
    },
  });

export default i18n;
