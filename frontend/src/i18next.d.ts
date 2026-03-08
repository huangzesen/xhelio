import type commonEn from './locales/en/common.json';
import type settingsEn from './locales/en/settings.json';
import type setupEn from './locales/en/setup.json';

declare module 'i18next' {
  interface CustomTypeOptions {
    defaultNS: 'common';
    resources: {
      common: typeof commonEn;
      settings: typeof settingsEn;
      setup: typeof setupEn;
    };
  }
}
