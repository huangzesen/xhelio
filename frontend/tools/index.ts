export { BrowserController, getBrowserController, closeBrowserController, type BrowserControllerConfig, type NavigateOptions, type ElementInfo } from './browser-controller';

export { ScreenshotTool, createScreenshotTool, type ScreenshotOptions, type ScreenshotResult, type AnnotatedScreenshotResult, type Annotation } from './screenshot';

export { UIAnalyzer, createUIAnalyzer, type PageAnalysis, type UIComponent, type FormInfo, type LinkInfo, type ButtonInfo, type InputInfo, type AccessibilityIssue } from './analyzer';

export { BugDetector, createBugDetector, type BugReport, type NavigationBug, type StateBug } from './bug-detector';
