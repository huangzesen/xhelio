# UI Control Skill

This skill provides tools to control, analyze, and debug the xhelio web UI.

## Usage

Load this skill using: `/skill ui-control`

## Available Commands

### Navigate to a page
```
/ui-navigate <path>
```
Navigates to a path in the xhelio app. Examples:
- `/ui-navigate /` - Chat page
- `/ui-navigate /data` - Data tools
- `/ui-navigate /settings` - Settings

### Capture screenshot
```
/ui-screenshot [label]
```
Captures a screenshot of the current viewport. Optional label for identification.

### Analyze page
```
/ui-analyze
```
Analyzes the current page and returns:
- URL and route
- Components count
- Forms, links, buttons, inputs
- Errors and warnings

### Find bugs
```
/ui-bugs
```
Detects and reports bugs in the current page including:
- Console errors
- Navigation issues
- State bugs
- Accessibility issues
- Performance problems

### Get page state
```
/ui-state
```
Returns current page URL, title, route, and loading state.

### Find elements
```
/ui-elements <selector>
```
Find all elements matching a CSS selector.

### Click element
```
/ui-click <selector>
```
Click an element by CSS selector.

### Fill input
```
/ui-fill <selector> <value>
```
Fill an input field with a value.

## Tool Functions

The following JavaScript functions are available in the frontend/tools directory:

```typescript
import { getBrowserController, closeBrowserController, createScreenshotTool, createUIAnalyzer, createBugDetector } from './tools/index';

// Launch browser (if not already running)
const controller = await getBrowserController({
  screenshotDir: './screenshots',
  headless: true
});

// Navigate to a URL
await controller.navigate('http://localhost:5173/');
await controller.navigate('http://localhost:5173/#/settings');

// Get page state
const state = await controller.getPageState();
console.log(state.url, state.route, state.title);

// Screenshot
const screenshotTool = await createScreenshotTool();
const screenshot = await screenshotTool.capture({ fullPage: true }, 'home-page');

// Analyze page
const analyzer = await createUIAnalyzer();
const analysis = await analyzer.analyzePage();
console.log(analysis.components, analysis.forms, analysis.errors);

// Find bugs
const bugDetector = await createBugDetector();
const bugs = await bugDetector.detectAll();
const report = bugDetector.generateReport(bugs);
```

## Example Workflows

### 1. Check settings page for bugs
```javascript
// In Claude Code
/ui-navigate /settings
/ui-screenshot settings-page
/ui-bugs
```

### 2. Analyze chat input functionality
```javascript
/ui-navigate /
/ui-analyze
/ui-elements input
/ui-click [data-testid="new-chat-btn"]
```

### 3. Full UI audit
```javascript
/ui-navigate /
/ui-screenshot chat-page
/ui-bugs
/ui-navigate /data
/ui-screenshot data-page
/ui-bugs
/ui-navigate /settings
/ui-screenshot settings-page
/ui-bugs
```

## Integration

This skill uses Playwright under the hood for browser automation. The browser is launched in headless mode by default and screenshots are saved to the `screenshots/` directory.
