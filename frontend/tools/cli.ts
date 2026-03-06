#!/usr/bin/env npx tsx
import { parseArgs } from 'util';
import { getBrowserController, closeBrowserController, ScreenshotTool, UIAnalyzer, BugDetector } from './index.js';

interface CliArgs {
  command: string;
  url?: string;
  selector?: string;
  text?: string;
  value?: string;
  label?: string;
  viewport?: string;
  fullPage?: boolean;
  options?: Record<string, any>;
}

async function main() {
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    printHelp();
    process.exit(1);
  }

  const command = args[0];
  
  try {
    switch (command) {
      case 'launch':
        await handleLaunch();
        break;
        
      case 'navigate':
      case 'goto':
        await handleNavigate(args);
        break;
        
      case 'click':
        await handleClick(args);
        break;
        
      case 'fill':
        await handleFill(args);
        break;
        
      case 'screenshot':
        await handleScreenshot(args);
        break;
        
      case 'analyze':
        await handleAnalyze(args);
        break;
        
      case 'bugs':
        await handleBugs(args);
        break;
        
      case 'state':
        await handleState(args);
        break;
        
      case 'elements':
        await handleElements(args);
        break;
        
      case 'close':
        await closeBrowserController();
        console.log('Browser closed');
        break;
        
      case 'help':
        printHelp();
        break;
        
      default:
        console.error(`Unknown command: ${command}`);
        printHelp();
        process.exit(1);
    }
  } catch (error) {
    console.error('Error:', error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

async function handleLaunch() {
  const controller = await getBrowserController();
  console.log('Browser launched');
  console.log('Screenshot directory:', controller.getScreenshotDir());
}

async function handleNavigate(args: string[]) {
  const url = args[1];
  if (!url) {
    console.error('Usage: navigate <url>');
    process.exit(1);
  }
  
  const controller = await getBrowserController();
  await controller.navigate(url);
  console.log(`Navigated to: ${url}`);
  
  const state = await controller.getPageState();
  console.log('Page title:', state.title);
  console.log('Route:', state.route);
}

async function handleClick(args: string[]) {
  const selector = args[1];
  if (!selector) {
    console.error('Usage: click <selector>');
    process.exit(1);
  }
  
  const controller = await getBrowserController();
  await controller.click(selector);
  console.log(`Clicked: ${selector}`);
}

async function handleFill(args: string[]) {
  const selector = args[1];
  const value = args[2];
  
  if (!selector || value === undefined) {
    console.error('Usage: fill <selector> <value>');
    process.exit(1);
  }
  
  const controller = await getBrowserController();
  await controller.fill(selector, value);
  console.log(`Filled ${selector} with: ${value}`);
}

async function handleScreenshot(args: string[]) {
  const label = args[1];
  const fullPage = args.includes('--full-page');
  
  const controller = await getBrowserController();
  const screenshotTool = new ScreenshotTool(controller);
  
  const result = fullPage 
    ? await screenshotTool.captureFullPage(label)
    : await screenshotTool.captureViewport(label);
    
  console.log('Screenshot saved:', result.relativePath);
}

async function handleAnalyze(args: string[]) {
  const controller = await getBrowserController();
  const analyzer = new UIAnalyzer(controller);
  
  const analysis = await analyzer.analyzePage();
  
  console.log('\n=== Page Analysis ===\n');
  console.log('URL:', analysis.url);
  console.log('Route:', analysis.route);
  console.log('Title:', analysis.title);
  console.log('\nComponents:', analysis.components.length);
  console.log('Forms:', analysis.forms.length);
  console.log('Links:', analysis.links.length);
  console.log('Buttons:', analysis.buttons.length);
  console.log('Inputs:', analysis.inputs.length);
  
  if (analysis.errors.length > 0) {
    console.log('\nErrors:', analysis.errors);
  }
  if (analysis.warnings.length > 0) {
    console.log('\nWarnings:', analysis.warnings);
  }
  
  if (args.includes('--verbose')) {
    console.log('\n=== Components ===\n');
    analysis.components.forEach(c => {
      console.log(`- ${c.type}${c.id ? `#${c.id}` : ''}: ${c.label || c.text || 'no label'}`);
    });
  }
}

async function handleBugs(args: string[]) {
  const controller = await getBrowserController();
  const analyzer = new UIAnalyzer(controller);
  const detector = new BugDetector(controller, analyzer);
  
  const bugs = await detector.detectAll();
  const report = detector.generateReport(bugs);
  console.log(report);
}

async function handleState(args: string[]) {
  const controller = await getBrowserController();
  const state = await controller.getPageState();
  
  console.log('\n=== Page State ===\n');
  console.log('URL:', state.url);
  console.log('Title:', state.title);
  console.log('Route:', state.route);
  console.log('Loading:', state.loading);
}

async function handleElements(args: string[]) {
  const selector = args[1];
  if (!selector) {
    console.error('Usage: elements <selector>');
    process.exit(1);
  }
  
  const controller = await getBrowserController();
  const elements = await controller.getAllElements(selector);
  
  console.log(`\nFound ${elements.length} elements matching: ${selector}\n`);
  elements.forEach((el, i) => {
    console.log(`${i + 1}. <${el.tag}>${el.id ? `#${el.id}` : ''} ${el.classes ? `.${el.classes.split(' ').join('.')}` : ''}`);
    if (el.text) console.log(`   Text: ${el.text.substring(0, 50)}`);
    if (el.boundingBox) console.log(`   Position: (${el.boundingBox.x}, ${el.boundingBox.y}) ${el.boundingBox.width}x${el.boundingBox.height}`);
  });
}

function printHelp() {
  console.log(`
UI Control CLI

Usage: ui-control <command> [options]

Commands:
  launch              Launch browser
  navigate <url>      Navigate to URL
  goto <url>          Navigate to URL (alias)
  click <selector>    Click element
  fill <selector> <value>  Fill input
  screenshot [label] Capture screenshot
  analyze             Analyze current page
  bugs                Detect bugs
  state               Get page state
  elements <selector> Find elements
  close               Close browser
  help                Show this help

Options:
  --full-page         Capture full page screenshot
  --verbose           Verbose output
  `);
}

main();
