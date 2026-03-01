import type { Page, Locator } from '@playwright/test';
import type { MockSSEEvent } from '../mocks/sse-scenarios';

/**
 * Page object for the Chat interface — input, messages, example prompts, SSE control.
 */
export class ChatPage {
  readonly page: Page;
  readonly chatContainer: Locator;
  readonly chatInput: Locator;
  readonly sendButton: Locator;
  readonly stopButton: Locator;
  readonly messageList: Locator;
  readonly examplePrompts: Locator;
  readonly newChatButton: Locator;
  readonly sessionList: Locator;

  constructor(page: Page) {
    this.page = page;
    this.chatContainer = page.getByTestId('chat-container');
    this.chatInput = page.getByTestId('chat-input');
    this.sendButton = page.getByTestId('chat-send-btn');
    this.stopButton = page.getByTestId('chat-stop-btn');
    this.messageList = page.getByTestId('message-list');
    this.examplePrompts = page.getByTestId('example-prompts');
    this.newChatButton = page.getByTestId('new-chat-btn');
    this.sessionList = page.getByTestId('session-list');
  }

  /** Type a message into the chat input */
  async typeMessage(text: string) {
    await this.chatInput.fill(text);
  }

  /** Type and send a message */
  async sendMessage(text: string) {
    await this.chatInput.fill(text);
    await this.sendButton.click();
  }

  /** Get all message elements for a given role */
  getMessages(role: 'user' | 'agent' | 'system') {
    return this.page.getByTestId(`message-${role}`);
  }

  /** Wait for at least one agent message to appear */
  async waitForAgentMessage(timeout = 5000) {
    await this.getMessages('agent').first().waitFor({ state: 'visible', timeout });
  }

  /** Click an example prompt card by index (0-based) */
  async clickExamplePrompt(index: number) {
    await this.page.getByTestId('example-prompt').nth(index).click();
  }

  /** Get the number of example prompt cards */
  async getExamplePromptCount() {
    return this.page.getByTestId('example-prompt').count();
  }

  /** Check if the command dropdown is visible */
  async isCommandDropdownVisible() {
    // The command dropdown is a div inside ChatInput that appears when typing '/'
    return this.page.locator('[data-testid="chat-input"]').locator('..').locator('..').locator('.absolute.bottom-full').isVisible();
  }

  /** Emit a single SSE event via the mock */
  async emitSSE(type: string, data: Record<string, unknown>) {
    await this.page.evaluate(
      ({ type, data }) => {
        (window as any).__mockSSE.emit(type, data);
      },
      { type, data },
    );
  }

  /** Emit a sequence of SSE events with a delay between them */
  async emitSSEScenario(events: MockSSEEvent[], delayMs = 50) {
    for (const event of events) {
      await this.emitSSE(event.type, event.data);
      if (delayMs > 0) {
        await this.page.waitForTimeout(delayMs);
      }
    }
  }

  /** Get the slash command dropdown entries */
  getCommandDropdownItems() {
    return this.page.locator('.absolute.bottom-full button');
  }

  /** Click "New Chat" in the sidebar */
  async clickNewChat() {
    await this.newChatButton.click();
  }
}
